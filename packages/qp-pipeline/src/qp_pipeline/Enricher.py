"""
Enricher.py - Production-Grade Enrichment Pipeline (Optimized Two-Pass)

This module implements an optimized two-pass enrichment pipeline for extracting
high-quality question-answer (QA) pairs and metadata from text "chunks".
It is designed to be robust and practical for production use:

High-level design:
- Pass 1 (batched): Produce candidate questions and source quotes from a chunk.
  This is intentionally grouped into a single (batched) LLM call per chunk to
  keep costs and latency predictable.
- Quick guards: Before invoking the more expensive Pass 2 answer generation,
  we run fast heuristic checks (quote matching via fuzzy similarity) to filter
  out weak candidates.
- Pass 2 (per-question): Generate a reference answer grounded in the chunk.
- Validation: Validate answers semantically and lexically using sentence
  embeddings and heuristics. Only validated QA pairs are persisted and added to
  a vector store.

Key behaviors / safeguards:
- Anti-bleed prompt fencing: system and user prompts instruct the model to only
  use the target chunk for question/answer generation.
- Token-sensitive behavior: very small chunks get fewer questions to avoid
  hallucination.
- Fast quote guard: avoids expensive second-pass generation if the quoted text
  doesn't match the chunk closely enough.
- Answer validation: semantic similarity + lexical overlap thresholds prevent
  irrelevant or hallucinated reference answers from being accepted.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Set, Tuple

# External deps used for embeddings and fuzzy matching
from openai import OpenAI
from rapidfuzz import fuzz
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sentence_transformers import SentenceTransformer, util

# ---------------- ROBUST PROJECT ROOT DETECTION ----------------
# Resolve project root in a robust way to support different execution contexts.
# If the environment provides a RAG_PROJECT_ROOT, prefer it; otherwise walk up
# the filesystem tree a fixed number of times to find the repository root.
_env_root = os.environ.get("RAG_PROJECT_ROOT")
if _env_root:
    project_root = Path(_env_root).resolve()
else:
    # We expect this file to live several directories inside the repository.
    # This relative traversal targets the repository root reliably for local runs.
    project_root = Path(__file__).resolve().parents[4]

# Ensure the project root is on sys.path so we can import local packages.
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Paths for local persistence (SQLite DB and Chroma vector store)
DB_PATH = str(project_root / "data" / "rag_staging.db")
CHROMA_DIR = str(project_root / "data" / "chroma_store")

# Create the directories if they don't already exist. This makes the module
# safe to run on a fresh checkout or container.
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

# Import project-local DB and Vector store abstractions. If imports fail, print
# a short message — the runtime will raise the ImportError later when functionality
# is required.
try:
    from qp_core.DBManager import DBManager
    from qp_core.VectorStore import QAVectorStore
except ImportError:
    print("Import errors")

# ---------------- CONFIG ----------------
# Core LLM endpoint settings and model selection
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "lfm-2.5-1.2b"

# Threading and matching thresholds
MAX_WORKERS = 4
QUOTE_MATCH_THRESHOLD = 75.0  # fuzzy partial ratio threshold for quote->chunk matching

# Map human difficulty levels to internal question types (used when saving)
DIFFICULTY_TYPE = {
    "Easy": "Fact",
    "Medium": "Mechanism",
    "Hard": "Critical",
}

# ---------------- RICH LOGGING SETUP ----------------
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("Enricher")


# ---------------- QUESTION LIMITS ----------------
def get_questions_per_difficulty(estimated_tokens: int) -> int:
    """
    Decide how many questions to generate per difficulty bucket based on the
    estimated token length of the chunk.

    Rationale:
    - Very small chunks (e.g. short sentences) should not produce multiple
      questions per difficulty because the risk of inventing unsupported
      content grows.
    - For normal-sized chunks, allow up to 3 questions per difficulty.

    Args:
        estimated_tokens: approximate size of the chunk in tokens (int)

    Returns:
        int: number of questions to request per difficulty level
    """
    return 1 if estimated_tokens < 80 else 3


# ---------------- ANSWER VALIDATOR ----------------
class AnswerValidator:
    """
    Validate a candidate reference answer against the original chunk.

    The validator uses a combination of:
    - Structural heuristics (sentence count constraints)
    - Embedding-based semantic similarity (cosine similarity threshold)
    - Lexical overlap (simple content word overlap)
    - Question-type specific checks (e.g., "why" questions require causal phrasing)

    These combined checks reduce false positives and ensure answers are grounded.
    """

    # Tunable thresholds for validation
    SIMILARITY_THRESHOLD = 0.52  # cosine similarity threshold (embedding-based)
    LEXICAL_OVERLAP_THRESHOLD = (
        0.25  # fraction of answer content words overlapping chunk
    )
    MIN_SENTENCES = 1
    MAX_SENTENCES = 5

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize embedding model used for semantic similarity checks.

        Args:
            model_name: name of the sentence-transformers compatible model
        """
        logger.info(f"Loading Validator Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def validate(self, question: str, answer: str, chunk: str) -> Tuple[bool, str]:
        """
        Validate an answer.

        Returns:
            (is_valid: bool, reason: str). If is_valid is False, reason explains why.
        """
        # Reject empty answers quickly
        if not answer or not answer.strip():
            return False, "Empty answer"

        # Structural check (enforce answer length / sentence count heuristics)
        struct_ok, sent_count = self._structural_check(answer)
        if not struct_ok:
            return False, f"Structural fail: {sent_count} sentences"

        # Semantic similarity using embeddings (chunk vs answer)
        embeddings = self.model.encode([chunk, answer], convert_to_tensor=True)
        score = float(util.cos_sim(embeddings[0], embeddings[1]))
        if score < self.SIMILARITY_THRESHOLD:
            return False, f"Semantic fail: cos={score:.3f}"

        # Lexical overlap: ensure the answer shares reasonable content words
        c_words = self._content_words(chunk)
        a_words = self._content_words(answer)
        if not a_words:
            return False, "No content words"
        overlap = len(c_words & a_words) / len(a_words)
        if overlap < self.LEXICAL_OVERLAP_THRESHOLD:
            return False, f"Lexical fail: overlap={overlap:.3f}"

        # Heuristic: "why" or "how" questions should contain causal/mechanical phrasing.
        # Using .split() for single-word tokens prevents "by" matching inside "thereby".
        if question.strip().lower().startswith(("why", "how")):
            causal = {
                "because",
                "due to",
                "as a result",
                "therefore",
                "leads to",
                "causes",
                "by",  # captures "by selecting", "by training"
                "through",  # captures "through model disagreement"
                "allows",
                "enables",
                "since",
            }
            answer_words = set(answer.lower().split())
            if not any(
                w in answer.lower() if " " in w else w in answer_words for w in causal
            ):
                return False, "Why/How-question without causal/mechanism phrase"

        # Passed all checks
        return True, ""

    def _sentence_count(self, text: str) -> int:
        """
        Count sentences in a permissive way. Short fragments (<= 5 chars) are ignored.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return len([s for s in sentences if len(s) > 5])

    def _content_words(self, text: str) -> Set[str]:
        """
        Extract "content words" using a combined regex:
        - Alphabetic tokens >= 4 chars (general terms)
        - Numeric tokens with optional unit suffixes (e.g. 60%, 15ms, 128blocks)
        The numeric part prevents precise short answers like "60%" or "15ms" from
        being incorrectly rejected due to no content word overlap.
        """
        tokens = re.findall(r"[a-zA-Z]{4,}|\d+[%a-zA-Z]*", text)
        return {t.lower() for t in tokens}

    def _structural_check(self, text: str) -> Tuple[bool, int]:
        """
        Ensure the number of sentences in `text` falls within MIN_SENTENCES .. MAX_SENTENCES.
        Returns (ok: bool, count: int).
        """
        count = self._sentence_count(text)
        return (self.MIN_SENTENCES <= count <= self.MAX_SENTENCES), count


# ---------------- LLM CLIENT ----------------
class LLMClient:
    """
    Encapsulates calls to the LLM backend.

    Responsibilities:
    - Provide helper to extract JSON safely from model output (robust to fences).
    - Encapsulate prompts for metadata, question candidate generation (Pass 1),
      and reference answer generation (Pass 2).
    """

    def __init__(self, base_url: str, api_key: str = "no-key"):
        # Wrap the OpenAI-compatible client with the configured endpoint.
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract JSON object from LLM text. Handles fenced ```json blocks
        and attempts fallback extraction if fences are not provided.

        On failure, log an error and return an empty dict (so callers can handle).
        """
        try:
            # Prefer explicit ```json fenced code blocks
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Otherwise try to find the first top-level {...} block
            braces_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if braces_match:
                return json.loads(braces_match.group(1))
            # Last resort: try parsing the entire text
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON Parse Error: {e} | Raw: {text[:100]}...")
            return {}

    def _call_model(self, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Portable wrapper for calling the LLM chat-completions endpoint and
        returning the parsed JSON result from into a Python dict.

        The function expects the model to return a JSON payload (fenced or raw).
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return self._extract_json(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

    def generate_metadata(self, text: str, context: str) -> Dict[str, Any]:
        """
        Generate structured metadata for a text chunk.

        Expected output schema (JSON):
        {
          "summary": "2-3 sentence summary starting with the main subject",
          "tags": ["NounTag1", "NounTag2", ...],
          "triplets": [
            {"subject": "Entity", "predicate": "predicate", "object": "Entity"}
          ]
        }

        The context is provided for better understanding but the prompt explicitly
        instructs the model to output only the structured JSON.
        """
        sys_prompt = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "Always output valid JSON following the exact schema below."
        )
        user_prompt = f"""
        ### CONTEXT (for understanding only):
            {context}

        ### TEXT CHUNK:
            {text}

        ### TASK:
        Return ONLY this JSON structure:
        {{
        "summary": "2-3 sentence summary starting with the main subject",
        "tags": ["NounTag1", "NounTag2", ...],
        "triplets": [
            {{"subject": "Entity", "predicate": "predicate", "object": "Entity"}}
        ]
        }}
        """
        return self._call_model(sys_prompt, user_prompt)

    # ====================== PASS 1 - PER-DIFFICULTY CALLS ======================
    def generate_question_candidates(self, chunk: Dict, context_str: str) -> List[Dict]:
        """
        Generate candidate questions and verbatim source quotes from the chunk.

        Three separate LLM calls, one per difficulty level, rather than a single
        batched call. A single call asking for 12 QA pairs overwhelms the 1.2B
        model's generation capacity and it stops early. Focused per-difficulty
        calls reliably produce the full quota and allow difficulty-specific
        instruction framing. Difficulty label is enforced after each call to
        prevent the model from drifting the label.
        """
        estimated_tokens = chunk.get("estimated_tokens", 150)
        per_diff = get_questions_per_difficulty(estimated_tokens)

        difficulty_instructions = {
            "Easy": "factual recall — ask about specific definitions, names, or stated facts",
            "Medium": "conceptual / mechanism — ask how or why something works",
            "Hard": "analytical / critical — ask about limitations, trade-offs, or comparisons",
        }

        sys_prompt = "You are an Expert Technical Interviewer. Output ONLY valid JSON."
        all_candidates: List[Dict] = []

        for difficulty, instruction in difficulty_instructions.items():
            user_prompt = f"""
### BACKGROUND CONTEXT (for understanding only — DO NOT use this for questions):
{context_str}

### TARGET TEXT CHUNK (You MUST extract questions and quotes ONLY from here):
{chunk["content"]}

### TASK:
Generate up to {per_diff} '{difficulty}' questions ({instruction}).
Questions must be EXCLUSIVELY answerable from the TARGET TEXT CHUNK above.
If the TARGET TEXT CHUNK is too short to answer a question, do not invent information.

### STRICT RULES:
1. Every "question" MUST be answerable using ONLY the TARGET TEXT CHUNK.
2. Every "source_quote" MUST be copied verbatim from the TARGET TEXT CHUNK (≥25 chars).
   Never pull quotes from the Background Context.
3. Never generate answers here.

Output ONLY:
{{"qa_pairs": [ {{"question": "...", "source_quote": "...", "difficulty": "{difficulty}"}}, ... ] }}
"""
            result = self._call_model(sys_prompt, user_prompt)
            raw = result.get("qa_pairs", [])

            count = 0
            for item in raw:
                if count >= per_diff:
                    break
                item["difficulty"] = difficulty  # enforce correct label
                all_candidates.append(item)
                count += 1

            logger.debug(f"Pass 1 [{difficulty}]: {count} candidate(s)")

        return all_candidates

    # ====================== PASS 2 ======================
    def generate_reference_answer(self, question: str, chunk_content: str) -> str:
        """
        Generate a concise reference answer to `question` using ONLY `chunk_content`.

        Prompt is intentionally less punitive than a strict "not enough info" framing.
        On small models (e.g. LFM 1.2B) an overly punitive prompt causes the model to
        defensively refuse questions that ARE supported by the text. Instead we instruct
        explicit fact extraction and only allow refusal if the core subject is completely
        absent.
        """
        sys_prompt = "You are a technical document analyzer. Extract facts directly from the text."
        user_prompt = f"""
### SOURCE TEXT:
{chunk_content}

### QUESTION:
{question}

### INSTRUCTIONS:
- Analyze the Source Text carefully to find the answer to the Question.
- Your answer MUST be a concise, complete sentence.
- If the text specifies a numeric threshold, percentage, or specific condition
  (e.g., 60%, 15ms, 128 blocks), you MUST include it verbatim in your answer.
- If, and ONLY IF, the core subject of the question is completely absent from
  the text, output EXACTLY: NOT_ENOUGH_INFORMATION
"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return ""


# ---------------- ENRICHMENT MANAGER ----------------
class EnrichmentManager:
    """
    Orchestrates the end-to-end pipeline: read chunks from DB, run the two-pass
    enrichment, validate, persist metadata and QA pairs, and add embeddings into
    a vector store for downstream retrieval.

    Responsibilities:
    - Iterate files with pending enrichment
    - Process each chunk: metadata -> candidate questions -> reference answers
      -> validation -> persist and vectorize
    - Use a small history window to provide contextual prompts for adjacent prose
      chunks while avoiding leakage when chunk types differ (e.g., code/table).
    """

    def __init__(self):
        logger.info("🚀 Starting Optimized Two-Pass Pipeline...")
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)
        self.validator = AnswerValidator()
        self.vector_store = QAVectorStore(chroma_path=CHROMA_DIR)

    def process_chunk(self, chunk: Dict, context_str: str) -> Tuple[bool, str]:
        """
        Process a single chunk through metadata generation, QA candidate generation,
        answer generation, validation, and persistence.

        Returns:
            (success: bool, summary: str) where summary is the emitted metadata summary
            (used as history for contextual prompts on later chunks).
        """
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        tokens = chunk.get("estimated_tokens", 0)
        ctype = chunk.get("content_type", "prose")

        console.rule(
            f"[bold cyan]Chunk {chunk_id[:8]}[/] | {tokens} tokens | type={ctype}"
        )

        # 1) Metadata generation
        logger.info("[bold]\\[1/3] Generating metadata...[/]")
        meta = self.llm.generate_metadata(content, context_str)
        if not meta:
            logger.error("Metadata generation FAILED — skipping chunk")
            self.db.save_rejections(
                chunk_id, [{"reason": "Metadata generation failed"}]
            )
            return False, ""

        logger.info(
            f"[green]✓[/] Metadata OK | "
            f"tags=[cyan]{meta.get('tags', [])}[/] | "
            f"{len(meta.get('triplets', []))} triplets"
        )

        # 1.5) Small chunk skip
        if tokens < 30:
            logger.warning(f"Chunk too small ({tokens} tokens) — skipping QA")
            self.db.save_enrichment(chunk_id, meta)
            return True, meta.get("summary", "")

        # 2) Pass 1
        logger.info("[bold]\\[2/3] Pass 1 — generating candidates...[/]")
        candidates = self.llm.generate_question_candidates(chunk, context_str)
        easy = sum(1 for c in candidates if c.get("difficulty") == "Easy")
        medium = sum(1 for c in candidates if c.get("difficulty") == "Medium")
        hard = sum(1 for c in candidates if c.get("difficulty") == "Hard")
        logger.info(
            f"[green]✓[/] {len(candidates)} candidates — "
            f"[blue]Easy={easy}[/]  [yellow]Medium={medium}[/]  [red]Hard={hard}[/]"
        )

        all_valid_qa: List[Dict] = []
        rejected_qa: List[Dict] = []

        # 3) Pass 2
        logger.info("[bold]\\[3/3] Pass 2 — answer generation & validation...[/]")
        for i, cand in enumerate(candidates):
            question = cand.get("question", "").strip()
            source_quote = cand.get("source_quote", "").strip()
            level = cand.get("difficulty", "Medium")
            q_preview = question[:80] + ("..." if len(question) > 80 else "")

            level_colour = {"Easy": "blue", "Medium": "yellow", "Hard": "red"}.get(
                level, "white"
            )
            console.print(
                f"  [{i + 1}/{len(candidates)}] "
                f"[{level_colour}]\\[{level}][/] {q_preview}"
            )

            # Sanity check
            if not question or len(source_quote) < 25:
                console.print("    [red]✗ REJECTED[/] — invalid Pass 1 output")
                rejected_qa.append(
                    {"level": level, "question": question, "reason": "Invalid Pass 1"}
                )
                continue

            # Quote guard
            quote_score = fuzz.partial_ratio(source_quote.lower(), content.lower())
            score_colour = "green" if quote_score >= QUOTE_MATCH_THRESHOLD else "red"
            console.print(f"    Quote match: [{score_colour}]{quote_score:.1f}%[/]")
            if quote_score < QUOTE_MATCH_THRESHOLD:
                console.print(
                    f"    [red]✗ REJECTED[/] — quote guard "
                    f"({quote_score:.1f}% < {QUOTE_MATCH_THRESHOLD}%)"
                )
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "reason": f"Weak quote ({quote_score:.1f}%)",
                    }
                )
                continue

            # Generate answer
            answer = self.llm.generate_reference_answer(question, content)
            ans_preview = answer[:100] + ("..." if len(answer) > 100 else "")
            console.print(f"    [dim]Answer:[/] {ans_preview}")

            # NOT_ENOUGH_INFORMATION check
            normalised = answer.replace("_", " ").upper()
            if not answer or "NOT ENOUGH INFORMATION" in normalised:
                console.print(
                    "    [red]✗ REJECTED[/] — model reported insufficient info"
                )
                rejected_qa.append(
                    {"level": level, "question": question, "reason": "Pass 2 refused"}
                )
                continue

            # Validate
            is_valid, reason = self.validator.validate(question, answer, content)
            if is_valid:
                cand["answer"] = answer
                cand["type"] = DIFFICULTY_TYPE.get(level, "Fact")
                all_valid_qa.append(cand)
                console.print("    [green]✅ ACCEPTED[/]")
            else:
                console.print(f"    [red]✗ REJECTED[/] — {reason}")
                rejected_qa.append(
                    {"level": level, "question": question, "reason": reason}
                )

        # 4) Persist
        self.db.save_enrichment(chunk_id, meta)
        if all_valid_qa:
            self.db.save_questions(chunk_id, all_valid_qa)
            for qa in all_valid_qa:
                self.vector_store.add_qa_pair(
                    chunk_id=chunk_id,
                    question_text=qa["question"],
                    answer_text=qa["answer"],
                    source_quote=qa["source_quote"],
                    difficulty=qa["difficulty"],
                    question_type=qa["type"],
                    tags=meta.get("tags", []),
                )

        # 5) Rejections
        if rejected_qa:
            self.db.save_rejections(chunk_id, rejected_qa)

        # Summary panel
        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_row(
            f"[green]✅ Accepted:[/] {len(all_valid_qa)}",
            f"[red]✗ Rejected:[/] {len(rejected_qa)}",
            f"[dim]Total:[/] {len(candidates)}",
        )
        console.print(
            Panel(
                summary_table, title=f"[cyan]Chunk {chunk_id[:8]} done[/]", expand=False
            )
        )

        return True, meta.get("summary", "")

    def process_file(self, file_id: str):
        """
        Process all chunks for a given file_id in order.

        The method maintains a short 'history' deque of the last N chunk summaries
        (N=3) to provide a limited context for subsequent chunks. For structural
        chunk types (table/math/code) we avoid history and instead supply snippets
        from adjacent chunks as context (to reduce hallucination risk).
        """
        chunks = self.db.get_chunks_for_file_ordered(file_id)
        total = len(chunks)
        console.print(
            Panel(
                f"[bold]File:[/] {file_id}\n[bold]Chunks:[/] {total}",
                title="[bold cyan]📂 Processing File[/]",
                expand=False,
            )
        )

        history: Deque[str] = deque(maxlen=3)

        for i, chunk in enumerate(chunks):
            ctype = chunk.get("content_type", "prose")
            logger.info(f"Chunk [bold][{i + 1}/{total}][/] — type=[cyan]{ctype}[/]")

            if ctype in ["table", "math", "code"]:
                parts = []
                if i > 0:
                    parts.append(f"PREV: {chunks[i - 1]['content'][:100]}...")
                if i < len(chunks) - 1:
                    parts.append(f"NEXT: {chunks[i + 1]['content'][:100]}...")
                context_str = "\n".join(parts)
            else:
                context_str = "History:\n" + "\n".join([f"- {s}" for s in history])

            success, summary = self.process_chunk(chunk, context_str)
            if success and summary:
                history.append(summary)

        self.vector_store.persist()
        logger.info(
            f"[green]✅ File {file_id[:8]} completed[/] — vector store persisted"
        )

    def run(self):
        """
        Main loop to process pending files in the DB. Uses a small thread pool to
        parallelize file-level processing. Files are fetched in small batches to
        keep memory/latency bounded.
        """
        console.print(
            Panel(
                "[bold green]Two-Pass Enrichment Pipeline[/]\n"
                f"Model: [cyan]{MODEL_NAME}[/]  |  "
                f"Workers: [cyan]{MAX_WORKERS}[/]  |  "
                f"Quote threshold: [cyan]{QUOTE_MATCH_THRESHOLD}%[/]",
                title="[bold]🚀 Starting Pipeline[/]",
                expand=False,
            )
        )

        total_processed = 0
        while True:
            files = self.db.get_pending_files(limit=5)
            if not files:
                console.print(
                    Panel(
                        f"[green]All done.[/] Total files processed: [bold]{total_processed}[/]",
                        title="✅ Pipeline Complete",
                        expand=False,
                    )
                )
                break

            logger.info(
                f"Found [bold]{len(files)}[/] pending file(s) — processing batch..."
            )
            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = {ex.submit(self.process_file, fid): fid for fid in files}
                for future in as_completed(futures):
                    fid = futures[future]
                    try:
                        future.result()
                        total_processed += 1
                        logger.info(
                            f"[green]✓[/] File {fid[:8]} done — {total_processed} total"
                        )
                    except Exception as e:
                        logger.error(f"[red]✗ Thread crash[/] on file {fid[:8]}: {e}")


# Entry point for manual runs. When executed as a script, run the manager.
if __name__ == "__main__":
    console.print(
        Panel(
            "[bold cyan]Enricher.py[/] — Production-Grade Two-Pass Pipeline\n"
            "Generator: [green]LFM 2.5 1.2B[/]  |  Validator: [green]BGE-Small-EN[/]",
            title="[bold]🐍 Enrichment Pipeline[/]",
            expand=False,
        )
    )
    EnrichmentManager().run()
