"""
Enricher.py - Production-Grade Enrichment Pipeline (Optimized Two-Pass)

This module implements an optimized two-pass enrichment pipeline for extracting
high-quality question-answer (QA) pairs and metadata from text "chunks".
It is designed to be robust and practical for production use.
"""

import json
import logging
import os
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

# External deps
from openai import OpenAI
from rapidfuzz import fuzz
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import SentenceTransformer, util

# ---------------- ROBUST PROJECT ROOT DETECTION ----------------
_env_root = os.environ.get("RAG_PROJECT_ROOT")
if _env_root:
    project_root = Path(_env_root).resolve()
else:
    project_root = Path(__file__).resolve().parents[4]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Paths
DB_PATH = str(project_root / "data" / "rag_staging.db")
CHROMA_DIR = str(project_root / "data" / "chroma_store")

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

try:
    from qp_core.DBManager import DBManager
except ImportError:
    print("Import errors")

# ---------------- CONFIG ----------------
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "lfm-2.5-1.2b"

MAX_WORKERS = 4
QUOTE_MATCH_THRESHOLD = 75.0
DEDUP_SIMILARITY_THRESHOLD = 80
MIN_TOKENS_FOR_METADATA = 30
MIN_TOKENS_FOR_QA = 30

DIFFICULTY_TYPE = {
    "Easy": "Fact",
    "Medium": "Mechanism",
    "Hard": "Critical",
}

# ---------------- DEFLECTION PATTERN ----------------
DEFLECTION_PATTERN = re.compile(
    r"the\s+text\s+does\s+not\s+"
    r"(specify|mention|state|provide|include|address|discuss|indicate|describe)"
    r"|the\s+(passage|chunk|document|context)\s+does\s+not"
    r"|no\s+(specific|numeric|direct)\s+(threshold|value|detail|information|mention)"
    r"|is\s+not\s+(mentioned|specified|discussed|provided|stated)\s+in\s+the\s+text"
    r"|cannot\s+be\s+(determined|inferred|found)\s+from",
    re.IGNORECASE,
)

# ---------------- RICH LOGGING ----------------
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
    return 1 if estimated_tokens < 80 else 3


# ---------------- ANSWER VALIDATOR ----------------
class AnswerValidator:
    SIMILARITY_THRESHOLD = 0.52
    LEXICAL_OVERLAP_THRESHOLD = 0.25
    MIN_SENTENCES = 1
    MAX_SENTENCES = 5

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading Validator Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def validate(self, question: str, answer: str, chunk: str) -> Tuple[bool, str]:
        if not answer or not answer.strip():
            return False, "Empty answer"

        struct_ok, sent_count = self._structural_check(answer)
        if not struct_ok:
            return False, f"Structural fail: {sent_count} sentences"

        embeddings = self.model.encode([chunk, answer], convert_to_tensor=True)
        score = float(util.cos_sim(embeddings[0], embeddings[1]))
        if score < self.SIMILARITY_THRESHOLD:
            return False, f"Semantic fail: cos={score:.3f}"

        c_words = self._content_words(chunk)
        a_words = self._content_words(answer)
        if not a_words:
            return False, "No content words"
        overlap = len(c_words & a_words) / len(a_words)
        if overlap < self.LEXICAL_OVERLAP_THRESHOLD:
            return False, f"Lexical fail: overlap={overlap:.3f}"

        if question.strip().lower().startswith(("why", "how")):
            causal = {
                "because",
                "due to",
                "as a result",
                "therefore",
                "leads to",
                "causes",
                "by",
                "through",
                "allows",
                "enables",
                "since",
            }
            answer_words = set(answer.lower().split())
            if not any(
                w in answer.lower() if " " in w else w in answer_words for w in causal
            ):
                return False, "Why/How-question without causal/mechanism phrase"

        return True, ""

    def _sentence_count(self, text: str) -> int:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return len([s for s in sentences if len(s) > 5])

    def _content_words(self, text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z]{4,}|\d+[%a-zA-Z]*", text)
        return {t.lower() for t in tokens}

    def _structural_check(self, text: str) -> Tuple[bool, int]:
        count = self._sentence_count(text)
        return (self.MIN_SENTENCES <= count <= self.MAX_SENTENCES), count


# ---------------- LLM CLIENT ----------------
# ---------------- LLM CLIENT ----------------
class LLMClient:
    """
    Encapsulates calls to the LLM backend.

    Responsibilities:
    - Provide helper to extract JSON safely from model output (robust to fences).
    - Encapsulate prompts for metadata, question candidate generation (Pass 1),
      and reference answer generation (Pass 2).
    - Deduplicate near-identical questions produced across difficulty calls.
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
        returning the parsed JSON result into a Python dict.

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

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate questions using fuzzy token_sort_ratio.

        token_sort_ratio is used rather than partial_ratio because it handles
        reordered words well — e.g. "Why does X cause Y?" vs "What causes Y in X?"
        would score high on token_sort but not on plain ratio.

        Two questions are considered duplicates if their score exceeds
        DEDUP_SIMILARITY_THRESHOLD. The first occurrence is kept.

        Args:
            candidates: list of candidate dicts from Pass 1

        Returns:
            Filtered list with near-duplicates removed.
        """
        seen: List[str] = []
        unique: List[Dict] = []
        for cand in candidates:
            q = cand.get("question", "").lower().strip()
            if not q:
                continue
            if any(
                fuzz.token_sort_ratio(q, s) > DEDUP_SIMILARITY_THRESHOLD for s in seen
            ):
                logger.debug(f"Deduped question: {q[:60]}")
                continue
            seen.append(q)
            unique.append(cand)

        removed = len(candidates) - len(unique)
        if removed:
            logger.info(f"[yellow]Deduplication removed {removed} near-duplicate(s)[/]")
        return unique

    @staticmethod
    def _normalize_tags(tags: List[str]) -> List[str]:
        """
        Post-process tags returned by the LLM into a consistent, deduplicated format.

        Problems this solves:
        - Mixed casing: "DataCuration", "data curation", "Data Curation" → one form
        - Duplicates: same concept appearing twice with different casing
        - Generic noise tags: overly broad terms that appear on almost every chunk
          and carry no discriminative value for retrieval or filtering

        Normalisation steps:
        1. Strip whitespace
        2. Convert CamelCase and Title Case to lowercase_with_underscores
        3. Remove tags that are on the generic blocklist
        4. Deduplicate (preserving first-seen order)

        Args:
            tags: raw list of tag strings from the LLM

        Returns:
            Cleaned, normalised, deduplicated list of tag strings.
        """
        # Tags so generic they appear on nearly every chunk and add no signal
        GENERIC_BLOCKLIST = {
            "rag",
            "nlp",
            "llm",
            "machine_learning",
            "machinelearning",
            "data_management",
            "datamanagement",
            "document_retrieval",
            "documentretrieval",
            "inference",
            "inferencelogic",
            "rag_framework",
            "ragframework",
            "preprocessing",
        }

        def _to_snake(tag: str) -> str:
            # Insert underscore between a lowercase letter and an uppercase letter
            # to handle CamelCase → snake_case
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tag.strip())
            # Replace spaces and hyphens with underscores
            s = re.sub(r"[\s\-]+", "_", s)
            return s.lower()

        seen: set = set()
        result: List[str] = []
        for raw in tags:
            if not isinstance(raw, str) or not raw.strip():
                continue
            normalised = _to_snake(raw)
            if normalised in GENERIC_BLOCKLIST:
                logger.debug(f"Tag blocked (generic): {raw!r} → {normalised!r}")
                continue
            if normalised in seen:
                logger.debug(f"Tag deduplicated: {normalised!r}")
                continue
            seen.add(normalised)
            result.append(normalised)

        return result

    def generate_metadata(
        self, text: str, context: str, estimated_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Generate structured metadata for a text chunk.

        Skips generation entirely for chunks below MIN_TOKENS_FOR_METADATA — these
        are too short to produce meaningful tags or triplets and the LLM would
        hallucinate structure from the context window instead.

        Expected output schema (JSON):
        {
          "summary": "2-3 sentence summary starting with the main subject",
          "tags": ["NounTag1", "NounTag2", ...],
          "triplets": [
            {"subject": "Entity", "predicate": "predicate", "object": "Entity"}
          ]
        }

        Args:
            text:             raw chunk content
            context:          surrounding context (for understanding only)
            estimated_tokens: token estimate for the chunk; used to gate generation

        Returns:
            Parsed metadata dict, or empty dict if the chunk is too small.
        """
        if estimated_tokens < MIN_TOKENS_FOR_METADATA:
            logger.warning(
                f"Chunk too small ({estimated_tokens} tokens) — skipping metadata generation"
            )
            return {}

        sys_prompt = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "Always output valid JSON following the exact schema below. "
            "You extract information ONLY from the TEXT CHUNK — never from context."
        )
        user_prompt = f"""
    ### BACKGROUND CONTEXT (strictly for orientation — DO NOT extract tags, triplets, or
    ### summary content from here; it describes a different part of the document):
    {context}

    ### TEXT CHUNK — THIS IS YOUR ONLY SOURCE:
    {text}

    ### TASK:
    Analyse the TEXT CHUNK above and return ONLY this JSON structure.

    ### STRICT RULES:
    1. Tags must be noun phrases that appear explicitly in the TEXT CHUNK.
    DO NOT invent tags from the Background Context (e.g. do not tag every chunk
    with "RAG" or "DocumentRetrieval" unless those exact concepts appear in the chunk).
    2. Every triplet subject and object must be an entity named in the TEXT CHUNK.
    3. The summary must describe what the TEXT CHUNK discusses, not the overall project.
    4. Use lowercase_with_underscores for all tags (e.g. "heuristic_filtering", "common_crawl").

    {{
    "summary": "2-3 sentence summary of the TEXT CHUNK starting with its main subject",
    "tags": ["noun_tag1", "noun_tag2"],
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

        After all three calls, near-duplicate questions are removed via
        _deduplicate_candidates before returning.
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
4. Every question MUST target a DIFFERENT concept, fact, or mechanism from the chunk.
   Do NOT rephrase or paraphrase any previously listed question.
   Each question must have a meaningfully distinct answer from all others.

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

        # Remove near-duplicates produced across the three difficulty calls
        return self._deduplicate_candidates(all_candidates)

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
- Your answer MUST be a complete sentence.
- Your answer MUST be similar to how a human would write a reference answer in a technical interview setting.
- You MUST use ONLY the information in the SOURCE TEXT. Do NOT use any outside knowledge.
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
    def __init__(self):
        logger.info("🚀 Starting Optimized Two-Pass Pipeline...")
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)
        self.validator = AnswerValidator()

    # =========================================================
    # INTERNAL CORE — process_chunk  (UPDATED with safety guard)
    # =========================================================
    def _process_chunk(self, chunk: Dict, context_str: str) -> Tuple[bool, str]:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        tokens = chunk.get("estimated_tokens", 0)
        ctype = chunk.get("content_type", "prose")

        # ==================== FULL IDEMPOTENCY GUARD ====================
        if chunk.get("should_use") != 1:
            logger.info(f"Chunk {chunk_id[:8]} has should_use=0 — skipping")
            return True, chunk.get("existing_summary", "")

        if chunk.get("existing_summary") and self.db.get_questions_for_chunk(chunk_id):
            logger.info(f"Chunk {chunk_id[:8]} already fully enriched — skipping")
            return True, chunk.get("existing_summary", "")
        # =================================================================

        console.rule(
            f"[bold cyan]Chunk {chunk_id[:8]}[/] | {tokens} tokens | type={ctype}"
        )

        logger.info("[bold]\\[1/3] Generating metadata...[/]")
        meta = self.llm.generate_metadata(content, context_str, estimated_tokens=tokens)
        if not meta:
            if tokens < MIN_TOKENS_FOR_METADATA:
                logger.warning(f"Chunk {chunk_id[:8]} too small — no metadata or QA")
                return True, ""
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

        meta["tags"] = self.llm._normalize_tags(meta.get("tags", []))
        logger.info(f"[dim]Normalised tags:[/] {meta['tags']}")

        self.db.save_enrichment(chunk_id, meta)

        if tokens < MIN_TOKENS_FOR_QA:
            logger.warning(f"Chunk too small ({tokens} tokens) — skipping QA")
            return True, meta.get("summary", "")

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
                f"  [{i + 1}/{len(candidates)}] [{level_colour}]\\[{level}][/] {q_preview}"
            )

            if not question or len(source_quote) < 25:
                console.print("    [red]✗ REJECTED[/] — invalid Pass 1 output")
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": "",
                        "reason": "Invalid Pass 1",
                    }
                )
                continue

            quote_score = fuzz.partial_ratio(source_quote.lower(), content.lower())
            score_colour = "green" if quote_score >= QUOTE_MATCH_THRESHOLD else "red"
            console.print(f"    Quote match: [{score_colour}]{quote_score:.1f}%[/]")
            if quote_score < QUOTE_MATCH_THRESHOLD:
                console.print(
                    f"    [red]✗ REJECTED[/] — quote guard ({quote_score:.1f}%)"
                )
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": "",
                        "reason": f"Weak quote ({quote_score:.1f}%)",
                    }
                )
                continue

            answer = self.llm.generate_reference_answer(question, content)
            ans_preview = answer[:100] + ("..." if len(answer) > 100 else "")
            console.print(f"    [dim]Answer:[/] {ans_preview}")

            normalised = answer.replace("_", " ").upper()
            if not answer or "NOT ENOUGH INFORMATION" in normalised:
                console.print(
                    "    [red]✗ REJECTED[/] — model reported insufficient info"
                )
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": answer,
                        "reason": "Pass 2 refused",
                    }
                )
                continue

            if DEFLECTION_PATTERN.search(answer):
                console.print("    [red]✗ REJECTED[/] — deflection phrase detected")
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": answer,
                        "reason": f"Deflection: '{answer[:80]}'",
                    }
                )
                continue

            is_valid, reason = self.validator.validate(question, answer, content)
            if is_valid:
                cand["answer"] = answer
                cand["type"] = DIFFICULTY_TYPE.get(level, "Fact")
                all_valid_qa.append(cand)
                console.print("    [green]✅ ACCEPTED[/]")
            else:
                console.print(f"    [red]✗ REJECTED[/] — {reason}")
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": answer,
                        "reason": reason,
                    }
                )

        if all_valid_qa:
            self.db.save_questions(chunk_id, all_valid_qa)

        if rejected_qa:
            self.db.save_rejections(chunk_id, rejected_qa)

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

    # =========================================================
    # PUBLIC: enrich_single_file  (UPDATED with real prev/next)
    # =========================================================
    def enrich_single_file(self, file_id: str) -> None:
        chunks = self.db.get_chunks_for_file_ordered(file_id)
        total = len(chunks)

        console.print(
            Panel(
                f"[bold]File:[/] {file_id}\n[bold]Chunks:[/] {total}",
                title="[bold cyan]📂 Enriching Single File[/]",
                expand=False,
            )
        )

        if not chunks:
            logger.warning(f"No processable chunks found for file {file_id[:8]}")
            return

        # Fast lookup for real prev_chunk_id / next_chunk_id
        chunk_map = {c["chunk_id"]: c["content"] for c in chunks}

        history: Deque[str] = deque(maxlen=3)

        for i, chunk in enumerate(chunks):
            ctype = chunk.get("content_type", "prose")
            logger.info(f"Chunk [bold][{i + 1}/{total}][/] — type=[cyan]{ctype}[/]")

            if ctype in ["table", "math", "code"]:
                parts = []

                prev_id = chunk.get("prev_chunk_id")
                if prev_id and prev_id in chunk_map:
                    prev_text = chunk_map[prev_id][:150]
                    parts.append(
                        f"PREV: {prev_text}{'...' if len(prev_text) == 150 else ''}"
                    )

                next_id = chunk.get("next_chunk_id")
                if next_id and next_id in chunk_map:
                    next_text = chunk_map[next_id][:150]
                    parts.append(
                        f"NEXT: {next_text}{'...' if len(next_text) == 150 else ''}"
                    )

                context_str = (
                    "\n".join(parts) if parts else "No adjacent structural chunks"
                )
            else:
                context_str = "History:\n" + "\n".join([f"- {s}" for s in history])

            success, summary = self._process_chunk(chunk, context_str)
            if success and summary:
                history.append(summary)

        logger.info(f"[green]✅ File {file_id[:8]} enrichment complete[/]")

    # =========================================================
    # Other public methods (enrich_single_chunk, run) remain unchanged
    # =========================================================
    def enrich_single_chunk(
        self, chunk: Dict, context_str: Optional[str] = None
    ) -> Dict[str, Any]:
        ctx = context_str or ""
        console.print(
            Panel(
                f"[bold]Chunk:[/] {chunk.get('chunk_id', 'unknown')[:8]}\n"
                f"[bold]Tokens:[/] {chunk.get('estimated_tokens', '?')}",
                title="[bold cyan]🔍 Enriching Single Chunk[/]",
                expand=False,
            )
        )
        success, summary = self._process_chunk(chunk, ctx)
        return {
            "success": success,
            "summary": summary,
            "chunk_id": chunk.get("chunk_id", ""),
        }

    def run(self) -> None:
        # (your original run method - unchanged)
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
                futures = {
                    ex.submit(self.enrich_single_file, fid): fid for fid in files
                }
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


# Entry point
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
