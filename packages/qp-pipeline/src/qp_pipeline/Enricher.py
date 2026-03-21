"""
Enricher.py - Production-Grade Enrichment Pipeline (Optimized Two-Pass)

This module implements an optimized two-pass enrichment pipeline for extracting
high-quality question-answer (QA) pairs and metadata from text "chunks".
It is designed to be robust and practical for production use.

Changelog vs original:
- [LFM] Added top_k=50 and repetition_penalty=1.05 to all API calls (official LFM2.5-1.2B-Instruct settings)
- [LFM] Removed response_format JSON flag; replaced with assistant-role prefill (more reliable on small models)
- [PROMPT] Metadata sys_prompt: Chain-of-Draft reasoning style + self-termination instruction
- [PROMPT] Pass 1 sys_prompt: Chain-of-Draft + commit-on-first-confidence instruction
- [PROMPT] Pass 2 sys_prompt: Chain-of-Draft + write-immediately-and-stop instruction
- [PROMPT] Summary prompt: declared purpose, content-type branching, no meta-commentary framing
- [MATH] Math-specific difficulty instructions (definition/computation/derivation axis)
- [MATH] Math quote guard: exact substring check instead of fragile fuzzy match
- [TAG] _verify_tags_grounded: post-generation check that each tag appears in chunk text
- [TRIPLET] _verify_triplets_grounded: checks subject and object appear in chunk text
- [VALIDATOR] Removed "by" from causal set (too generic, causes false passes)
- [HISTORY] Summary deflection check + min-length gate before appending to deque
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
MIN_SUMMARY_TOKENS = 10  # [NEW] summaries shorter than this are treated as failures

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
            # [FIX] Removed "by" — too generic, causes false passes on non-causal answers.
            # e.g. "produced by the pipeline" or "answered by the document" would pass
            # without explaining any mechanism. Kept only words with genuine causal signal.
            causal = {
                "because",
                "due to",
                "as a result",
                "therefore",
                "leads to",
                "causes",
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
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract JSON object from LLM text.

        First attempts strict json.loads (fastest, zero overhead for valid output).
        On failure, falls back to json_repair which handles the most common LLM
        pathology: unescaped double-quotes or newlines inside string values.
        Strips ```json fences before either attempt.
        """
        from json_repair import repair_json

        # Strip ```json ... ``` fences if present
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        raw = fenced.group(1) if fenced else text

        # Fast path — valid JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Slow path — let json_repair fix unescaped characters
        try:
            result = repair_json(raw, return_objects=True)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        logger.error(f"JSON Parse Error: unrecoverable | Raw: {text[:100]}...")
        return {}

    def _call_model(self, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Wrapper for calling the LLM chat-completions endpoint.

        Uses assistant-role prefill with '{' to force JSON output — more reliable
        than response_format on a 1.2B model. Uses official LFM2.5-1.2B-Instruct
        sampling settings: temperature=0.1, top_k=50, repetition_penalty=1.05.
        top_p is intentionally omitted (only recommended for the Thinking variant).
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                    # [LFM] Prefill: forces the model into JSON mode without relying
                    # on response_format, which may be ignored by small models.
                    {"role": "assistant", "content": "{"},
                ],
                temperature=0.1,  # [LFM] Official recommended value
                top_k=50,  # [LFM] Official recommended value
                repetition_penalty=1.05,  # [LFM] Official recommended value
                # top_p intentionally omitted — only for LFM2.5-1.2B-Thinking
            )
            # Re-attach the prefilled '{' that was injected via assistant role
            raw_content = "{" + response.choices[0].message.content
            return self._extract_json(raw_content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate questions using fuzzy token_sort_ratio.

        token_sort_ratio handles reordered words well —
        e.g. "Why does X cause Y?" vs "What causes Y in X?" would score high
        on token_sort but not on plain ratio.

        Two questions are considered duplicates if their score exceeds
        DEDUP_SIMILARITY_THRESHOLD. The first occurrence is kept.
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
        Post-process tags into consistent, deduplicated snake_case format.

        Normalisation steps:
        1. Strip whitespace
        2. Convert CamelCase and Title Case to lowercase_with_underscores
        3. Remove tags on the generic blocklist
        4. Deduplicate (preserving first-seen order)
        """
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
            s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tag.strip())
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

    @staticmethod
    def _verify_tags_grounded(tags: List[str], text: str) -> List[str]:
        """
        [NEW] Filter out tags that have no textual basis in the chunk.

        _normalize_tags enforces format but never checks whether the concept
        actually appears in the source text. This step catches hallucinated tags
        that are thematically plausible but not grounded in the chunk content.

        Checks both 'tag_with_underscores' and 'tag with spaces' forms.
        """
        text_lower = text.lower()
        grounded = []
        for tag in tags:
            surface = tag.replace("_", " ")
            if surface in text_lower or tag in text_lower:
                grounded.append(tag)
            else:
                logger.debug(f"Tag ungrounded (removed): {tag!r}")
        removed = len(tags) - len(grounded)
        if removed:
            logger.info(f"[yellow]Tag grounding removed {removed} ungrounded tag(s)[/]")
        return grounded

    @staticmethod
    def _verify_triplets_grounded(triplets: List[Dict], text: str) -> List[Dict]:
        """
        [NEW] Filter out triplets whose subject or object cannot be found in the chunk.

        Triplets have no validation in the original pipeline — a model can produce
        plausible-sounding but hallucinated relationships on every chunk without any
        rejection being logged. This check ensures both subject and object are
        named entities that actually appear in the source text.
        """
        text_lower = text.lower()
        grounded = []
        for t in triplets:
            subj = t.get("subject", "").lower().strip()
            obj = t.get("object", "").lower().strip()
            if subj and obj and subj in text_lower and obj in text_lower:
                grounded.append(t)
            else:
                logger.debug(
                    f"Triplet ungrounded (removed): ({t.get('subject')!r}, "
                    f"{t.get('predicate')!r}, {t.get('object')!r})"
                )
        removed = len(triplets) - len(grounded)
        if removed:
            logger.info(
                f"[yellow]Triplet grounding removed {removed} ungrounded triplet(s)[/]"
            )
        return grounded

    def generate_metadata(
        self,
        text: str,
        context: str,
        estimated_tokens: int = 150,
        content_type: str = "prose",  # [NEW] used for summary instruction branching
    ) -> Dict[str, Any]:
        """
        Generate structured metadata for a text chunk.

        Skips generation for chunks below MIN_TOKENS_FOR_METADATA.

        Summary instruction is branched by content_type so the model produces
        a purpose-appropriate summary rather than a generic table-of-contents entry.
        Tags and triplets are grounded against the chunk text after generation.
        """
        if estimated_tokens < MIN_TOKENS_FOR_METADATA:
            logger.warning(
                f"Chunk too small ({estimated_tokens} tokens) — skipping metadata generation"
            )
            return {}

        # [PROMPT] Chain-of-Draft + self-termination for metadata extraction
        sys_prompt = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
            "Stop reasoning the moment you have identified the tags, triplets, and summary — "
            "do not verify them a second time. "
            "Extract information ONLY from the TEXT CHUNK — never from context. "
            "Always output valid JSON following the exact schema provided."
        )

        # [NEW] Content-type-specific summary instructions.
        # Each variant declares the downstream purpose (history context for subsequent chunks)
        # and prohibits meta-commentary framing ("this chunk discusses...").
        summary_instruction = {
            "math": (
                "1-2 sentences naming the formula or theorem and stating what it computes "
                "or proves. Include the formula itself if it fits in one line. "
                "Do not start with 'This chunk' or 'This section'."
            ),
            "code": (
                "1-2 sentences naming the function or class, describing its inputs, outputs, "
                "and purpose. Do not start with 'This chunk' or 'This section'."
            ),
            "table": (
                "1-2 sentences stating what the table measures and the key values or ranges it contains. "
                "Do not start with 'This chunk' or 'This section'."
            ),
        }.get(
            content_type,
            # Default (prose): written for downstream model use as history context
            (
                "2-3 sentences written for a downstream model that will use this as context. "
                "Include: the main concept, any specific values or thresholds mentioned, "
                "and how this chunk relates to the preceding context if relevant. "
                "Do not use phrases like 'this chunk discusses' or 'this section covers'."
            ),
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
   DO NOT invent tags from the Background Context. There can be 5 tags at best and 1 tag at minimum.
2. Every triplet subject and object must be an entity named in the TEXT CHUNK.
3. Use lowercase_with_underscores for all tags (e.g. "heuristic_filtering", "common_crawl").
4. Summary instruction: {summary_instruction}

{{
"summary": "...",
"tags": ["Tag1", "Tag2"],
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

        Three separate LLM calls, one per difficulty level. Difficulty instructions
        are branched by content_type so math chunks receive mathematically appropriate
        difficulty axes (definition/computation/derivation) rather than prose axes
        (recall/mechanism/critical).

        After all three calls, near-duplicates are removed via _deduplicate_candidates.
        """
        estimated_tokens = chunk.get("estimated_tokens", 150)
        per_diff = get_questions_per_difficulty(estimated_tokens)
        content_type = chunk.get("content_type", "prose")

        # [NEW] Math-specific difficulty framing.
        # Prose framing ("analytical / critical") produces vague meta-questions on math
        # chunks. Math difficulty maps naturally to definition → computation → derivation.
        if content_type == "math":
            difficulty_instructions = {
                "Easy": "definition or notation — ask what a symbol, term, or operator means",
                "Medium": "computation — ask what the formula computes or what its components represent",
                "Hard": "derivation or proof — ask why the formula takes this form, what assumption it relies on, or what happens if a constraint is relaxed",
            }
        else:
            difficulty_instructions = {
                "Easy": "factual recall — ask about specific definitions, names, or stated facts",
                "Medium": "conceptual / mechanism — ask how or why something works",
                "Hard": "analytical / critical — ask about limitations, trade-offs, or comparisons",
            }

        # [PROMPT] Chain-of-Draft + commit-on-first-confidence instruction
        sys_prompt = (
            "You are an Expert Technical Interviewer. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
            "Once you have identified a valid question and its source quote, commit to it — "
            "do not re-evaluate or rephrase it. "
            "Output ONLY valid JSON."
        )
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
                item["difficulty"] = difficulty
                all_candidates.append(item)
                count += 1

            logger.debug(f"Pass 1 [{difficulty}]: {count} candidate(s)")

        return self._deduplicate_candidates(all_candidates)

    # ====================== PASS 2 ======================
    def generate_reference_answer(self, question: str, chunk_content: str) -> str:
        """
        Generate a concise reference answer using ONLY chunk_content.

        Uses official LFM2.5-1.2B-Instruct sampling settings.
        Prompt is intentionally less punitive than strict "not enough info" framing
        to avoid defensive refusals on small models.
        """
        # [PROMPT] Chain-of-Draft + write-immediately-and-stop instruction
        sys_prompt = (
            "You are a technical document analyzer. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
            "Once you have located the relevant fact in the source text, write the answer "
            "immediately and stop. "
            "Do not re-read the text or reconsider your answer after forming it. "
            "Extract facts directly from the text."
        )
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
                temperature=0.1,  # [LFM] Official recommended value
                top_k=50,  # [LFM] Official recommended value
                repetition_penalty=1.05,  # [LFM] Official recommended value
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
    # INTERNAL CORE — _process_chunk
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
        # [FIX] Pass content_type so summary instruction is branched correctly
        meta = self.llm.generate_metadata(
            content, context_str, estimated_tokens=tokens, content_type=ctype
        )
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

        # [FIX] Normalise → ground tags against chunk text → log
        meta["tags"] = self.llm._normalize_tags(meta.get("tags", []))
        meta["tags"] = self.llm._verify_tags_grounded(meta["tags"], content)
        logger.info(f"[dim]Grounded tags:[/] {meta['tags']}")

        # [FIX] Ground triplets against chunk text before saving
        meta["triplets"] = self.llm._verify_triplets_grounded(
            meta.get("triplets", []), content
        )

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

            # [FIX] Math quote guard: fuzzy matching breaks on LaTeX/notation because
            # whitespace and formatting differences cause score drops on valid verbatim
            # quotes. For math chunks, use exact substring check instead.
            if ctype == "math":
                quote_ok = source_quote.strip() in content
                if not quote_ok:
                    console.print(
                        "    [red]✗ REJECTED[/] — math quote not found verbatim in chunk"
                    )
                    rejected_qa.append(
                        {
                            "level": level,
                            "question": question,
                            "answer": "",
                            "reason": "Math quote not verbatim",
                        }
                    )
                    continue
            else:
                quote_score = fuzz.partial_ratio(source_quote.lower(), content.lower())
                score_colour = (
                    "green" if quote_score >= QUOTE_MATCH_THRESHOLD else "red"
                )
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
    # PUBLIC: enrich_single_file
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

            # [FIX] Guard history deque against deflection summaries and degenerate
            # outputs. A bad summary poisons context for all subsequent chunks in the
            # file. Only append if the summary passes the deflection check and meets
            # minimum length (MIN_SUMMARY_TOKENS words as a proxy for token count).
            if (
                success
                and summary
                and not DEFLECTION_PATTERN.search(summary)
                and len(summary.split()) >= MIN_SUMMARY_TOKENS
            ):
                history.append(summary)
            elif success and summary:
                logger.warning(
                    f"Summary for chunk {chunk.get('chunk_id', '')[:8]} failed quality "
                    f"gate — not added to history context"
                )

        logger.info(f"[green]✅ File {file_id[:8]} enrichment complete[/]")

    # =========================================================
    # Other public methods
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
