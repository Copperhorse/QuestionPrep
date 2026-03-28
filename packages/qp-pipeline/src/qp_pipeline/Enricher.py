"""
Enricher.py — Production-Grade Enrichment Pipeline (Optimized Two-Pass)

"""

import json
import logging
import os
import re
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from json_repair import repair_json
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

# B21: Was 4. llama-server processes exactly one request at a time, so multiple
# workers only add thread-switching overhead. Set to 1 for sequential processing.
MAX_WORKERS = 1

QUOTE_MATCH_THRESHOLD = 70.0
DEDUP_SIMILARITY_THRESHOLD = 80
MIN_TOKENS_FOR_METADATA = 30
MIN_TOKENS_FOR_QA = 30
MIN_SUMMARY_TOKENS = 10

DIFFICULTY_TYPE = {
    "Easy": "Fact",
    "Medium": "Mechanism",
    "Hard": "Critical",
}

DEFLECTION_PATTERN = re.compile(
    r"the\s+text\s+does\s+not\s+"
    r"(specify|mention|state|provide|include|address|discuss|indicate|describe)"
    r"|the\s+(passage|chunk|document|context)\s+does\s+not"
    r"|no\s+(specific|numeric|direct)\s+(threshold|value|detail|information|mention)"
    r"|is\s+not\s+(mentioned|specified|discussed|provided|stated)\s+in\s+the\s+text"
    r"|cannot\s+be\s+(determined|inferred|found)\s+from",
    re.IGNORECASE,
)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("Enricher")


QUOTE_MIN_CHARS = 40  # require substantive quotes


def _quote_is_grounded(quote: str, content: str) -> tuple[bool, str]:
    """
    Returns (is_grounded, reason).
    Two-step check:
      1. Minimum length — short quotes are not specific enough to be useful.
      2. Exact or near-exact substring — the quote must appear in the content
         with high fidelity (not just share common words).
    """
    quote = quote.strip()
    if len(quote) < QUOTE_MIN_CHARS:
        return False, f"Quote too short ({len(quote)} chars, min {QUOTE_MIN_CHARS})"

    # Try exact first (fast)
    if quote.lower() in content.lower():
        return True, "exact"

    # Fuzzy fallback for minor OCR/formatting differences
    score = fuzz.partial_ratio(quote.lower(), content.lower())
    if score >= QUOTE_MATCH_THRESHOLD:
        return True, f"fuzzy ({score:.0f}%)"

    return False, f"not found ({score:.0f}%)"


def _content_words(t: str):
    return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", t)}


def _ensure_dict(value: Any) -> Dict[str, Any]:
    """
    Guarantee the return value is a dict.
    - dict  → return as-is
    - list of one dict → unwrap (model sometimes wraps in an array)
    - anything else    → return {} so caller degrades gracefully
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        logger.warning("JSON parsed as single-element list — unwrapping to dict")
        return value[0]
    if isinstance(value, list) and value:
        logger.warning(
            f"JSON parsed as list with {len(value)} elements — "
            "expected a dict. Returning empty."
        )
    return {}


def get_questions_per_difficulty(estimated_tokens: int) -> int:
    """
    Scale question quota to chunk size.
    Asking for 9 questions (3 per difficulty) from an 80-token paragraph
    forces the model to duplicate or hallucinate — small chunks get fewer.
    """
    if estimated_tokens < 80:
        return 1  # ~60 words: one question per difficulty max
    elif estimated_tokens < 200:
        return 2  # medium chunk
    else:
        return 3  # large chunk only


class AnswerValidator:
    SIMILARITY_THRESHOLD = 0.55
    LEXICAL_OVERLAP_THRESHOLD = 0.75
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
        return _content_words(text)

    def _structural_check(self, text: str) -> Tuple[bool, int]:
        count = self._sentence_count(text)
        return (self.MIN_SENTENCES <= count <= self.MAX_SENTENCES), count


class LLMClient:
    def __init__(self, base_url: str, api_key: str = "no-key"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """ """

        # B17: greedy `.*` + re.DOTALL — captures full nested JSON from fenced blocks
        fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        raw = fenced.group(1) if fenced else text

        try:
            parsed = json.loads(raw)
            result = _ensure_dict(parsed)
            if result:
                return result
            # Fell through (was a list or unexpected type) — try repair below
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(raw, return_objects=True)
            result = _ensure_dict(repaired)
            if result:
                return result
        except Exception:
            pass

        logger.error(f"JSON Parse Error: unrecoverable | Raw: {text[:100]}...")
        return {}

    def _call_model(self, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "{"},
                ],
                temperature=0.1,
                # top_k and repetition_penalty are llama-server extensions —
                # the OpenAI Python client rejects them as direct kwargs (TypeError).
                # They must be forwarded via extra_body.
                extra_body={
                    "top_k": 50,
                    "repetition_penalty": 1.05,
                },
            )
            content = response.choices[0].message.content

            # B16: Some backends echo the assistant prefill back into the response.
            # If the content already starts with '{', prepending another '{' produces
            # '{{...}' which is invalid JSON. Check before prepending.
            content_stripped = content.lstrip()
            if content_stripped.startswith("{"):
                raw_content = content_stripped
            else:
                raw_content = "{" + content

            return self._extract_json(raw_content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate questions by checking both question text AND source quote.

        Two questions may have different wording but target the same sentence —
        their answers would be nearly identical, so they're pedagogically redundant.
        Quote-based dedup catches this second case.
        """
        seen_qs: List[str] = []
        seen_quotes: List[str] = []
        unique: List[Dict] = []

        for cand in candidates:
            if not isinstance(cand, dict):
                # LLM occasionally returns qa_pairs items as arrays instead of dicts.
                # .get() on a list raises 'list has no attribute get'.
                logger.debug(f"Candidate skipped (not a dict): {cand!r}")
                continue
            q = cand.get("question", "").lower().strip()
            quote = cand.get("source_quote", "").lower().strip()
            if not q:
                continue

            # Reject if question text is too similar to an existing one
            if any(
                fuzz.token_sort_ratio(q, s) > DEDUP_SIMILARITY_THRESHOLD
                for s in seen_qs
            ):
                logger.debug(f"Deduped question (text): {q[:60]}")
                continue

            # Reject if it targets the exact same sentence as an existing question
            if quote and any(fuzz.partial_ratio(quote, sq) > 90 for sq in seen_quotes):
                logger.debug(f"Deduped question (shared quote): {q[:60]}")
                continue

            seen_qs.append(q)
            if quote:
                seen_quotes.append(quote)
            unique.append(cand)

        removed = len(candidates) - len(unique)
        if removed:
            logger.info(f"[yellow]Deduplication removed {removed} near-duplicate(s)[/]")
        return unique

    @staticmethod
    def _normalize_tags(tags: List[str]) -> List[str]:
        GENERIC_BLOCKLIST = {
            " ",
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
                continue
            if normalised in seen:
                continue
            seen.add(normalised)
            result.append(normalised)

        return result

    @staticmethod
    def _verify_tags_grounded(tags: List[str], text: str) -> List[str]:
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
        text_lower = text.lower()
        grounded = []
        for t in triplets:
            if not isinstance(t, dict):
                # LLM occasionally returns triplets as [subj, pred, obj] arrays
                # instead of {"subject": ..., "predicate": ..., "object": ...} dicts.
                # Calling .get() on a list raises 'list has no attribute get' — skip.
                logger.debug(f"Triplet skipped (not a dict): {t!r}")
                continue
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
        content_type: str = "prose",
    ) -> Dict[str, Any]:
        if estimated_tokens < MIN_TOKENS_FOR_METADATA:
            logger.warning(
                f"Chunk too small ({estimated_tokens} tokens) — skipping metadata generation"
            )
            return {}

        sys_prompt = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
            "Stop reasoning the moment you have identified the tags, triplets, and summary — "
            "do not verify them a second time. "
            "Extract information ONLY from the TEXT CHUNK — never from context. "
            "Always output valid JSON following the exact schema provided."
        )

        # Improved summary instruction (forces direct, concise, non-repetitive summaries)
        summary_instruction = {
            "math": (
                "1-2 sentences naming the formula or theorem and stating what it computes "
                "or proves. Include the formula itself if it fits in one line."
            ),
            "code": (
                "1-2 sentences naming the function or class, describing its inputs, outputs, "
                "and purpose."
            ),
            "table": (
                "1-2 sentences stating what the table measures and the key values or ranges it contains."
            ),
        }.get(
            content_type,
            (
                "Write exactly 2 complete sentences. "
                "Sentence 1: state the core technical concept or claim. "
                "Sentence 2: name the specific methods, techniques, or entities mentioned. "
                "Never start with 'The text', 'This chunk', or 'This section'."
            ),
        )

        user_prompt = f"""
   ### BACKGROUND CONTEXT (strictly for orientation — DO NOT extract tags, triplets, or summary from here):
   {context}

   ### TEXT CHUNK — THIS IS YOUR ONLY SOURCE:
   {text}

   ### TASK:
   Analyse ONLY the TEXT CHUNK above and return EXACTLY this JSON structure.

   ### STRICT RULES:
   1. Tags: 2–5 noun phrases that appear explicitly in the TEXT CHUNK. Use lowercase_with_underscores.
      Examples: "weak_labels", "gan_training", "entity_augmentation", "semi_supervised_learning".
      Do not invent tags. Do not use generic tags like "data" or "machine_learning".
   2. Triplets: Create meaningful subject-predicate-object triples where BOTH subject and object appear in the TEXT CHUNK.
      Predicate must be a single verb or short verb phrase (e.g. "uses", "generates", "improves", "specializes_for").
      Aim for 1–4 high-quality triplets. If none are meaningful, return an empty list.
   3. Summary: Follow the instruction exactly. Be concise and factual.

   {{
   "summary": "[2-sentence factual summary]",
   "tags": ["tag1", "tag2", ...],
   "triplets": [
       {{"subject": "Entity", "predicate": "verb_phrase", "object": "Entity"}}
   ]
   }}
   """

        return self._call_model(sys_prompt, user_prompt)

    def generate_question_candidates(self, chunk: Dict, context_str: str) -> List[Dict]:
        estimated_tokens = chunk.get("estimated_tokens", 150)
        per_diff = get_questions_per_difficulty(estimated_tokens)
        content_type = chunk.get("content_type", "prose")

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

        sys_prompt = (
            "You are an Expert Technical Interviewer. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
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

### STRICT RULES (NEVER VIOLATE):
1. NEVER copy, reuse, or paraphrase any example question or phrasing that appears anywhere in this prompt.
2. Do NOT use the word "X" or the phrase "is a bottleneck" in any question unless those exact words are in the chunk.
3. Every "question" MUST be answerable using ONLY the TARGET TEXT CHUNK.
4. Every "source_quote" MUST be verbatim sentence(s) from the TARGET TEXT CHUNK that actually contain the answer (minimum 25 characters).
5. Never generate answers here.
6. Every question MUST target a DIFFERENT concept, fact, or mechanism from the chunk.

Output ONLY:
{{"qa_pairs": [ {{"question": "...", "source_quote": "...", "difficulty": "{difficulty}"}}, ... ] }}
"""

            result = self._call_model(sys_prompt, user_prompt)
            raw = result.get("qa_pairs", [])

            # Fallback: model emitted flat numbered keys
            if not raw and result:
                from collections import defaultdict

                groups: dict = defaultdict(dict)
                for k, v in result.items():
                    m = re.match(r"^(question|source_quote)(\d*)$", k)
                    if m:
                        groups[m.group(2)][m.group(1)] = v
                flat_items = [g for g in groups.values() if "question" in g]
                if flat_items:
                    logger.warning(
                        f"Pass 1 [{difficulty}]: model used flat keys — reconstructed {len(flat_items)} item(s)"
                    )
                    raw = flat_items

            count = 0
            for item in raw:
                if count >= per_diff:
                    break
                if not isinstance(item, dict):
                    continue
                item["difficulty"] = difficulty
                all_candidates.append(item)
                count += 1

            logger.debug(f"Pass 1 [{difficulty}]: {count} candidate(s)")

        return self._deduplicate_candidates(all_candidates)

    def generate_reference_answer(
        self, question: str, chunk_content: str, source_quote: str = ""
    ) -> str:
        """
        Generate a reference answer for *question* grounded in *chunk_content*.
        """
        sys_prompt = (
            "You are a knowledgeable technical interviewer writing model reference answers. "
            "Your answers should read like what a well-prepared human candidate would say — "
            "complete, explanatory, and grounded in the source text. "
            "Output ONLY the answer text — no preamble, no meta-commentary."
        )

        quote_block = (
            f"""
### KEY PASSAGE (this is the specific part of the source that answers the question):
{source_quote.strip()}

"""
            if source_quote and len(source_quote.strip()) >= 25
            else ""
        )

        user_prompt = f"""
### SOURCE TEXT:
{chunk_content}
{quote_block}
### QUESTION:
{question}

### INSTRUCTIONS:
Write a reference answer that a well-prepared human candidate would give in a technical interview.
{"Focus your answer ONLY on explaining the KEY PASSAGE above." if quote_block else ""}

Rules (must follow exactly):
1. Use ONLY information stated in the SOURCE TEXT. Do not add outside knowledge.
2. Do NOT repeat the question or its premise. Start directly with the substance.
3. Explain using the exact details from the text — no added commentary.
4. NO EXTRAPOLATION: Do not add "this helps", "this is important because", "researchers can", or any concluding sentence unless the text itself says it.
5. Write in plain, clear prose (maximum 3 sentences). No bullet points.
6. If the text gives specific values, thresholds, or named techniques, include them.
7. If the core subject is completely absent from the source text, output EXACTLY: NOT_ENOUGH_INFORMATION
"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                extra_body={"top_k": 50, "repetition_penalty": 1.05},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return ""


class EnrichmentManager:
    def __init__(self):
        logger.info("🚀 Starting Optimized Two-Pass Pipeline...")
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)
        self.validator = AnswerValidator()

    def _process_chunk(self, chunk: Dict, context_str: str) -> Tuple[bool, str]:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        tokens = chunk.get("estimated_tokens", 0)
        ctype = chunk.get("content_type", "prose")

        if chunk.get("should_use") != 1:
            logger.info(f"Chunk {chunk_id[:8]} has should_use=0 — skipping")
            return True, chunk.get("existing_summary", "")

        if chunk.get("existing_summary") and self.db.get_questions_for_chunk(chunk_id):
            logger.info(f"Chunk {chunk_id[:8]} already fully enriched — skipping")
            return True, chunk.get("existing_summary", "")

        console.rule(
            f"[bold cyan]Chunk {chunk_id[:8]}[/] | {tokens} tokens | type={ctype}"
        )

        logger.info("[bold]\\[1/3] Generating metadata...[/]")
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

        meta["tags"] = self.llm._normalize_tags(meta.get("tags", []))
        meta["tags"] = self.llm._verify_tags_grounded(meta["tags"], content)
        logger.info(f"[dim]Grounded tags:[/] {meta['tags']}")

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
                is_grounded, reason = _quote_is_grounded(source_quote, content)

                if is_grounded:
                    score_colour = "green"
                    console.print(
                        f"    Quote match: [{score_colour}]Passed ({reason})[/]"
                    )
                else:
                    console.print(
                        f"    [red]✗ REJECTED[/] — quote guard failed: {reason}"
                    )
                    rejected_qa.append(
                        {
                            "level": level,
                            "question": question,
                            "answer": "",
                            "reason": f"Weak quote: {reason}",
                        }
                    )
                    continue

            answer = self.llm.generate_reference_answer(question, content, source_quote)
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

            # ── Post-hoc quote relevance check ────────────────────────────────
            # The Pass 1 quote guard only verified the quote exists in the chunk.
            # Now that we have the answer, verify the quote actually contains the
            # answer content — not just the topic-introduction sentence.
            # If the answer and quote share very few content words, the model
            # picked the wrong sentence as the source (e.g. "there are two reasons"
            # instead of the sentences that describe what those reasons are).
            if source_quote and ctype != "math":
                answer_words = _content_words(answer)
                quote_words = _content_words(source_quote)
                if answer_words and quote_words:
                    overlap = len(answer_words & quote_words) / len(answer_words)
                    if overlap < 0.12:
                        # Quote and answer share almost no content words — Pass 1
                        # extracted a setup/intro sentence instead of the answer-bearing one.
                        #
                        # RESCUE: scan the chunk sentence by sentence and promote the one
                        # with the best content-word overlap to the answer. This stays fully
                        # grounded in verbatim document text — no hallucination risk.
                        sentences = [
                            s.strip()
                            for s in re.split(r"(?<=[.!?])\s+", content)
                            if len(s.strip()) >= 25
                        ]
                        best_sent, best_overlap = "", 0.0
                        for sent in sentences:
                            sent_words = _content_words(sent)
                            if sent_words:
                                sent_overlap = len(answer_words & sent_words) / len(
                                    answer_words
                                )
                                if sent_overlap > best_overlap:
                                    best_overlap, best_sent = sent_overlap, sent

                        RESCUE_THRESHOLD = 0.20
                        if best_overlap >= RESCUE_THRESHOLD and best_sent:
                            console.print(
                                f"    [yellow]⚠ RESCUED[/] — swapped quote to best-matching "
                                f"sentence ({best_overlap:.0%} overlap)"
                            )
                            source_quote = best_sent
                            cand["source_quote"] = best_sent
                            overlap = best_overlap
                        else:
                            console.print(
                                f"    [red]✗ REJECTED[/] — quote/answer mismatch "
                                f"({overlap:.0%} overlap); best rescue only "
                                f"{best_overlap:.0%}. Pass 1 grabbed the wrong sentence."
                            )
                            rejected_qa.append(
                                {
                                    "level": level,
                                    "question": question,
                                    "answer": answer,
                                    "reason": f"Quote/answer mismatch ({overlap:.0%} overlap)",
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

            if (
                success
                and summary
                and not DEFLECTION_PATTERN.search(summary)
                and len(summary.split()) >= MIN_SUMMARY_TOKENS
            ):
                history.append(summary)
            elif success and summary:
                logger.warning(
                    f"Summary for chunk {chunk.get('chunk_id', '')[:8]} failed quality gate"
                )

        logger.info(f"[green]✅ File {file_id[:8]} enrichment complete[/]")

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
                f"Workers: [cyan]{MAX_WORKERS}[/]  |  "  # B21: now 1
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
            with ThreadPoolExecutor(MAX_WORKERS) as ex:  # B21: MAX_WORKERS=1
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
