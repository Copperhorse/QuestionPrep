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

import numpy as np
from json_repair import repair_json
from openai import OpenAI
from rapidfuzz import fuzz
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from sentence_transformers import CrossEncoder, SentenceTransformer, util

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

MAX_WORKERS = 1

QUOTE_MATCH_THRESHOLD = 60.0
DEDUP_SIMILARITY_THRESHOLD = 80
MIN_TOKENS_FOR_METADATA = 30
MIN_TOKENS_FOR_QA = 30
MIN_SUMMARY_TOKENS = 10
QUOTE_MIN_CHARS = 25  # Relaxed from 40

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


def _quote_is_grounded(quote: str, content: str) -> tuple[bool, str]:
    quote = quote.strip()
    if len(quote) < QUOTE_MIN_CHARS:
        return False, f"Quote too short ({len(quote)} chars, min {QUOTE_MIN_CHARS})"

    if quote.lower() in content.lower():
        return True, "exact"

    score = fuzz.partial_ratio(quote.lower(), content.lower())
    if score >= QUOTE_MATCH_THRESHOLD:
        return True, f"fuzzy ({score:.0f}%)"

    return False, f"not found ({score:.0f}%)"


def _content_words(t: str):
    return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", t)}


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        logger.warning("JSON parsed as single-element list — unwrapping to dict")
        return value[0]
    if isinstance(value, list) and value:
        logger.warning(
            f"JSON parsed as list with {len(value)} elements — expected a dict. Returning empty."
        )
    return {}


def get_questions_per_difficulty(estimated_tokens: int) -> int:
    if estimated_tokens < 80:
        return 1
    elif estimated_tokens < 200:
        return 2
    else:
        return 3


class AnswerValidator:
    SIMILARITY_THRESHOLD = 0.50
    LEXICAL_OVERLAP_THRESHOLD = (
        0.30  # Baseline gate: must have at least 30% word overlap
    )
    ENTAILMENT_THRESHOLD = 0.60
    CONTRADICTION_THRESHOLD = 0.60
    MIN_SENTENCES = 1
    MAX_SENTENCES = 5

    def __init__(
        self,
        bi_model_name: str = "BAAI/bge-small-en-v1.5",
        ce_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
    ):
        logger.info(f"Loading Validator Bi-Encoder: {bi_model_name}...")
        self.bi_encoder = SentenceTransformer(bi_model_name)

        logger.info(f"Loading Validator Cross-Encoder (NLI): {ce_model_name}...")
        self.cross_encoder = CrossEncoder(ce_model_name)

    def validate(
        self, question: str, answer: str, chunk: str, source_quote: str
    ) -> Tuple[bool, str]:
        if not answer or not answer.strip():
            return False, "Empty answer"

        struct_ok, sent_count = self._structural_check(answer)
        if not struct_ok:
            return False, f"Structural fail: {sent_count} sentences"

        # GATE 1: Lexical (Always run)
        c_words = self._content_words(chunk)
        a_words = self._content_words(answer)
        if not a_words:
            return False, "No content words in answer"

        overlap = len(c_words & a_words) / len(a_words)
        if overlap < self.LEXICAL_OVERLAP_THRESHOLD:
            return (
                False,
                f"Lexical fail: overlap={overlap:.2f} (gate is {self.LEXICAL_OVERLAP_THRESHOLD})",
            )

        # GATE 2: Bi-Encoder Similarity (Answer vs Quote & Answer vs Chunk)
        embeddings = self.bi_encoder.encode(
            [answer, source_quote, chunk], convert_to_tensor=True
        )
        ans_emb, quote_emb, chunk_emb = embeddings[0], embeddings[1], embeddings[2]

        # Use .item() to safely extract the scalar value and avoid PyTorch ValueError
        sim_ans_quote = util.cos_sim(ans_emb, quote_emb).item()
        sim_ans_chunk = util.cos_sim(ans_emb, chunk_emb).item()

        bi_score = max(sim_ans_quote, sim_ans_chunk)

        # GATE 3: Cross-Encoder NLI (Answer vs Source Quote)
        logits = self.cross_encoder.predict([[source_quote, answer]])
        e = np.exp(np.array(logits[0], dtype=np.float64) - np.max(logits[0]))
        probs = e / e.sum()

        p_contra = probs[0]
        p_entail = probs[1]
        p_neutral = probs[2]

        if p_contra > self.CONTRADICTION_THRESHOLD:
            return (
                False,
                f"NLI Contradiction ({p_contra:.2f} > {self.CONTRADICTION_THRESHOLD})",
            )

        if p_entail > self.ENTAILMENT_THRESHOLD:
            if bi_score < 0.40:  # Extremely lenient bi-encoder floor
                return (
                    False,
                    f"Entailed ({p_entail:.2f}), but bi-score too low ({bi_score:.2f})",
                )

        else:
            adjusted_bi_score = bi_score * (0.50 + 0.50 * p_entail)
            if adjusted_bi_score < self.SIMILARITY_THRESHOLD:
                return (
                    False,
                    f"Neutral NLI ({p_neutral:.2f}) dragged down bi-score: {adjusted_bi_score:.2f} < {self.SIMILARITY_THRESHOLD}",
                )

        # GATE 4: Question Heuristics
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
            answer_words_set = set(answer.lower().split())
            if not any(
                w in answer.lower() if " " in w else w in answer_words_set
                for w in causal
            ):
                return False, "Why/How-question lacks causal/mechanism phrase"

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
        fenced = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
        raw = fenced.group(1) if fenced else text

        try:
            parsed = json.loads(raw)
            result = _ensure_dict(parsed)
            if result:
                return result
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
                ],
                temperature=0.1,
                extra_body={
                    "top_k": 50,
                    "repetition_penalty": 1.05,
                },
            )
            content = response.choices[0].message.content
            return self._extract_json(content)

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        seen_qs: List[str] = []
        seen_quotes: List[str] = []
        unique: List[Dict] = []

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            q = cand.get("question", "").lower().strip()
            quote = cand.get("source_quote", "").lower().strip()
            if not q:
                continue

            if any(
                fuzz.token_sort_ratio(q, s) > DEDUP_SIMILARITY_THRESHOLD
                for s in seen_qs
            ):
                continue

            if quote and any(fuzz.partial_ratio(quote, sq) > 90 for sq in seen_quotes):
                continue

            seen_qs.append(q)
            if quote:
                seen_quotes.append(quote)
            unique.append(cand)

        return unique

    @staticmethod
    def _normalize_tags(tags: List[str]) -> List[str]:
        GENERIC_BLOCKLIST = {" "}

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
            if normalised in GENERIC_BLOCKLIST or normalised in seen:
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
        return grounded

    @staticmethod
    def _verify_triplets_grounded(triplets: List[Dict], text: str) -> List[Dict]:
        text_lower = text.lower()
        grounded = []
        for t in triplets:
            if not isinstance(t, dict):
                continue
            subj = t.get("subject", "").lower().strip()
            obj = t.get("object", "").lower().strip()
            if subj and obj and subj in text_lower and obj in text_lower:
                grounded.append(t)
        return grounded

    def generate_metadata(
        self,
        text: str,
        context: str,
        estimated_tokens: int = 150,
        content_type: str = "prose",
    ) -> Dict[str, Any]:
        if estimated_tokens < MIN_TOKENS_FOR_METADATA:
            return {}

        sys_prompt = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "If you reason before answering, keep each reasoning step to 5 words or fewer. "
            "Stop reasoning the moment you have identified the tags, triplets, and summary — "
            "do not verify them a second time. "
            "Extract information ONLY from the TEXT CHUNK — never from context. "
            "Always output valid JSON following the exact schema provided."
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
   3. Summary: Write exactly 2 complete sentences. Sentence 1: state the core technical concept or claim. Sentence 2: name the specific methods, techniques, or entities mentioned. Never start with 'The text', 'This chunk', or 'This section'.

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

            if not raw and result:
                from collections import defaultdict

                groups: dict = defaultdict(dict)
                for k, v in result.items():
                    m = re.match(r"^(question|source_quote)(\d*)$", k)
                    if m:
                        groups[m.group(2)][m.group(1)] = v
                flat_items = [g for g in groups.values() if "question" in g]
                if flat_items:
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

        return self._deduplicate_candidates(all_candidates)

    def generate_reference_answer(
        self, question: str, chunk_content: str, source_quote: str = ""
    ) -> str:
        sys_prompt = (
            "You are a knowledgeable technical interviewer writing model reference answers. "
            "Your answers should read like what a well-prepared human candidate would say — "
            "complete, explanatory, and grounded in the source text. "
            "Output ONLY the answer text — no preamble, no meta-commentary."
        )

        quote_block = (
            f"\n### KEY PASSAGE (this is the specific part of the source that answers the question):\n{source_quote.strip()}\n"
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
            return True, chunk.get("existing_summary", "")

        if chunk.get("existing_summary") and self.db.get_questions_for_chunk(chunk_id):
            return True, chunk.get("existing_summary", "")

        console.rule(
            f"[bold cyan]Chunk {chunk_id[:8]}[/] | {tokens} tokens | type={ctype}"
        )

        meta = self.llm.generate_metadata(
            content, context_str, estimated_tokens=tokens, content_type=ctype
        )
        if not meta:
            if tokens < MIN_TOKENS_FOR_METADATA:
                return True, ""
            self.db.save_rejections(
                chunk_id, [{"reason": "Metadata generation failed"}]
            )
            return False, ""

        meta["tags"] = self.llm._normalize_tags(meta.get("tags", []))
        meta["tags"] = self.llm._verify_tags_grounded(meta["tags"], content)
        meta["triplets"] = self.llm._verify_triplets_grounded(
            meta.get("triplets", []), content
        )
        self.db.save_enrichment(chunk_id, meta)

        if tokens < MIN_TOKENS_FOR_QA:
            return True, meta.get("summary", "")

        candidates = self.llm.generate_question_candidates(chunk, context_str)
        all_valid_qa: List[Dict] = []
        rejected_qa: List[Dict] = []

        for i, cand in enumerate(candidates):
            question = cand.get("question", "").strip()
            source_quote = cand.get("source_quote", "").strip()
            level = cand.get("difficulty", "Medium")

            if not question or len(source_quote) < 25:
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
                if source_quote.strip() not in content:
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
                if not is_grounded:
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

            if (
                not answer
                or "NOT ENOUGH INFORMATION" in answer.replace("_", " ").upper()
            ):
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": answer,
                        "reason": "Pass 2 refused",
                    }
                )
                continue

            if source_quote and ctype != "math":
                answer_words = _content_words(answer)
                quote_words = _content_words(source_quote)
                if answer_words and quote_words:
                    overlap = len(answer_words & quote_words) / len(answer_words)
                    if overlap < 0.12:
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
                        if best_overlap >= 0.20 and best_sent:
                            source_quote = best_sent
                            cand["source_quote"] = best_sent
                        else:
                            rejected_qa.append(
                                {
                                    "level": level,
                                    "question": question,
                                    "answer": answer,
                                    "reason": f"Quote/answer mismatch",
                                }
                            )
                            continue

            if DEFLECTION_PATTERN.search(answer):
                rejected_qa.append(
                    {
                        "level": level,
                        "question": question,
                        "answer": answer,
                        "reason": "Deflection",
                    }
                )
                continue

            is_valid, reason = self.validator.validate(
                question, answer, content, source_quote
            )
            if is_valid:
                cand["answer"] = answer
                cand["type"] = DIFFICULTY_TYPE.get(level, "Fact")
                all_valid_qa.append(cand)
            else:
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

        return True, meta.get("summary", "")

    def enrich_single_file(self, file_id: str) -> None:
        chunks = self.db.get_chunks_for_file_ordered(file_id)
        if not chunks:
            return

        chunk_map = {c["chunk_id"]: c["content"] for c in chunks}
        history: Deque[str] = deque(maxlen=3)

        for chunk in chunks:
            ctype = chunk.get("content_type", "prose")
            if ctype in ["table", "math", "code"]:
                parts = []
                prev_id = chunk.get("prev_chunk_id")
                if prev_id and prev_id in chunk_map:
                    parts.append(f"PREV: {chunk_map[prev_id][:150]}")
                next_id = chunk.get("next_chunk_id")
                if next_id and next_id in chunk_map:
                    parts.append(f"NEXT: {chunk_map[next_id][:150]}")
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

    def run(self) -> None:
        while True:
            files = self.db.get_pending_files(limit=5)
            if not files:
                break
            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = {
                    ex.submit(self.enrich_single_file, fid): fid for fid in files
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Thread crash: {e}")


if __name__ == "__main__":
    EnrichmentManager().run()
