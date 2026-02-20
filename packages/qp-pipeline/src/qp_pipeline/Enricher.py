"""
Enricher.py - Production-Grade Enrichment Pipeline
Generator: LFM 2.5 (1.2b) | Validator: BGE-Small (Heuristic)
Backend-Ready Version
"""

import json
import logging
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Set, Tuple

from openai import OpenAI
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# --- PROJECT PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DB_PATH = str(project_root / "data" / "rag_staging.db")
CHROMA_DIR = str(project_root / "data" / "chroma_store")

try:
    from qp_core.DBManager import DBManager
    from qp_core.VectorStore import QAVectorStore
except ImportError:
    print("Import errors")


# ---------------- CONFIG ----------------
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "lfm-2.5-1.2b"
MAX_WORKERS = 4
QUOTE_MATCH_THRESHOLD = 65.0  # ← Lowered for small model (raise back to 75 later)

PREDICATE_WHITELIST = [
    "is_a",
    "part_of",
    "contains",
    "causes",
    "prevents",
    "optimizes",
    "requires",
    "enables",
    "produces",
    "uses",
    "calls",
    "inherits_from",
    "defined_as",
    "has_property",
    "critiques",
    "contrasts_with",
    "limits",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Enricher")


# ---------------- DYNAMIC QUESTION LIMIT ----------------
def get_max_questions(estimated_tokens: int, difficulty: str) -> int:
    if estimated_tokens < 80:
        base = 2
    elif estimated_tokens < 200:
        base = 3
    elif estimated_tokens < 400:
        base = 5
    elif estimated_tokens < 700:
        base = 7
    else:
        base = 9

    if difficulty == "Easy":
        return min(10, base + 1)
    elif difficulty == "Hard":
        return max(2, base - 1)
    else:  # Medium
        return base


# ---------------- ANSWER VALIDATOR ----------------
class AnswerValidator:
    SIMILARITY_THRESHOLD = 0.48
    LEXICAL_OVERLAP_THRESHOLD = 0.25
    MIN_SENTENCES = 1
    MAX_SENTENCES = 5

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading Validator Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def _sentence_count(self, text: str) -> int:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return len([s for s in sentences if len(s) > 5])

    def _content_words(self, text: str) -> Set[str]:
        return {w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)}

    def validate(self, answer: str, chunk: str) -> Tuple[bool, str]:
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

        return True, ""

    def _structural_check(self, answer: str) -> Tuple[bool, int]:
        count = self._sentence_count(answer)
        return self.MIN_SENTENCES <= count <= self.MAX_SENTENCES, count


# ---------------- LLM CLIENT ----------------
class LLMClient:
    def __init__(self, base_url, api_key="no-key"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            braces_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if braces_match:
                return json.loads(braces_match.group(1))
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON Parse Error: {e} | Raw: {text[:100]}...")
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
                response_format={"type": "json_object"},
            )
            return self._extract_json(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return {}

    def generate_metadata(self, text: str, context: str) -> Dict[str, Any]:
        logger.debug(f"Generating structured metadata for chunk ({len(text)} chars)")
        sys = (
            "You are a precise Technical Knowledge Graph Extractor. "
            "Always output valid JSON following the exact schema below. "
            "Do not add any extra keys or explanations."
        )
        user = f"""
### CONTEXT (for understanding only):
{context}

### TEXT CHUNK:
{text}

### TASK:
Extract metadata in the following EXACT JSON structure:
{{
  "summary": "2-3 sentence summary. Start with the main SUBJECT. Capture the core technical content.",
  "tags": ["NounTag1", "NounTag2", ...],
  "triplets": [
    {{"subject": "Entity", "predicate": "allowed_predicate", "object": "Entity"}}
  ]
}}

### EXTRA RULES:
1. Summary: 2-3 sentences maximum. Always begin with the main subject.
2. Tags: Exactly 5-8 clean noun/phrase tags.
3. Triplets: Only explicitly stated ones using allowed predicates. Empty array if none.

Return ONLY the JSON object.
"""
        return self._call_model(sys, user)

    def generate_questions(
        self, chunk: Dict, context_str: str, difficulty: str
    ) -> List[Dict[str, Any]]:
        estimated_tokens = chunk.get("estimated_tokens", 150)
        max_q = get_max_questions(estimated_tokens, difficulty)

        prompts = {
            "Easy": "Factual Recall (Type: Fact)",
            "Medium": "Conceptual (Type: Mechanism)",
            "Hard": "Analytical Risks/Significance (Type: Critical)",
        }

        sys = "You are an Expert Technical Interviewer. Output ONLY valid JSON. Never leave fields empty."

        user = f"""
### SECTION CONTEXT (for understanding only):
{context_str}

### TEXT CHUNK:
{chunk["content"]}

### TASK:
Generate **UP TO {max_q}** '{difficulty}' Q/A pairs. {prompts[difficulty]}

### STRICT RULES (MUST FOLLOW):
1. Every object MUST contain three NON-EMPTY strings: "question", "answer", "source_quote"
2. "answer" must be a complete, meaningful sentence (minimum 15 words).
3. "source_quote" must be copied VERBATIM from the TEXT CHUNK above (at least 20 characters, exact match — do NOT paraphrase).
4. If you cannot create a high-quality pair, return fewer items — never return empty fields.

Output ONLY this structure:
{{"qa_pairs": [ {{"question": "...", "answer": "...", "source_quote": "..."}}, ... ] }}
"""

        result = self._call_model(sys, user)
        logger.debug(
            f"Raw LLM output for {difficulty}:\n{json.dumps(result, indent=2)}"
        )

        qa_pairs = result.get("qa_pairs", [])[:max_q]

        # Extra safety: remove any with empty answer or too-short quote
        cleaned = []
        for qa in qa_pairs:
            ans = qa.get("answer", "").strip()
            quote = qa.get("source_quote", "").strip()
            if ans and len(quote) >= 20:
                cleaned.append(qa)
        return cleaned


# ---------------- ENRICHMENT MANAGER ----------------
class EnrichmentManager:
    def __init__(self):
        logger.info("🚀 Initializing Enrichment Pipeline...")
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)
        self.validator = AnswerValidator()
        self.vector_store = QAVectorStore(chroma_path=CHROMA_DIR)

    def process_chunk(self, chunk: Dict, context_str: str) -> Tuple[bool, str]:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        tokens = chunk.get("estimated_tokens", 0)
        ctype = chunk.get("content_type", "prose")

        logger.info(
            f"  → Processing chunk {chunk_id[:8]} | {tokens} tokens | type={ctype}"
        )

        meta = self.llm.generate_metadata(content, context_str)
        if not meta:
            logger.warning(f"    Metadata generation FAILED for chunk {chunk_id[:8]}")
            self.db.save_rejections(
                chunk_id, [{"reason": "Metadata generation failed"}]
            )
            return False, ""

        logger.info(
            f"    Metadata OK → {len(meta.get('tags', []))} tags, "
            f"{len(meta.get('triplets', []))} triplets"
        )

        all_valid_qa = []
        rejected_qa = []

        for level in ["Easy", "Medium", "Hard"]:
            logger.debug(f"    Generating {level} questions...")
            candidates = self.llm.generate_questions(chunk, context_str, level)
            logger.info(f"    {level}: Generated {len(candidates)} candidate(s)")

            for i, cand in enumerate(candidates):
                q_text = cand.get("question", "")[:80]
                answer = cand.get("answer", "").strip()
                quote = cand.get("source_quote", "").strip()

                # Early empty answer filter
                if not answer:
                    rejected_qa.append(
                        {
                            "level": level,
                            "question": cand.get("question"),
                            "reason": "Empty answer from LLM",
                        }
                    )
                    logger.debug(
                        f"      [{i + 1}] Rejected (empty answer) → {q_text}..."
                    )
                    continue

                quote_score = fuzz.partial_ratio(quote.lower(), content.lower())
                if quote_score < QUOTE_MATCH_THRESHOLD:
                    reason = f"Quote match failed ({quote_score:.1f}%)"
                    rejected_qa.append(
                        {
                            "level": level,
                            "question": cand.get("question"),
                            "reason": reason,
                        }
                    )
                    logger.debug(f"      [{i + 1}] Rejected (quote) → {q_text}...")
                    continue

                is_valid, reason = self.validator.validate(answer, content)

                if is_valid:
                    cand.update(
                        {
                            "difficulty": level,
                            "type": {
                                "Easy": "Fact",
                                "Medium": "Mechanism",
                                "Hard": "Critical",
                            }[level],
                        }
                    )
                    all_valid_qa.append(cand)
                    logger.debug(f"      [{i + 1}] ACCEPTED {level} QA → {q_text}...")
                else:
                    embeddings = self.validator.model.encode(
                        [content, answer], convert_to_tensor=True
                    )
                    sem_score = float(util.cos_sim(embeddings[0], embeddings[1]))
                    rejected_qa.append(
                        {
                            "level": level,
                            "question": cand.get("question"),
                            "reason": reason,
                            "semantic_score": round(sem_score, 3),
                        }
                    )
                    logger.debug(f"      [{i + 1}] Rejected ({reason}) → {q_text}...")

        self.db.save_enrichment(chunk_id, meta)

        if all_valid_qa:
            self.db.save_questions(chunk_id, all_valid_qa)
            for qa in all_valid_qa:
                self.vector_store.add_qa_pair(
                    chunk_id=chunk_id,
                    question_text=qa.get("question", ""),
                    answer_text=qa.get("answer", ""),
                    source_quote=qa.get("source_quote", ""),
                    difficulty=qa.get("difficulty", ""),
                    question_type=qa.get("type", ""),
                    tags=meta.get("tags", []),
                    generation_score=qa.get("generation_score"),
                )
            logger.info(
                f"    Saved {len(all_valid_qa)} valid questions to DB + VectorStore"
            )

        if rejected_qa:
            self.db.save_rejections(chunk_id, rejected_qa)
            logger.info(f"    Saved {len(rejected_qa)} rejected items for auditing")

        logger.info(
            f"  Chunk {chunk_id[:8]} finished → {len(all_valid_qa)} accepted, {len(rejected_qa)} rejected"
        )
        return True, meta.get("summary", "")

    def process_file(self, file_id: str):
        logger.info(f"📂 Starting file {file_id[:8]}...")
        chunks = self.db.get_chunks_for_file_ordered(file_id)
        logger.info(f"   Loaded {len(chunks)} chunks")

        history = deque(maxlen=3)

        for i, chunk in enumerate(chunks):
            content, ctype = chunk["content"], chunk.get("content_type", "prose")

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
        logger.info(f"✅ File {file_id[:8]} completed | Vector store persisted")

    def run(self):
        logger.info("🚀 Enrichment Pipeline started")
        while True:
            files = self.db.get_pending_files(limit=5)
            if not files:
                logger.info("No more pending files. Pipeline finished.")
                break

            logger.info(f"Found {len(files)} pending file(s)")
            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = {ex.submit(self.process_file, fid): fid for fid in files}
                for future in as_completed(futures):
                    fid = futures[future]
                    try:
                        future.result()
                        logger.info(f"File {fid[:8]} processed successfully")
                    except Exception as e:
                        logger.error(f"Thread crash on file {fid[:8]}: {e}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("          STARTING ENRICHMENT PIPELINE")
    logger.info("=" * 80)
    EnrichmentManager().run()
