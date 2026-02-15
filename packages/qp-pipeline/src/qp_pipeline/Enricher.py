"""
qa_enricher.py
Grounded Q&A Enrichment Pipeline
--------------------------------
Features:
1. Metadata Extraction:
   - Summary & Triplets (LLM)
   - Tags (SpaCy - Broad Indexing Scope)
2. Context Awareness:
   - Prose: Rolling summaries.
   - Table/Code: Linked List neighbors.
3. Batch Q&A Generation: Generates candidates per difficulty.
4. Robust Validation:
   - NER-only Grounding Check (Prevents false positives on paraphrases).
   - "Triangle" Semantic Check (Q-Quote-Answer).
   - Deduplication against Vector Store.
5. Vector Indexing: Fault-tolerant ChromaDB integration.

Dependencies: spacy, sentence-transformers, openai, numpy, rapidfuzz, chromadb
"""

import json
import logging
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from openai import OpenAI
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# --- PROJECT PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from qp_core.DBManager import DBManager
except ImportError:
    print("Warning: DBManager not found. Using MockDB.")

    class DBManager:
        def __init__(self, p):
            pass

        def get_pending_files(self, l):
            return []

        def get_chunks_for_file_ordered(self, f):
            return []

        def save_questions(self, c, d):
            pass

        def save_enrichment(self, c, d):
            pass


try:
    from qp_core.VectorStore import QAVectorStore
except ImportError:
    print("Warning: QAVectorStore not found. Vector storage disabled.")

    class QAVectorStore:
        def __init__(self):
            pass

        def add_qa_pair(self, **kwargs):
            pass

        def find_similar_questions(self, **kwargs):
            return []

        def persist(self):
            pass

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = project_root / "data" / "rag_staging.db"
LLM_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "llama-model"
TIMEOUT = 120

# --- Validation Thresholds ---
SIMILARITY_THRESHOLD = 0.55
QUOTE_ANSWER_THRESHOLD = 0.60
QUESTION_QUOTE_THRESHOLD = 0.45
QUOTE_MATCH_THRESHOLD = 90.0
DEDUP_SIMILARITY_THRESHOLD = 0.85  # Skip Q if >85% similar to existing

HALLUCINATION_THRESHOLDS = {"Easy": 0.20, "Medium": 0.20, "Hard": 0.30}

# --- Generation Config ---
CANDIDATES_PER_LEVEL = 5
MAX_CONTEXT_HISTORY = 3

# --- Knowledge Graph Predicates ---
PREDICATE_WHITELIST = {
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
}

# --- Prompts ---
METADATA_SYSTEM_PROMPT = """
You are a Knowledge Extraction Engine.
1. Extract a concise technical summary.
2. Extract Knowledge Graph triplets explicitly stated in the text.
Output STRICT JSON.
"""

QA_SYSTEM_PROMPT = """
You are a strict technical interviewer.
1. GENERATE questions based ONLY on the provided text.
2. PROVIDE the exact quote from the text that supports your answer.
3. Avoid tautologies (e.g., "X is important because it is key").
4. OUTPUT valid JSON only.
"""

DIFFICULTY_PROMPTS = {
    "Easy": "Generate 5 'Easy' questions (factual recall, 2-4 sentence answers). For EACH question, provide the exact supporting quote from the text.",
    "Medium": "Generate 5 'Medium' questions (synthesis of two points, 3-5 sentence answers). For EACH question, provide the exact supporting quote from the text.",
    "Hard": "Generate 5 'Hard' questions (implications/reasoning, 3-5 sentence answers). For EACH question, provide the exact supporting quote from the text.",
}

# ============================================================
# LOGGING & MODELS
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("QAEnricher")

_embedding_model = None
_nlp = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading Bi-Encoder: BAAI/bge-small-en-v1.5...")
        _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedding_model


def get_nlp():
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy: en_core_web_sm...")
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error(
                "Model not found. Run: python -m spacy download en_core_web_sm"
            )
            sys.exit(1)
    return _nlp


# ============================================================
# UTILITIES
# ============================================================


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().split())


def extract_tags_for_indexing(text: str, top_n: int = 8) -> List[str]:
    """
    Broad extraction: Noun Chunks + Named Entities.
    Used for: Metadata Tags (Search Indexing).
    """
    nlp = get_nlp()
    doc = nlp(text)
    candidates = []

    # 1. Named Entities (Allowed types)
    allowed_labels = {"ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART", "EVENT", "LAW"}
    for ent in doc.ents:
        if ent.label_ in allowed_labels and len(ent.text) > 2:
            candidates.append(ent.text.lower().strip())

    # 2. Technical Noun Chunks
    ignored_starts = {"the", "a", "an", "this", "that", "these", "those", "my", "your"}
    for chunk in doc.noun_chunks:
        clean_chunk = chunk.text.lower().strip()
        words = clean_chunk.split()
        # Heuristic: 1-4 words, not starting with stopword, valid POS
        if 1 <= len(words) <= 4:
            if words[0] not in ignored_starts and chunk.root.pos_ in {"NOUN", "PROPN"}:
                candidates.append(clean_chunk)

    counts = Counter(candidates)
    return [tag for tag, _ in counts.most_common(top_n)]


def extract_entities_for_grounding(text: str) -> Set[str]:
    """
    Strict extraction: Named Entities ONLY.
    Used for: Hallucination Checks.
    Why: Prevents penalizing valid paraphrases (e.g. "scalability" vs "scales well").
    """
    nlp = get_nlp()
    doc = nlp(text)
    entities = set()

    # Only track specific Named Entities that MUST match exactly
    target_labels = {"ORG", "PRODUCT", "GPE", "PERSON", "EVENT", "LAW"}

    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities.add(ent.text.lower().strip())

    return entities


def compute_vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def validate_structure(answer: str) -> bool:
    nlp = get_nlp()
    doc = nlp(answer)
    sentences = [s for s in doc.sents if len(s.text.split()) > 3]
    if not (2 <= len(sentences) <= 6):
        return False
    words = [token for token in doc if not token.is_punct]
    if len(words) < 20:
        return False
    forbidden = ["...", "[insert]", "continued below", "etc."]
    if any(m in answer.lower() for m in forbidden):
        return False
    return True


def text_contains(text: str, snippet: str, threshold: float = 90.0) -> bool:
    if not snippet:
        return False
    norm_text = normalize_text(text)
    norm_snippet = normalize_text(snippet)
    if norm_snippet in norm_text:
        return True
    score = fuzz.partial_ratio(norm_snippet, norm_text)
    return score >= threshold


# ============================================================
# MAIN CLASS
# ============================================================


class QAEnricher:
    def __init__(self, db_manager: DBManager):
        self.db = db_manager
        self.client = OpenAI(base_url=LLM_API_URL, api_key="no-key")

        try:
            self.vector_store = QAVectorStore()
            logger.info("Vector Store initialized.")
        except Exception as e:
            logger.error(f"Failed to init Vector Store: {e}")
            self.vector_store = None

        get_embedding_model()
        get_nlp()

    # --------------------------------------------------------
    # STEP 1: METADATA EXTRACTION
    # --------------------------------------------------------

    def generate_metadata(self, chunk_content: str, context_history: str) -> Dict:
        """
        Extracts summary & triplets (LLM) and tags (SpaCy).
        """
        prompt = f"""
        ### PREVIOUS CONTEXT:
        {context_history if context_history else "(None)"}

        ### CHUNK TEXT:
        {chunk_content}

        ### INSTRUCTIONS:
        1. Summary: 2-3 sentences, technical tone. Start with the subject.
        2. Triplets: Extract relations explicitly stated. Predicates must be one of: {list(PREDICATE_WHITELIST)}

        ### OUTPUT JSON:
        {{
            "summary": "...",
            "triplets": [
                {{"subject": "...", "predicate": "...", "object": "..."}}
            ]
        }}
        """
        metadata = {}
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=TIMEOUT,
            )
            metadata = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Metadata generation failed: {e}")

        # SpaCy for Tags (Broad Indexing Scope)
        spacy_tags = extract_tags_for_indexing(chunk_content, top_n=8)
        metadata["tags"] = spacy_tags

        return metadata

    def validate_metadata(self, metadata: Dict, content: str) -> Dict:
        """
        Validates triplets against the chunk text.
        """
        valid_data = {
            "summary": metadata.get("summary", ""),
            "tags": metadata.get("tags", []),
            "triplets": [],
        }

        # Validate Triplets
        for triplet in metadata.get("triplets", []):
            subj = triplet.get("subject", "")
            obj = triplet.get("object", "")
            pred = triplet.get("predicate", "")

            if pred not in PREDICATE_WHITELIST:
                continue

            # Fuzzy check ensures triplets are grounded
            if text_contains(content, subj, 85) and text_contains(content, obj, 85):
                valid_data["triplets"].append(triplet)

        return valid_data

    # --------------------------------------------------------
    # STEP 2: Q&A GENERATION & VALIDATION
    # --------------------------------------------------------

    def generate_qa_candidates(
        self, context: str, difficulty: str, count: int = 5
    ) -> List[Dict]:
        instruction = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["Medium"])
        prompt = f"""
        Context:
        {context}

        Task: {instruction}

        Output JSON format (generate {count} questions):
        {{
            "qa_pairs": [
                {{
                    "question": "...",
                    "answer": "...",
                    "source_quote": "...",
                    "type": "factual"
                }}
            ]
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                timeout=TIMEOUT,
            )
            return json.loads(response.choices[0].message.content).get("qa_pairs", [])
        except Exception as e:
            logger.error(f"QA Generation error ({difficulty}): {e}")
            return []

    def validate_candidate(
        self,
        candidate: Dict,
        content_str: str,
        chunk_embedding: np.ndarray,
        chunk_grounding_ents: Set[str],
        difficulty: str,
    ) -> Tuple[bool, Dict]:
        q_text = candidate.get("question", "").strip()
        a_text = candidate.get("answer", "").strip()
        quote = candidate.get("source_quote", "").strip()
        metrics = {}

        # 1. Claim Grounding
        if not quote:
            return False, {}
        norm_quote, norm_content = normalize_text(quote), normalize_text(content_str)

        if norm_quote in norm_content:
            match_score = 100.0
        else:
            match_score = fuzz.partial_ratio(norm_quote, norm_content)

        metrics["quote_match_score"] = match_score
        if match_score < QUOTE_MATCH_THRESHOLD:
            logger.debug(f"Rejected {difficulty}: Quote Mismatch ({match_score})")
            return False, {}

        # 2. Embedding Checks (Q, A, Quote)
        try:
            # Efficient: Encode candidates here, reuse chunk_embedding
            vectors = get_embedding_model().encode([q_text, a_text, quote])
            q_vec, a_vec, quote_vec = vectors[0], vectors[1], vectors[2]
        except:
            return False, {}

        if compute_vector_similarity(quote_vec, a_vec) < QUOTE_ANSWER_THRESHOLD:
            return False, {}
        if compute_vector_similarity(q_vec, quote_vec) < QUESTION_QUOTE_THRESHOLD:
            return False, {}
        if compute_vector_similarity(a_vec, chunk_embedding) < SIMILARITY_THRESHOLD:
            return False, {}

        # 3. Structure
        if not validate_structure(a_text):
            return False, {}

        # 4. Entity Hallucination (Strict NER only)
        ans_ents = extract_entities_for_grounding(a_text)
        novel = ans_ents - chunk_grounding_ents

        # Fix: Adjusted denominator to prevent explosion on short answers
        denominator = max(len(ans_ents), 3)
        hall_score = len(novel) / denominator

        if hall_score > HALLUCINATION_THRESHOLDS.get(difficulty, 0.20):
            logger.debug(f"Rejected {difficulty}: Hallucination ({hall_score:.2f})")
            return False, {}

        metrics["hallucination"] = hall_score
        return True, metrics

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # --------------------------------------------------------

    def process_chunk(self, chunk: Dict, context_history: str) -> Tuple[bool, str]:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        logger.info(f"--- Processing Chunk {chunk_id} ---")

        # --- A. METADATA EXTRACTION ---
        raw_meta = self.generate_metadata(content, context_history)
        valid_meta = self.validate_metadata(raw_meta, content)
        self.db.save_enrichment(chunk_id, valid_meta)

        chunk_summary = valid_meta.get("summary", "")
        if not chunk_summary:
            chunk_summary = content[:200] + "..."

        # --- B. Q&A GENERATION ---

        # PRE-COMPUTE CHUNK ARTIFACTS ONCE
        # 1. Chunk Embedding
        chunk_embedding = get_embedding_model().encode(content)
        # 2. Strict Grounding Entities (NER Only)
        chunk_grounding_ents = extract_entities_for_grounding(content)

        valid_questions = []
        type_map = {
            "Concept": "Fact",
            "Definition": "Fact",
            "Explanation": "Mechanism",
            "Process": "Mechanism",
            "Insight": "Critical",
            "Trade-off": "Critical",
        }

        for level in ["Easy", "Medium", "Hard"]:
            candidates = self.generate_qa_candidates(
                context_history + "\n\n" + content, level, CANDIDATES_PER_LEVEL
            )

            for cand in candidates:
                q_text = cand.get("question", "").strip()

                # --- DEDUPLICATION CHECK ---
                if self.vector_store:
                    similar_qs = self.vector_store.find_similar_questions(
                        q_text, top_k=1, min_similarity=DEDUP_SIMILARITY_THRESHOLD
                    )
                    if similar_qs:
                        logger.debug(f"Skipping duplicate: {q_text[:30]}...")
                        continue

                # --- VALIDATION ---
                is_valid, metrics = self.validate_candidate(
                    cand, content, chunk_embedding, chunk_grounding_ents, level
                )

                if is_valid:
                    q_type = cand.get("type", "factual").capitalize()
                    q_type = type_map.get(q_type, q_type)
                    if q_type not in ["Fact", "Mechanism", "Critical"]:
                        q_type = "Fact"

                    qa_object = {
                        "question_text": q_text,
                        "answer_text": cand.get("answer", "").strip(),
                        "source_quote": cand.get("source_quote", "").strip(),
                        "difficulty": level,
                        "question_type": q_type,
                        "metrics": metrics,
                        "tags": valid_meta.get("tags", []),
                    }
                    valid_questions.append(qa_object)
                    logger.info(f"✓ Accepted {level} Q")

                    # --- C. VECTOR STORE INDEXING (Fault Tolerant) ---
                    if self.vector_store:
                        try:
                            self.vector_store.add_qa_pair(
                                chunk_id=chunk_id,
                                question_text=qa_object["question_text"],
                                answer_text=qa_object["answer_text"],
                                source_quote=qa_object["source_quote"],
                                difficulty=level,
                                question_type=qa_object["question_type"],
                                tags=qa_object["tags"],
                                hallucination_score=metrics["hallucination"],
                            )
                            # Persist after successful add to avoid data loss on crash
                            self.vector_store.persist()
                        except Exception as e:
                            logger.error(f"Vector Store Insertion Failed: {e}")

                    break  # Take best 1 per level

        if valid_questions:
            self.db.save_questions(chunk_id, valid_questions)
            return True, chunk_summary

        return False, chunk_summary

    def run_pipeline(self):
        logger.info("Starting Enrichment Pipeline...")

        while True:
            files = self.db.get_pending_files(limit=1)
            if not files:
                logger.info("No pending files.")
                break

            for file_id in files:
                logger.info(f"=== File: {file_id} ===")
                chunks = self.db.get_chunks_for_file_ordered(file_id)
                chunk_map = {c["chunk_id"]: c for c in chunks}
                history_queue = deque(maxlen=MAX_CONTEXT_HISTORY)

                for chunk in chunks:
                    ctype = chunk.get("content_type", "prose")
                    context_str = ""

                    if ctype in ["table", "math", "code"]:
                        parts = []
                        prev_id = chunk.get("prev_chunk_id")
                        if prev_id and prev_id in chunk_map:
                            parts.append(
                                f"### PREVIOUS TEXT:\n{chunk_map[prev_id]['content']}"
                            )
                        next_id = chunk.get("next_chunk_id")
                        if next_id and next_id in chunk_map:
                            parts.append(
                                f"### FOLLOWING TEXT:\n{chunk_map[next_id]['content']}"
                            )
                        context_str = "\n\n".join(parts)

                    if not context_str:
                        context_str = "\n".join([f"- {s}" for s in history_queue])

                    success, summary = self.process_chunk(chunk, context_str)
                    if success and summary:
                        history_queue.append(summary)


def main():
    db = DBManager(DB_PATH)
    enricher = QAEnricher(db)
    enricher.run_pipeline()


if __name__ == "__main__":
    main()
