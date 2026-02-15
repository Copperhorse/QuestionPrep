"""
qa_vector_store.py
Chroma-backed Vector Store for Interview Q/A Evaluation
-------------------------------------------------------

Purpose:
- Store reference Q/A pairs
- Enable question deduplication
- Enable semantic scoring vs user answers
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "qa_pairs"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ---------------- INIT ----------------

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedding_model


def embed(text: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(text).tolist()


class QAVectorStore:
    def __init__(self):
        self.client = chromadb.Client(
            Settings(
                persist_directory=CHROMA_DIR,
                anonymized_telemetry=False,
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"purpose": "interview_qa_evaluation"}
        )

    # --------------------------------------------------
    # INSERT
    # --------------------------------------------------

    def add_qa_pair(
        self,
        chunk_id: str,
        question_text: str,
        answer_text: str,
        source_quote: str,
        difficulty: str,
        question_type: str,
        tags: Optional[List[str]] = None,
        generation_score: Optional[float] = None,
        hallucination_score: Optional[float] = None,
    ):
        question_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        metadata = {
            "chunk_id": chunk_id,
            "question_id": question_id,
            "difficulty": difficulty,
            "question_type": question_type,
            "source_quote": source_quote,
            "tags": tags or [],
            "generation_score": generation_score,
            "hallucination_score": hallucination_score,
            "created_at": created_at,
        }

        # We store QUESTION embeddings as the primary vector
        self.collection.add(
            ids=[question_id],
            documents=[question_text],
            embeddings=[embed(question_text)],
            metadatas=[metadata],
        )

        # Optional: store answer embedding separately if you want
        # deterministic scoring later without recomputing
        self.collection.add(
            ids=[f"{question_id}::answer"],
            documents=[answer_text],
            embeddings=[embed(answer_text)],
            metadatas={
                **metadata,
                "role": "reference_answer",
            },
        )

    # --------------------------------------------------
    # QUERY (DEDUP / NAVIGATION)
    # --------------------------------------------------

    def find_similar_questions(
        self,
        question_text: str,
        top_k: int = 5,
        min_similarity: float = 0.85,
    ):
        results = self.collection.query(
            query_embeddings=[embed(question_text)],
            n_results=top_k,
        )

        filtered = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= min_similarity:
                filtered.append(
                    {
                        "question": doc,
                        "metadata": meta,
                        "similarity": round(similarity, 3),
                    }
                )

        return filtered

    def persist(self):
        self.client.persist()
