"""
qa_vector_store.py
Chroma-backed Vector Store for Interview Q/A Evaluation
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb

# No longer needed for modern PersistentClient
# from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("QAVectorStore")

# ---------------- CONFIG ----------------
# Default fallback only. The Enricher should pass the real path.
DEFAULT_CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "qa_pairs"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class QAVectorStore:
    def __init__(self, chroma_path: str = DEFAULT_CHROMA_DIR, embedding_model=None):
        """
        Args:
            chroma_path: Path to the persistent directory.
            embedding_model: Optional pre-loaded SentenceTransformer instance.
                             If None, it loads its own.
        """
        logger.info(f"🔌 Connecting to ChromaDB at: {chroma_path}")

        # 1. Use PersistentClient (Modern API)
        self.client = chromadb.PersistentClient(path=chroma_path)

        # 2. Get/Create Collection with Cosine Similarity
        # (Default is L2, but Cosine is usually better for text semantic search)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        # 3. Model Management (Avoid double loading)
        if embedding_model:
            self.model = embedding_model
        else:
            logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}...")
            self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def _embed(self, text: str) -> List[float]:
        # Ensure we return a standard python list, not numpy array
        return self.model.encode(text).tolist()

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

        # Chroma metadata must be flat primitives (str, int, float, bool)
        # Lists (like tags) usually need to be joined as strings or handled carefully.
        # Newer Chroma versions support lists, but comma-joined strings are safer for compatibility.
        tags_str = ",".join(tags) if tags else ""

        metadata = {
            "chunk_id": chunk_id,
            "difficulty": difficulty,
            "question_type": question_type,
            "source_quote": source_quote,
            "tags": tags_str,
            "created_at": created_at,
            "type": "question",
        }

        # Add scores only if they exist (None values can cause errors in some DB versions)
        if generation_score is not None:
            metadata["generation_score"] = generation_score
        if hallucination_score is not None:
            metadata["hallucination_score"] = hallucination_score

        # 1. Store QUESTION (Primary Vector)
        self.collection.add(
            ids=[question_id],
            documents=[question_text],
            embeddings=[self._embed(question_text)],
            metadatas=[metadata],
        )

        # 2. Store ANSWER (Secondary Vector - Optional but useful for reverse lookup)
        # We modify the metadata to indicate it's an answer
        answer_meta = metadata.copy()
        answer_meta["type"] = "reference_answer"
        answer_meta["linked_question_id"] = question_id

        self.collection.add(
            ids=[f"{question_id}::answer"],
            documents=[answer_text],
            embeddings=[self._embed(answer_text)],
            metadatas=[answer_meta],
        )

    # --------------------------------------------------
    # QUERY
    # --------------------------------------------------

    def find_similar_questions(
        self,
        question_text: str,
        top_k: int = 5,
        min_similarity: float = 0.80,
    ) -> List[Dict[str, Any]]:
        query_vec = self._embed(question_text)

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            # Filter to only match against questions, not answers
            where={"type": "question"},
        )

        filtered = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                # If using cosine distance: Similarity = 1 - Distance
                similarity = 1.0 - dist

                if similarity >= min_similarity:
                    filtered.append(
                        {
                            "id": results["ids"][0][i],
                            "question": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity": round(similarity, 3),
                        }
                    )

        return filtered

    def persist(self):
        # In modern Chroma (PersistentClient), data is auto-persisted.
        # This method is kept for API compatibility but does nothing.
        pass
