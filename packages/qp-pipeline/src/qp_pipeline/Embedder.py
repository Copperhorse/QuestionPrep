"""
Embedder.py — QA Vector Store + Indexer
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

CHROMA_DIR = str(project_root / "data" / "chroma_store")
DB_PATH = str(project_root / "data" / "rag_staging.db")

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)


class QAVectorStore:
    """Thin wrapper around a ChromaDB collection for Q&A pairs."""

    COLLECTION_NAME = "qa_pairs"

    def __init__(self, persist_directory: str = CHROMA_DIR):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is not installed")
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"QAVectorStore ready — {self._collection.count()} document(s) in collection"
        )

    def add_qa_pair(
        self,
        question_id: str,
        question_text: str,
        answer_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        combined = f"Question: {question_text}\nAnswer: {answer_text}"
        self._collection.upsert(
            ids=[question_id],
            documents=[combined],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete_by_ids(self, ids: List[str]) -> None:
        """Remove specific document IDs from the collection."""
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} embedding(s) from Chroma")

    def count(self) -> int:
        return self._collection.count()


class VectorIndexer:
    """Generates BGE-small embeddings and writes them to QAVectorStore."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, persist_directory: str = CHROMA_DIR, db_path: str = DB_PATH):
        self._store = QAVectorStore(persist_directory)
        self._db_path = db_path
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _embed(self, text: str) -> List[float]:
        return self._get_model().encode(text, normalize_embeddings=True).tolist()

    # ── Indexing ────────────────────────────────────────────────────────────

    def index_file(self, file_id: str) -> int:
        """Embed all questions for *file_id* and upsert them into Chroma."""
        from qp_core.DBManager import DBManager

        db = DBManager(self._db_path)
        questions = db.get_questions_for_file(file_id)

        if not questions:
            logger.warning(
                f"No questions found for file {file_id[:8]} — nothing to index"
            )
            return 0

        indexed = 0
        for q in questions:
            try:
                text = f"Question: {q['question_text']}\nAnswer: {q['answer_text']}"
                embedding = self._embed(text)
                self._store.add_qa_pair(
                    question_id=q["question_id"],
                    question_text=q["question_text"],
                    answer_text=q["answer_text"],
                    embedding=embedding,
                    metadata={
                        "file_id": file_id,
                        "chunk_id": q.get("chunk_id", ""),
                        "difficulty": q.get("difficulty", "Medium"),
                        "type": q.get("question_type", "Fact"),
                    },
                )
                indexed += 1
            except Exception as e:
                logger.error(
                    f"Failed to index question {q.get('question_id', '?')}: {e}"
                )

        logger.info(
            f"Indexed {indexed}/{len(questions)} question(s) for file {file_id[:8]}"
        )
        return indexed

    # ── B12: Deletion ───────────────────────────────────────────────────────

    def delete_embeddings_for_file(self, file_id: str) -> int:
        """
        B12: Remove all Chroma embeddings belonging to *file_id*.

        Called by DELETE /api/files/{file_id} in main.py BEFORE DBManager.delete_file()
        so the two stores never diverge. Uses the 'file_id' metadata field that
        index_file() writes into every document's metadata at index time.

        Returns the number of embeddings deleted.
        """
        try:
            # Query Chroma for all question_ids associated with this file
            results = self._store._collection.get(
                where={"file_id": file_id},
                include=["metadatas"],
            )
            ids_to_delete: List[str] = results.get("ids", [])

            if not ids_to_delete:
                logger.info(
                    f"No Chroma embeddings found for file {file_id[:8]} — nothing to delete"
                )
                return 0

            self._store.delete_by_ids(ids_to_delete)
            logger.info(
                f"B12: Deleted {len(ids_to_delete)} Chroma embedding(s) for file {file_id[:8]}"
            )
            return len(ids_to_delete)

        except Exception as e:
            logger.error(
                f"B12: Failed to delete Chroma embeddings for {file_id[:8]}: {e}"
            )
            raise  # re-raise so the caller can log the warning and continue

    # ── Semantic search ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        file_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        embedding = self._embed(query)
        where = {"file_id": file_id} if file_id else None
        raw = self._store.query(embedding, n_results=n_results, where=where)

        results = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for qid, doc, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                {
                    "question_id": qid,
                    "document": doc,
                    "metadata": meta,
                    "score": round(1 - dist, 4),  # cosine distance → similarity
                }
            )
        return results
