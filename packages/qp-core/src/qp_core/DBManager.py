import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class DBManager:
    def __init__(self, db_path: str = "data/rag_staging.db", log_level=logging.INFO):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)
        self._init_db()

    # =========================================================
    # CONNECTION MANAGEMENT
    # =========================================================
    def get_all_simhashes(self) -> Dict[str, str]:
        """Retrieve all file simhashes for duplicate detection."""
        sql = "SELECT simhash, file_id FROM files WHERE simhash IS NOT NULL"
        try:
            with self._connection() as con:
                rows = con.execute(sql).fetchall()
                return {row["simhash"]: row["file_id"] for row in rows}
        except Exception as e:
            self.logger.error(f"Failed to fetch simhashes: {e}")
            return {}

    @contextmanager
    def _connection(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON;")
        con.execute("PRAGMA journal_mode = WAL;")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # =========================================================
    # SCHEMA (with new rejection table)
    # =========================================================
    def _init_db(self):
        with self._connection() as con:
            cur = con.cursor()

            # Existing tables (unchanged)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    file_name TEXT,
                    simhash TEXT,
                    format TEXT,
                    title TEXT,
                    author TEXT,
                    creation_date TEXT,
                    processed_at TEXT,
                    content_length INTEGER
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    content_type TEXT,
                    estimated_tokens INTEGER,
                    section_header TEXT,
                    parent_section TEXT,
                    top_header TEXT,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    quality_score REAL,
                    should_use INTEGER DEFAULT 1,
                    FOREIGN KEY(file_id) REFERENCES files(file_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_enrichments (
                    chunk_id TEXT PRIMARY KEY,
                    tags TEXT,
                    triplets TEXT,
                    summary TEXT,
                    processed_at TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_questions (
                    question_id TEXT PRIMARY KEY,
                    chunk_id TEXT,
                    question_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    source_quote TEXT,
                    difficulty TEXT CHECK(difficulty IN ('Easy','Medium','Hard')),
                    question_type TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                )
            """)

            # ==================== NEW: REJECTION TABLE ====================
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_rejections (
                    rejection_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    level TEXT,                    -- Easy / Medium / Hard
                    question_text TEXT,
                    reason TEXT NOT NULL,          -- "Semantic fail", "Duplicate in Chroma", etc.
                    semantic_score REAL,
                    rejected_at TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                )
            """)

            # Indexes for performance
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_use ON chunks(should_use);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_enrich_chunk ON chunk_enrichments(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_questions_chunk ON chunk_questions(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rejections_chunk ON chunk_rejections(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rejections_reason ON chunk_rejections(reason);"
            )

            # Migrations (safe)
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN parent_section TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN top_header TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cur.execute("ALTER TABLE chunk_questions ADD COLUMN source_quote TEXT")
                self.logger.info("Migrated DB: Added source_quote column")
            except sqlite3.OperationalError:
                pass

    # =========================================================
    # REJECTIONS & DUPLICATES (NEW)
    # =========================================================
    def save_rejections(self, chunk_id: str, rejections: List[Dict[str, Any]]) -> None:
        """
        Save rejected and duplicated questions.
        Works for both validator rejections and Chroma duplicates.
        """
        if not rejections:
            return

        rows = []
        now = pd.Timestamp.now().isoformat()

        for idx, rej in enumerate(rejections):
            rejection_id = f"{chunk_id}_rej{idx + 1}"
            rows.append(
                (
                    rejection_id,
                    chunk_id,
                    rej.get("level"),
                    rej.get("question", "")[:500],  # truncate for safety
                    rej.get("reason", "Unknown"),
                    rej.get("semantic_score"),
                    now,
                )
            )

        try:
            with self._connection() as con:
                con.executemany(
                    """
                    INSERT OR REPLACE INTO chunk_rejections
                    (rejection_id, chunk_id, level, question_text, reason, semantic_score, rejected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            self.logger.info(
                f"Saved {len(rejections)} rejections/duplicates for chunk {chunk_id[:8]}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save rejections for {chunk_id}: {e}")

    # =========================================================
    # EXISTING METHODS (unchanged)
    # =========================================================
    def save_chunks(self, file_id: str, chunks: List[Dict[str, Any]]) -> bool:
        # ... (your existing code unchanged) ...
        if not chunks:
            return False
        rows = []
        for chunk in chunks:
            eval_data = chunk.get("evaluation", {})
            meta = chunk.get("metadata", {})
            rows.append(
                (
                    chunk["chunk_id"],
                    file_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    meta.get("content_type", "text"),
                    chunk.get("estimated_tokens", 0),
                    meta.get("section_header", ""),
                    meta.get("parent_section", ""),
                    meta.get("top_header", ""),
                    chunk.get("prev_chunk_id"),
                    chunk.get("next_chunk_id"),
                    eval_data.get("quality_score", 0.0),
                    1 if eval_data.get("should_use", True) else 0,
                )
            )
        try:
            with self._connection() as con:
                sql = """INSERT OR REPLACE INTO chunks (chunk_id, file_id, chunk_index, content,
                         content_type, estimated_tokens, section_header, parent_section, top_header,
                         prev_chunk_id, next_chunk_id, quality_score, should_use)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                con.executemany(sql, rows)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save chunks: {e}")
            return False

    def save_file_metadata(
        self, file_id, file_path, simhash, metadata, content_length
    ) -> bool:
        # ... (your existing code unchanged) ...
        data = {
            "file_id": file_id,
            "file_path": str(file_path),
            "file_name": Path(file_path).name,
            "simhash": simhash,
            "format": metadata.get("format", ""),
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "creation_date": metadata.get("creationDate", ""),
            "processed_at": pd.Timestamp.now().isoformat(),
            "content_length": content_length,
        }
        try:
            with self._connection() as con:
                cols = ", ".join(data.keys())
                vals = ", ".join("?" * len(data))
                con.execute(
                    f"INSERT OR REPLACE INTO files ({cols}) VALUES ({vals})",
                    list(data.values()),
                )
            return True
        except Exception as e:
            self.logger.error(f"Metadata save failed: {e}")
            return False

    def get_pending_files(self, limit: int = 10) -> List[str]:
        # ... (your existing code unchanged) ...
        sql = """
            SELECT DISTINCT c.file_id
            FROM chunks c
            LEFT JOIN chunk_questions q ON c.chunk_id = q.chunk_id
            WHERE c.should_use = 1 AND q.question_id IS NULL
            LIMIT ?
        """
        with self._connection() as con:
            rows = con.execute(sql, (limit,)).fetchall()
        return [r["file_id"] for r in rows]

    def get_chunks_for_file_ordered(self, file_id: str) -> List[Dict[str, Any]]:
        # ... (your existing code unchanged) ...
        sql = """
            SELECT c.*, e.summary AS existing_summary, e.tags AS existing_tags
            FROM chunks c
            LEFT JOIN chunk_enrichments e ON c.chunk_id = e.chunk_id
            WHERE c.file_id = ? AND c.should_use = 1
            ORDER BY c.chunk_index
        """
        with self._connection() as con:
            return [dict(r) for r in con.execute(sql, (file_id,)).fetchall()]

    def save_enrichment(self, chunk_id: str, data: Dict[str, Any]) -> None:
        # ... (your existing code unchanged) ...
        try:
            tags = json.dumps(data.get("tags", []))
            triplets = json.dumps(data.get("triplets", []))
        except Exception:
            return
        with self._connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO chunk_enrichments
                (chunk_id, tags, triplets, summary, processed_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chunk_id,
                    tags,
                    triplets,
                    data.get("summary", ""),
                    pd.Timestamp.now().isoformat(),
                ),
            )

    def save_questions(self, chunk_id: str, qa_pairs: List[Dict[str, Any]]) -> None:
        # ... (your existing code unchanged) ...
        if not qa_pairs:
            return
        rows = []
        for idx, qa in enumerate(qa_pairs):
            q_id = f"{chunk_id}_q{idx + 1}"
            q_text = qa.get("question_text", qa.get("question", "")).strip()
            a_text = qa.get("answer_text", qa.get("answer", "")).strip()
            q_type = qa.get("question_type", qa.get("type", "Fact")).capitalize()
            source_quote = qa.get("source_quote", "").strip()

            rows.append(
                (
                    q_id,
                    chunk_id,
                    q_text,
                    a_text,
                    source_quote,
                    qa.get("difficulty", "Medium"),
                    q_type,
                )
            )
        with self._connection() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO chunk_questions
                (question_id, chunk_id, question_text, answer_text, source_quote, difficulty, question_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                rows,
            )
