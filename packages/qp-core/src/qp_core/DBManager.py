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
    # SCHEMA
    # =========================================================

    def _init_db(self):
        with self._connection() as con:
            cur = con.cursor()

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

            # Updated chunks table with hierarchy columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    content_type TEXT,
                    estimated_tokens INTEGER,

                    -- Content Context
                    section_header TEXT,
                    parent_section TEXT,
                    top_header TEXT,

                    -- Linkage
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,

                    -- Quality & Usage
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
                    difficulty TEXT CHECK(difficulty IN ('Easy','Medium','Hard')),
                    question_type TEXT CHECK(question_type IN ('Fact','Mechanism','Critical')),
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

            # --- MIGRATION CHECK ---
            # Automatically add new columns if they don't exist in an old DB
            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN parent_section TEXT")
                self.logger.info("Migrated DB: Added parent_section column")
            except sqlite3.OperationalError:
                pass  # Column exists

            try:
                cur.execute("ALTER TABLE chunks ADD COLUMN top_header TEXT")
                self.logger.info("Migrated DB: Added top_header column")
            except sqlite3.OperationalError:
                pass  # Column exists

    # =========================================================
    # FILES & CHUNKS
    # =========================================================
    def save_chunks(self, file_id: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Saves a list of chunks to the database.
        """
        if not chunks:
            self.logger.warning(f"No chunks provided for file {file_id}")
            return False

        rows = []
        for chunk in chunks:
            # Handle cases where 'evaluation' might be missing or nested
            eval_data = chunk.get("evaluation", {})
            should_use = 1 if eval_data.get("should_use", True) else 0
            quality_score = eval_data.get("quality_score", 0.0)

            # Metadata access safely
            meta = chunk.get("metadata", {})

            rows.append(
                (
                    chunk["chunk_id"],
                    file_id,
                    chunk["chunk_index"],
                    chunk["content"],
                    meta.get("content_type", "text"),
                    chunk.get("estimated_tokens", 0),
                    # Context fields
                    meta.get("section_header", ""),
                    meta.get("parent_section", ""),
                    meta.get("top_header", ""),
                    # Linkage
                    chunk.get("prev_chunk_id"),
                    chunk.get("next_chunk_id"),
                    # Quality
                    quality_score,
                    should_use,
                )
            )

        try:
            with self._connection() as con:
                sql = """
                        INSERT OR REPLACE INTO chunks (
                            chunk_id, file_id, chunk_index, content,
                            content_type, estimated_tokens,
                            section_header, parent_section, top_header,
                            prev_chunk_id, next_chunk_id,
                            quality_score, should_use
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                con.executemany(sql, rows)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save chunks for {file_id}: {e}")
            return False

    def save_file_metadata(
        self, file_id, file_path, simhash, metadata, content_length
    ) -> bool:
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

    # =========================================================
    # ENRICHMENT
    # =========================================================

    def get_pending_files(self, limit: int = 10) -> List[str]:
        sql = """
            SELECT DISTINCT c.file_id
            FROM chunks c
            LEFT JOIN chunk_enrichments e ON c.chunk_id = e.chunk_id
            WHERE c.should_use = 1 AND e.chunk_id IS NULL
            LIMIT ?
        """
        with self._connection() as con:
            rows = con.execute(sql, (limit,)).fetchall()
        return [r["file_id"] for r in rows]

    def get_chunks_for_file_ordered(self, file_id: str) -> List[Dict[str, Any]]:
        sql = """
            SELECT
                c.chunk_id,
                c.content,
                c.section_header,
                c.parent_section,
                c.top_header,
                c.chunk_index,
                e.summary AS existing_summary,
                e.tags AS existing_tags
            FROM chunks c
            LEFT JOIN chunk_enrichments e ON c.chunk_id = e.chunk_id
            WHERE c.file_id = ? AND c.should_use = 1
            ORDER BY c.chunk_index
        """
        with self._connection() as con:
            return [dict(r) for r in con.execute(sql, (file_id,)).fetchall()]

    def save_enrichment(self, chunk_id: str, data: Dict[str, Any]) -> None:
        try:
            tags = json.dumps(data.get("tags", []))
            triplets = json.dumps(data.get("triplets", []))
        except Exception:
            self.logger.error(f"Invalid JSON in enrichment for {chunk_id}")
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

    # =========================================================
    # QUESTIONS
    # =========================================================

    def save_questions(self, chunk_id: str, qa_pairs: List[Dict[str, Any]]) -> None:
        if not qa_pairs:
            return

        rows = []
        for idx, qa in enumerate(qa_pairs):
            # Deterministic ID: stable across re-runs
            q_id = f"{chunk_id}_q{idx + 1}"

            rows.append(
                (
                    q_id,
                    chunk_id,
                    qa.get("question", "").strip(),
                    qa.get("answer", "").strip(),
                    qa.get("difficulty", "Medium"),
                    qa.get("type", "Fact"),
                )
            )

        with self._connection() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO chunk_questions
                (question_id, chunk_id, question_text, answer_text, difficulty, question_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                rows,
            )
