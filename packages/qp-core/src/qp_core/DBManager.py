import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# IDGenerator is used to produce UUIDs for new users and user-file assignments
from qp_core.IDGenerator import IDGenerator


class DBManager:
    def __init__(self, db_path: str = "data/rag_staging.db", log_level=logging.INFO):
        self.db_path = db_path
        self._id_gen = IDGenerator()
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
    # SCHEMA
    # =========================================================
    def _init_db(self):
        with self._connection() as con:
            cur = con.cursor()

            # ---- Core document tables ----
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_rejections (
                    rejection_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    level TEXT,
                    question_text TEXT,
                    reason TEXT NOT NULL,
                    semantic_score REAL,
                    rejected_at TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                )
            """)

            # ---- User tables ----
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id   TEXT PRIMARY KEY,
                    username  TEXT NOT NULL UNIQUE,
                    email     TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
            """)

            # Join table — which user owns / has access to which file.
            # A single file can be shared with multiple users, and a user
            # can own multiple files, so this is a many-to-many relationship.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_files (
                    user_file_id TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    file_id      TEXT NOT NULL,
                    assigned_at  TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id),
                    FOREIGN KEY(file_id) REFERENCES files(file_id),
                    UNIQUE(user_id, file_id)   -- prevent duplicate assignments
                )
            """)
            # ---- Session Tracking ----
            cur.execute("""
                CREATE TABLE IF NOT EXISTS session_results (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    questions_attempted INTEGER,
                    average_score REAL,
                    final_difficulty TEXT,
                    history_json TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
            """)

            # Add an index to quickly look up a user's past sessions
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_results_user ON session_results(user_id);"
            )

            # ---- Indexes ----
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_file      ON chunks(file_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_use       ON chunks(should_use);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_enrich_chunk     ON chunk_enrichments(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_questions_chunk  ON chunk_questions(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rejections_chunk ON chunk_rejections(chunk_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rejections_reason ON chunk_rejections(reason);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_files_user  ON user_files(user_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_files_file  ON user_files(file_id);"
            )

            # ---- Safe migrations ----
            for migration in [
                "ALTER TABLE chunks ADD COLUMN parent_section TEXT",
                "ALTER TABLE chunks ADD COLUMN top_header TEXT",
                "ALTER TABLE chunk_questions ADD COLUMN source_quote TEXT",
                # NEW: Store rejected answers for debugging/analysis
                "ALTER TABLE chunk_rejections ADD COLUMN answer_text TEXT",
            ]:
                try:
                    cur.execute(migration)
                    self.logger.info(f"Migration applied: {migration}")
                except sqlite3.OperationalError:
                    pass  # column already exists — safe to ignore

    # =========================================================
    # USER MANAGEMENT
    # =========================================================
    def create_user(self, username: str, email: str) -> Optional[str]:
        """
        Create a new user and return their generated user_id (UUID).

        Uses IDGenerator to produce a UUID consistent with how file and chunk
        IDs are generated elsewhere in the pipeline.

        Args:
            username: unique display name for the user
            email:    unique email address for the user

        Returns:
            user_id string on success, or None if creation failed (e.g. duplicate).
        """
        user_id = self._id_gen.generate_file_id()  # reuses UUID4 logic from IDGenerator
        now = pd.Timestamp.now().isoformat()
        try:
            with self._connection() as con:
                con.execute(
                    """
                    INSERT INTO users (user_id, username, email, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, username, email, now),
                )
            self.logger.info(f"Created user '{username}' → {user_id[:8]}")
            return user_id
        except sqlite3.IntegrityError:
            self.logger.error(
                f"Failed to create user '{username}' — username or email already exists"
            )
            return None
        except Exception as e:
            self.logger.error(f"create_user failed: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single user record by their UUID.

        Args:
            user_id: the UUID of the user to retrieve

        Returns:
            Dict with user fields, or None if not found.
        """
        try:
            with self._connection() as con:
                row = con.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"get_user_by_id failed: {e}")
            return None

    # =========================================================
    # SESSION RESULTS MANAGEMENT
    # =========================================================
    def save_session_result(
        self,
        session_id: str,
        user_id: Optional[str],
        start_time: str,
        end_time: str,
        questions_attempted: int,
        average_score: float,
        final_difficulty: str,
        history_json: str,
    ) -> bool:
        """
        Save the complete results of an interview session.
        """
        sql = """
            INSERT OR REPLACE INTO session_results
            (session_id, user_id, start_time, end_time, questions_attempted,
                average_score, final_difficulty, history_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._connection() as con:
                con.execute(
                    sql,
                    (
                        session_id,
                        user_id,
                        start_time,
                        end_time,
                        questions_attempted,
                        average_score,
                        final_difficulty,
                        history_json,
                    ),
                )
            self.logger.info(f"Saved session result for session: {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save session result {session_id}: {e}")
            return False

    def get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific session result by its ID.
        """
        sql = "SELECT * FROM session_results WHERE session_id = ?"
        try:
            with self._connection() as con:
                row = con.execute(sql, (session_id,)).fetchone()
            if row:
                result = dict(row)
                # Parse the JSON string back into a Python list/dict
                result["history"] = (
                    json.loads(result["history_json"]) if result["history_json"] else []
                )
                del result["history_json"]
                return result
            return None
        except Exception as e:
            self.logger.error(f"Failed to get session result {session_id}: {e}")
            return None

    def get_session_results_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all past interview sessions for a specific user, ordered by most recent.
        """
        sql = "SELECT * FROM session_results WHERE user_id = ? ORDER BY end_time DESC"
        try:
            with self._connection() as con:
                rows = con.execute(sql, (user_id,)).fetchall()

            results = []
            for r in rows:
                result = dict(r)
                result["history"] = (
                    json.loads(result["history_json"]) if result["history_json"] else []
                )
                del result["history_json"]
                results.append(result)
            return results
        except Exception as e:
            self.logger.error(
                f"Failed to fetch session results for user {user_id}: {e}"
            )
            return []

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a user record by username.

        Args:
            username: the display name to look up

        Returns:
            Dict with user fields, or None if not found.
        """
        try:
            with self._connection() as con:
                row = con.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"get_user_by_username failed: {e}")
            return None

    def list_users(self) -> List[Dict[str, Any]]:
        """
        Return all users, ordered by creation date descending.

        Returns:
            List of user dicts.
        """
        try:
            with self._connection() as con:
                rows = con.execute(
                    "SELECT * FROM users ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            self.logger.error(f"list_users failed: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user and clean up their associated records.
        Manual deletion is required for child tables because ON DELETE CASCADE
        was not specified in the original schema creation.
        """
        try:
            with self._connection() as con:
                # 1. Remove user-file access assignments
                con.execute("DELETE FROM user_files WHERE user_id = ?", (user_id,))

                # 2. Remove interview session history
                con.execute("DELETE FROM session_results WHERE user_id = ?", (user_id,))

                # 3. Delete the user record
                cur = con.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

                deleted = cur.rowcount > 0
                if deleted:
                    self.logger.info(f"Successfully deleted user {user_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def update_user_email(self, user_id: str, new_email: str) -> bool:
        """Update a user's email address."""
        try:
            with self._connection() as con:
                cur = con.execute(
                    "UPDATE users SET email = ? WHERE user_id = ?", (new_email, user_id)
                )
                return cur.rowcount > 0
        except sqlite3.IntegrityError:
            self.logger.error(f"Email {new_email} is already in use.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to update user {user_id}: {e}")
            return False

    def get_questions_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Return all accepted QA pairs for every file assigned to a user.
        Joins through user_files so only the user's own documents are included.

        Args:
            user_id: UUID of the user

        Returns:
            List of QA dicts ordered by file assignment date and chunk index.
        """
        sql = """
                SELECT
                    q.question_id,
                    q.chunk_id,
                    q.question_text,
                    q.answer_text,
                    q.source_quote,
                    q.difficulty,
                    q.question_type,
                    e.tags
                FROM chunk_questions q
                JOIN chunks c       ON q.chunk_id  = c.chunk_id
                JOIN user_files uf  ON c.file_id   = uf.file_id
                LEFT JOIN chunk_enrichments e ON q.chunk_id = e.chunk_id
                WHERE uf.user_id = ?
                ORDER BY uf.assigned_at DESC, c.chunk_index
            """
        try:
            with self._connection() as con:
                rows = con.execute(sql, (user_id,)).fetchall()
            result = []
            for r in rows:
                row = dict(r)
                try:
                    row["tags"] = json.loads(row["tags"] or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["tags"] = []
                result.append(row)
            return result
        except Exception as e:
            self.logger.error(f"get_questions_for_user failed: {e}")
            return []

    # =========================================================
    # USER-FILE ASSIGNMENT
    # =========================================================
    def assign_file_to_user(self, user_id: str, file_id: str) -> bool:
        """
        Create a user → file ownership/access record.

        The underlying UNIQUE(user_id, file_id) constraint means calling this
        twice with the same pair is safe — the second call is silently ignored
        (INSERT OR IGNORE).

        Args:
            user_id: UUID of the user
            file_id: UUID of the file to assign

        Returns:
            True on success (including already-assigned), False on error.
        """
        user_file_id = self._id_gen.generate_file_id()
        now = pd.Timestamp.now().isoformat()
        try:
            with self._connection() as con:
                con.execute(
                    """
                    INSERT OR IGNORE INTO user_files
                        (user_file_id, user_id, file_id, assigned_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_file_id, user_id, file_id, now),
                )
            self.logger.info(f"Assigned file {file_id[:8]} → user {user_id[:8]}")
            return True
        except Exception as e:
            self.logger.error(f"assign_file_to_user failed: {e}")
            return False

    # =========================================================
    # ADMINISTRATION & MAINTENANCE
    # =========================================================

    def get_all_files(self) -> List[Dict[str, Any]]:
        """Retrieve all files in the database for administrative views."""
        try:
            with self._connection() as con:
                rows = con.execute(
                    "SELECT * FROM files ORDER BY processed_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            self.logger.error(f"get_all_files failed: {e}")
            return []

    def delete_file(self, file_id: str) -> bool:
        """
        Safely delete a file and all its cascaded data (chunks, QA pairs,
        enrichments, rejections, and user assignments).
        """
        try:
            with self._connection() as con:
                # 1. Remove user-file assignments
                con.execute("DELETE FROM user_files WHERE file_id = ?", (file_id,))

                # 2. Find all chunk IDs associated with this file to delete their metadata
                chunk_rows = con.execute(
                    "SELECT chunk_id FROM chunks WHERE file_id = ?", (file_id,)
                ).fetchall()
                chunk_ids = [row["chunk_id"] for row in chunk_rows]

                if chunk_ids:
                    # SQLite allows a max of 999 variables in an IN clause by default.
                    # We chunk the IDs safely in case of massive files.
                    for i in range(0, len(chunk_ids), 900):
                        batch = chunk_ids[i : i + 900]
                        placeholders = ",".join("?" * len(batch))

                        con.execute(
                            f"DELETE FROM chunk_enrichments WHERE chunk_id IN ({placeholders})",
                            batch,
                        )
                        con.execute(
                            f"DELETE FROM chunk_questions WHERE chunk_id IN ({placeholders})",
                            batch,
                        )
                        con.execute(
                            f"DELETE FROM chunk_rejections WHERE chunk_id IN ({placeholders})",
                            batch,
                        )

                # 3. Delete the chunks themselves
                con.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

                # 4. Finally, delete the file record
                cur = con.execute("DELETE FROM files WHERE file_id = ?", (file_id,))

                deleted = cur.rowcount > 0
                if deleted:
                    self.logger.info(
                        f"Successfully deleted file {file_id} and all related data."
                    )
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete file {file_id}: {e}")
            return False

    def delete_all_files(self) -> bool:
        """
        WARNING: Administrative function to wipe all files and related vector data.
        Clears files, chunks, questions, enrichments, rejections, and assignments.
        """
        try:
            with self._connection() as con:
                con.execute("DELETE FROM user_files")
                con.execute("DELETE FROM chunk_enrichments")
                con.execute("DELETE FROM chunk_questions")
                con.execute("DELETE FROM chunk_rejections")
                con.execute("DELETE FROM chunks")
                con.execute("DELETE FROM files")
            self.logger.warning(
                "All files and associated chunk data have been wiped from the database."
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete all files: {e}")
            return False

    def remove_file_from_user(self, user_id: str, file_id: str) -> bool:
        """
        Remove a user → file assignment.

        Args:
            user_id: UUID of the user
            file_id: UUID of the file to unassign

        Returns:
            True if a row was deleted, False otherwise.
        """
        try:
            with self._connection() as con:
                cur = con.execute(
                    "DELETE FROM user_files WHERE user_id = ? AND file_id = ?",
                    (user_id, file_id),
                )
            removed = cur.rowcount > 0
            if removed:
                self.logger.info(f"Removed file {file_id[:8]} from user {user_id[:8]}")
            return removed
        except Exception as e:
            self.logger.error(f"remove_file_from_user failed: {e}")
            return False

    def get_files_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Return all files assigned to a user, joined with file metadata.

        Args:
            user_id: UUID of the user

        Returns:
            List of dicts with full file metadata + assigned_at timestamp.
        """
        sql = """
            SELECT f.*, uf.assigned_at
            FROM user_files uf
            JOIN files f ON uf.file_id = f.file_id
            WHERE uf.user_id = ?
            ORDER BY uf.assigned_at DESC
        """
        try:
            with self._connection() as con:
                rows = con.execute(sql, (user_id,)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            self.logger.error(f"get_files_for_user failed: {e}")
            return []

    def get_users_for_file(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Return all users who have been assigned a given file.

        Useful for access-control checks or audit purposes.

        Args:
            file_id: UUID of the file

        Returns:
            List of user dicts with assigned_at timestamp.
        """
        sql = """
            SELECT u.*, uf.assigned_at
            FROM user_files uf
            JOIN users u ON uf.user_id = u.user_id
            WHERE uf.file_id = ?
            ORDER BY uf.assigned_at DESC
        """
        try:
            with self._connection() as con:
                rows = con.execute(sql, (file_id,)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            self.logger.error(f"get_users_for_file failed: {e}")
            return []

    # =========================================================
    # REJECTIONS & DUPLICATES
    # =========================================================
    def save_rejections(self, chunk_id: str, rejections: List[Dict[str, Any]]) -> None:
        """
        Save rejected QA candidates (including the generated answer when available).
        Early rejections (quote guard, invalid Pass 1, etc.) will have empty answer_text.
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
                    rej.get("answer", rej.get("answer_text", ""))[
                        :1000
                    ],  # NEW: store answer
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
                    (rejection_id, chunk_id, level, question_text, answer_text,
                     reason, semantic_score, rejected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            self.logger.info(
                f"Saved {len(rejections)} rejections for chunk {chunk_id[:8]} "
                f"(including rejected answers)"
            )
        except Exception as e:
            self.logger.error(f"Failed to save rejections for {chunk_id}: {e}")

    # =========================================================
    # EXISTING METHODS (unchanged)
    # =========================================================
    def save_chunks(self, file_id: str, chunks: List[Dict[str, Any]]) -> bool:
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
                sql = """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, file_id, chunk_index, content, content_type,
                     estimated_tokens, section_header, parent_section, top_header,
                     prev_chunk_id, next_chunk_id, quality_score, should_use)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                con.executemany(sql, rows)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save chunks: {e}")
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

    def get_pending_files(self, limit: int = 10) -> List[str]:
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
        sql = """
            SELECT c.*,
                e.summary AS existing_summary,
                e.tags AS existing_tags
            FROM chunks c
            LEFT JOIN chunk_enrichments e ON c.chunk_id = e.chunk_id
            LEFT JOIN chunk_questions q ON c.chunk_id = q.chunk_id
            WHERE c.file_id = ?
            AND c.should_use = 1
            AND q.question_id IS NULL          -- ← This is the key addition
            ORDER BY c.chunk_index
        """
        with self._connection() as con:
            return [dict(r) for r in con.execute(sql, (file_id,)).fetchall()]

    def save_enrichment(self, chunk_id: str, data: Dict[str, Any]) -> None:
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

    def get_questions_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Return all accepted QA pairs for a given chunk, joined with the chunk's
        normalised tags from chunk_enrichments.

        Used by Embedder.py to build the documents it pushes into the vector store.

        Args:
            chunk_id: UUID of the chunk

        Returns:
            List of dicts with keys: question_id, chunk_id, question_text,
            answer_text, source_quote, difficulty, question_type, tags (list).
        """
        sql = """
            SELECT
                q.question_id,
                q.chunk_id,
                q.question_text,
                q.answer_text,
                q.source_quote,
                q.difficulty,
                q.question_type,
                e.tags
            FROM chunk_questions q
            LEFT JOIN chunk_enrichments e ON q.chunk_id = e.chunk_id
            WHERE q.chunk_id = ?
        """
        try:
            with self._connection() as con:
                rows = con.execute(sql, (chunk_id,)).fetchall()
            result = []
            for r in rows:
                row = dict(r)
                try:
                    row["tags"] = json.loads(row["tags"] or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["tags"] = []
                result.append(row)
            return result
        except Exception as e:
            self.logger.error(f"get_questions_for_chunk failed: {e}")
            return []

    def get_questions_for_file(self, file_id: str) -> List[Dict[str, Any]]:
        """
        Return all accepted QA pairs for every chunk belonging to a file,
        joined with tags from chunk_enrichments.

        Used by Embedder.py to index an entire file in one pass.

        Args:
            file_id: UUID of the file

        Returns:
            Same schema as get_questions_for_chunk, ordered by chunk_index.
        """
        sql = """
            SELECT
                q.question_id,
                q.chunk_id,
                q.question_text,
                q.answer_text,
                q.source_quote,
                q.difficulty,
                q.question_type,
                e.tags
            FROM chunk_questions q
            JOIN chunks c ON q.chunk_id = c.chunk_id
            LEFT JOIN chunk_enrichments e ON q.chunk_id = e.chunk_id
            WHERE c.file_id = ?
            ORDER BY c.chunk_index
        """
        try:
            with self._connection() as con:
                rows = con.execute(sql, (file_id,)).fetchall()
            result = []
            for r in rows:
                row = dict(r)
                try:
                    row["tags"] = json.loads(row["tags"] or "[]")
                except (TypeError, json.JSONDecodeError):
                    row["tags"] = []
                result.append(row)
            return result
        except Exception as e:
            self.logger.error(f"get_questions_for_file failed: {e}")
            return []

    def get_all_enriched_file_ids(self) -> List[str]:
        """
        Return file IDs that have at least one accepted QA pair — i.e. files
        that have completed enrichment and are ready to be indexed.

        Used by Embedder.run() to find all indexable files.

        Returns:
            List of file_id strings.
        """
        sql = """
            SELECT DISTINCT c.file_id
            FROM chunk_questions q
            JOIN chunks c ON q.chunk_id = c.chunk_id
        """
        try:
            with self._connection() as con:
                rows = con.execute(sql).fetchall()
            return [r["file_id"] for r in rows]
        except Exception as e:
            self.logger.error(f"get_all_enriched_file_ids failed: {e}")
            return []

    def save_questions(self, chunk_id: str, qa_pairs: List[Dict[str, Any]]) -> None:
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
                (question_id, chunk_id, question_text, answer_text,
                 source_quote, difficulty, question_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
