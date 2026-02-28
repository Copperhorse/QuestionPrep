"""
game_loop.py
State-Machine Driven Interview Orchestrator (Coaching-Oriented)
FastAPI Compatible
"""

import json
import logging
import random
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# ---------------- PATH SETUP ----------------
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qp_core.DBManager import DBManager

# ---------------- OPTIONAL ML ----------------
try:
    from sentence_transformers import CrossEncoder

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ---------------- CONFIG ----------------
DB_PATH = project_root / "data" / "rag_staging.db"
GRADER_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
SIMILARITY_FLOOR = 0.40
CONFIDENCE_HINT = 0.65

logger = logging.getLogger("GameLoop")


# ---------------- STATE ----------------
class InterviewState(Enum):
    INIT = auto()
    IN_PROGRESS = auto()
    TERMINAL = auto()


# ---------------- DATA ----------------
@dataclass
class QuestionObj:
    id: str
    text: str
    answer: str
    type: str
    difficulty: str
    tags: List[str]
    confidence: float = 1.0


@dataclass
class TurnResult:
    question_id: str
    user_text: str
    similarity: float
    confidence: float
    feedback: str


@dataclass
class InterviewContext:
    current_question: Optional[QuestionObj] = None
    history: List[TurnResult] = field(default_factory=list)
    difficulty_score: float = 0.5
    difficulty_label: str = "Medium"


# ---------------- LOGIC ENGINE ----------------
class LogicEngine:
    """
    CPU-bound tasks (DB + ML), thread-safe.

    Scoped to a single user — only questions derived from that user's
    ingested files are loaded via get_questions_for_user().

    The CrossEncoder grader is expensive to load (~500MB). It is held as a
    class-level singleton so it is loaded exactly once regardless of how many
    sessions are active, and shared safely behind a class-level lock.
    """

    # ---- Shared grader singleton ----
    _grader = None
    _grader_lock = threading.Lock()
    _grader_loaded = False

    @classmethod
    def _get_grader(cls):
        """Lazy-load the CrossEncoder exactly once, thread-safely."""
        if not cls._grader_loaded:
            with cls._grader_lock:
                if not cls._grader_loaded:  # double-checked locking
                    if ML_AVAILABLE:
                        logger.info(f"Loading grader model: {GRADER_MODEL}")
                        cls._grader = CrossEncoder(GRADER_MODEL, activation_fn=None)
                    cls._grader_loaded = True
        return cls._grader

    def __init__(self, db_path, user_id: str):
        """
        Args:
            db_path: path to the SQLite DB
            user_id: UUID of the user — questions are filtered to this user's files
        """
        self.db = DBManager(db_path)
        self.user_id = user_id
        self.grader = self._get_grader()

        self.questions: Dict[str, QuestionObj] = {}
        self._load_questions()

    def _load_questions(self):
        """
        Load questions scoped to the user's assigned files.

        Uses DBManager.get_questions_for_user() which joins through user_files,
        so only documents this user uploaded are included. Tags are already
        deserialised to a list by DBManager.
        """
        rows = self.db.get_questions_for_user(self.user_id)

        if not rows:
            logger.warning(
                f"No questions found for user {self.user_id[:8]} — "
                "ensure files are ingested, enriched, indexed, and assigned."
            )
            return

        for r in rows:
            tags = r.get("tags", [])
            # Defensive: DBManager returns a list, but guard against raw JSON string
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (ValueError, TypeError):
                    tags = []

            self.questions[r["question_id"]] = QuestionObj(
                id=r["question_id"],
                text=r["question_text"],
                answer=r["answer_text"],
                type=r["question_type"],
                difficulty=r.get("difficulty") or "Medium",
                tags=tags,
            )

        logger.info(
            f"Loaded {len(self.questions)} question(s) for user {self.user_id[:8]}"
        )

    def select_next_question(
        self, difficulty: str, used_ids: Set[str]
    ) -> Optional[QuestionObj]:
        """
        Pick the next question, preferring the requested difficulty level.
        Falls back to any unused question if no difficulty match remains.
        """
        candidates = [
            q
            for q in self.questions.values()
            if q.difficulty.lower() == difficulty.lower() and q.id not in used_ids
        ]
        if not candidates:
            candidates = [q for q in self.questions.values() if q.id not in used_ids]

        return random.choice(candidates) if candidates else None

    def analyze_response(self, q: QuestionObj, user_text: str) -> TurnResult:
        similarity = 0.5

        if self.grader:
            with self._grader_lock:
                logits = self.grader.predict([(user_text, q.answer)])
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            similarity = float(probs[1])

        similarity = max(0.0, min(1.0, similarity))
        confidence = similarity * q.confidence

        if similarity < SIMILARITY_FLOOR:
            feedback = "Your answer does not align well with the source material."
        elif confidence >= CONFIDENCE_HINT:
            feedback = "Your explanation aligns well with the reference concepts."
        else:
            feedback = (
                "Your answer captures some relevant ideas. "
                "Consider emphasising the mechanisms described in the source."
            )

        return TurnResult(
            question_id=q.id,
            user_text=user_text,
            similarity=similarity,
            confidence=confidence,
            feedback=feedback,
        )


# ---------------- SESSION ----------------
class InterviewSession:
    """
    Manages state for a single user's interview session.

    Each session owns its own LogicEngine so question scoping is per-user.
    The grader model inside LogicEngine is a shared singleton, so creating
    many sessions in parallel does not reload the model each time.
    """

    def __init__(self, session_id: str, user_id: str, db_path=None):
        """
        Args:
            session_id: UUID for this session (generated by the API layer)
            user_id:    UUID of the user being interviewed
            db_path:    optional DB path override; defaults to module-level DB_PATH
        """
        self.session_id = session_id
        self.user_id = user_id
        self.logic = LogicEngine(db_path or DB_PATH, user_id)
        self.ctx = InterviewContext()
        self.used_ids: Set[str] = set()
        self.state = InterviewState.INIT

    def start_interview(self) -> Optional[str]:
        self.state = InterviewState.IN_PROGRESS
        self.ctx.current_question = self.logic.select_next_question(
            self.ctx.difficulty_label, self.used_ids
        )
        if self.ctx.current_question:
            self.used_ids.add(self.ctx.current_question.id)
            return self.ctx.current_question.text
        return None

    def evaluate_turn(self, user_text: str) -> Dict:
        q = self.ctx.current_question
        if not q:
            return {"error": "No active question."}

        result = self.logic.analyze_response(q, user_text)
        self.ctx.history.append(result)

        # EMA difficulty adaptation
        alpha = 0.3
        self.ctx.difficulty_score = (
            alpha * result.confidence + (1 - alpha) * self.ctx.difficulty_score
        )
        if self.ctx.difficulty_score > 0.75:
            self.ctx.difficulty_label = "Hard"
        elif self.ctx.difficulty_score < 0.35:
            self.ctx.difficulty_label = "Easy"
        else:
            self.ctx.difficulty_label = "Medium"

        # Select next question
        self.ctx.current_question = self.logic.select_next_question(
            self.ctx.difficulty_label, self.used_ids
        )
        is_terminal = self.ctx.current_question is None
        if not is_terminal:
            self.used_ids.add(self.ctx.current_question.id)
        else:
            self.state = InterviewState.TERMINAL

        return {
            "evaluation": {
                "similarity": result.similarity,
                "confidence": result.confidence,
                "feedback": result.feedback,
            },
            "next_question": (
                self.ctx.current_question.text if self.ctx.current_question else None
            ),
            "is_terminal": is_terminal,
        }
