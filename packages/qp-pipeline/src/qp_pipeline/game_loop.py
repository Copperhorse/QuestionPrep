"""
game_loop.py
State-Machine Driven Interview Orchestrator (Coaching-Oriented)
FastAPI Compatible

Changelog vs original:
- [FIX] Removed CrossEncoder NLI grader entirely.
    Root cause: nli-deberta-v3-xsmall label order is 0=contradiction, 1=entailment,
    2=neutral. Original code read probs[2] (neutral) instead of probs[1] (entailment),
    causing correct answers to score ~2% similarity.

- [NEW] Two-stage grading pipeline in analyze_response():
    Stage 1 — rapidfuzz partial_ratio (fast, zero model overhead):
        Normalises score to 0.0–1.0. If score >= LEXICAL_PASS_THRESHOLD (0.75),
        the answer is a strong lexical match; use this score directly and skip Stage 2.
        Catches verbatim, near-verbatim, and lightly paraphrased answers cheaply.
    Stage 2 — BGE-small cosine similarity (semantic fallback):
        Runs only when Stage 1 score is below threshold, meaning the answer is
        paraphrased or worded differently but may still be semantically correct.
        Uses the same BGE model already present in AnswerValidator (Enricher.py)
        for consistency.

- [CLEAN] Removed numpy import — was only needed for CrossEncoder softmax math.
- [CLEAN] Removed ML_AVAILABLE flag — BGE is a hard dependency via sentence-transformers,
    same as AnswerValidator. Kept a try/except for graceful degradation to lexical-only mode.
- [NEW] TurnResult.grader field records which stage produced the final score
    ("lexical" or "semantic") for transparency/debugging.
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

from rapidfuzz import fuzz

# ---------------- PATH SETUP ----------------
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qp_core.DBManager import DBManager

# ---------------- OPTIONAL ML ----------------
# BGE is used for Stage 2 semantic fallback. If sentence-transformers is not
# installed, grading degrades gracefully to lexical-only (Stage 1 only).
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util

    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False

# ---------------- CONFIG ----------------
DB_PATH = project_root / "data" / "rag_staging.db"

BGE_MODEL = "BAAI/bge-small-en-v1.5"  # Same model used in AnswerValidator

# Stage 1: if rapidfuzz partial_ratio (normalised 0–1) is >= this, accept directly
# and skip the BGE call entirely. 0.75 = strong lexical overlap.
LEXICAL_PASS_THRESHOLD = 0.75

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
    grader: str = "lexical"  # "lexical" | "semantic" — which stage scored this turn


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

    Grading pipeline (two-stage):
        1. rapidfuzz partial_ratio — O(n) string comparison, no model needed.
           If the normalised score clears LEXICAL_PASS_THRESHOLD, used directly.
        2. BGE cosine similarity — semantic embedding comparison.
           Runs only when Stage 1 is below threshold.

    The BGE model (~33MB) is held as a class-level singleton so it loads exactly
    once regardless of how many sessions are active, behind a class-level lock.
    """

    # ---- Shared BGE singleton ----
    _embedder = None
    _embedder_lock = threading.Lock()
    _embedder_loaded = False

    @classmethod
    def _get_embedder(cls):
        """Lazy-load BGE exactly once, thread-safely."""
        if not cls._embedder_loaded:
            with cls._embedder_lock:
                if not cls._embedder_loaded:  # double-checked locking
                    if BGE_AVAILABLE:
                        logger.info(f"Loading semantic grader model: {BGE_MODEL}")
                        cls._embedder = SentenceTransformer(BGE_MODEL)
                    cls._embedder_loaded = True
        return cls._embedder

    def __init__(self, db_path, user_id: str):
        """
        Args:
            db_path: path to the SQLite DB
            user_id: UUID of the user — questions are filtered to this user's files
        """
        self.db = DBManager(db_path)
        self.user_id = user_id
        self.questions: Dict[str, QuestionObj] = {}
        self._load_questions()
        # BGE loads lazily on first analyze_response call that needs Stage 2

    def _load_questions(self):
        """
        Load questions scoped to the user's assigned files.

        Uses DBManager.get_questions_for_user() which joins through user_files,
        so only documents this user uploaded are included.
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
        """
        Two-stage grading pipeline.

        Stage 1 — Lexical (rapidfuzz partial_ratio):
            Cheap, runs always. partial_ratio finds the best matching substring
            window, which handles answers that contain the key phrase among other
            words without penalising extra context.
            Score is normalised from 0–100 → 0.0–1.0.
            If score >= LEXICAL_PASS_THRESHOLD: accept, skip Stage 2.

        Stage 2 — Semantic (BGE cosine similarity):
            Runs only when Stage 1 is below threshold. Encodes both strings with
            BGE-small and computes cosine similarity. Catches paraphrased-but-correct
            answers that score low on lexical overlap.

        Why not CrossEncoder NLI (original approach):
            - Wrong tool: NLI entailment is asymmetric and does not map cleanly to
              answer quality. A paraphrase can have high cosine similarity but low
              entailment probability.
            - Was also bugged: code read probs[2] (neutral) instead of probs[1]
              (entailment) for nli-deberta-v3-xsmall, causing all scores to be ~2%.
        """
        grader_used = "lexical"

        # ---- Stage 1: rapidfuzz partial_ratio ----
        lexical_score = (
            fuzz.partial_ratio(user_text.lower(), q.answer.lower()) / 100.0
        )  # normalise to 0.0–1.0

        if lexical_score >= LEXICAL_PASS_THRESHOLD:
            # Strong lexical match — no need to invoke the embedding model
            similarity = lexical_score
            logger.debug(f"[Stage 1 pass] q={q.id[:8]} lexical={lexical_score:.3f}")
        else:
            # ---- Stage 2: BGE semantic similarity ----
            embedder = self._get_embedder()
            if embedder:
                with self._embedder_lock:
                    embeddings = embedder.encode(
                        [user_text, q.answer], convert_to_tensor=True
                    )
                semantic_score = float(st_util.cos_sim(embeddings[0], embeddings[1]))
                # Cosine similarity can be slightly negative for very dissimilar
                # texts; clamp to 0 so downstream logic stays in [0, 1].
                similarity = max(0.0, min(1.0, semantic_score))
                grader_used = "semantic"
                logger.debug(
                    f"[Stage 2] q={q.id[:8]} lexical={lexical_score:.3f} "
                    f"semantic={similarity:.3f}"
                )
            else:
                # BGE unavailable — fall back to lexical score only
                similarity = lexical_score
                grader_used = "lexical"
                logger.warning("BGE unavailable — using lexical score only")

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
            grader=grader_used,
        )


# ---------------- SESSION ----------------
class InterviewSession:
    """
    Manages state for a single user's interview session.

    Each session owns its own LogicEngine so question scoping is per-user.
    The BGE model inside LogicEngine is a shared singleton, so creating many
    sessions in parallel does not reload the model each time.
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
                "grader": result.grader,  # "lexical" | "semantic"
            },
            "next_question": (
                self.ctx.current_question.text if self.ctx.current_question else None
            ),
            "is_terminal": is_terminal,
        }
