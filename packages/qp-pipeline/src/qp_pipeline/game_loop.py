"""
game_loop.py
State-Machine Driven Interview Orchestrator.

Architecture: Deterministic State Machine (Moore-Style).
Priorities: Correctness > Latency.
Features:
- Thread-Safe ML Inference
- Numerically Stable Softmax
- EMA-Based Difficulty Adaptation
"""

import asyncio
import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# --- PATH SETUP (Dynamic) ---
current_file = Path(__file__).resolve()
# Assuming the file structure matches ingester.py:
# packages/qp-pipeline/src/qp_pipeline/game_loop.py -> parents[4] is Project Root
project_root = current_file.parents[4]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qp_core.DBManager import DBManager

# --- DEPENDENCIES ---
try:
    from sentence_transformers import CrossEncoder

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ WARNING: sentence-transformers not found.")

try:
    from qp_voice.speech_to_text import SpeechToText
    from qp_voice.text_to_speech import TextToSpeech

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️ WARNING: qp_voice not found. Using Mock I/O.")

# ---------------- CONFIG ----------------
# Dynamic DB Path Resolution
DB_PATH = project_root / "data" / "rag_staging.db"

GRADER_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
PASS_THRESHOLD = 0.65
MAX_WORKERS = 3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GameLoop")


# ---------------- STATE DEFINITIONS ----------------
class InterviewState(Enum):
    INIT = auto()  # Intro & Setup
    SELECT_Q = auto()  # Pick next question
    ASK = auto()  # TTS Output
    LISTEN = auto()  # STT Input
    GRADE = auto()  # ML Inference
    ADAPT = auto()  # Adjust Difficulty
    FEEDBACK = auto()  # Speak Feedback
    TERMINAL = auto()  # End session


@dataclass
class QuestionObj:
    id: str
    text: str
    answer: str
    type: str
    difficulty: str
    tags: List[str]


@dataclass
class TurnResult:
    question_id: str
    user_text: str
    score: float
    is_correct: bool
    feedback: str


@dataclass
class InterviewContext:
    """The Single Source of Truth for the Session."""

    current_question: Optional[QuestionObj] = None
    last_user_text: Optional[str] = None
    last_result: Optional[TurnResult] = None
    history: List[TurnResult] = field(default_factory=list)

    # EMA Difficulty State
    difficulty_score: float = 0.5  # 0.0 (Easy) to 1.0 (Hard)
    difficulty_label: str = "Medium"


# ---------------- LOGIC ENGINE (The Brain) ----------------
class LogicEngine:
    """Handles CPU-bound tasks (DB, ML) in a thread-safe way."""

    def __init__(self, db_path):
        self.db = DBManager(db_path)
        self.ml_lock = threading.Lock()

        if ML_AVAILABLE:
            logger.info(f"Loading Grader: {GRADER_MODEL}...")
            # We handle activation manually for stability
            self.grader = CrossEncoder(GRADER_MODEL, activation_fn=None)
        else:
            self.grader = None

        self.questions: Dict[str, QuestionObj] = {}
        self.used_ids = set()
        self._load_questions()

    def _load_questions(self):
        """Loads questions from DB. Injects Mock Data if DB is empty."""
        try:
            logger.info(f"Connecting to DB at: {self.db.db_path}")

            with self.db._connection() as con:
                rows = con.execute("""
                    SELECT q.question_id, q.question_text, q.answer_text,
                           q.difficulty, q.question_type
                    FROM chunk_questions q
                    JOIN chunks c ON q.chunk_id = c.chunk_id
                    WHERE c.should_use = 1
                """).fetchall()

            for r in rows:
                self.questions[r["question_id"]] = QuestionObj(
                    id=r["question_id"],
                    text=r["question_text"],
                    answer=r["answer_text"],
                    type=r["question_type"],
                    difficulty=r["difficulty"] or "Medium",
                    tags=[],
                )

            logger.info(f"DB Load Success: Found {len(self.questions)} questions.")

        except Exception as e:
            logger.error(f"DB Error: {e}")

        # --- FALLBACK: INJECT MOCK DATA IF EMPTY ---
        if not self.questions:
            logger.warning(
                "⚠️ DB returned 0 questions. Injecting MOCK DATA so you can test the loop."
            )
            mock_data = [
                (
                    "m1",
                    "What is the difference between TCP and UDP?",
                    "TCP is connection-oriented; UDP is connectionless.",
                    "Concept",
                    "Medium",
                ),
                (
                    "m2",
                    "Explain Python decorators.",
                    "Functions that modify other functions.",
                    "Code",
                    "Hard",
                ),
            ]
            for qid, txt, ans, typ, diff in mock_data:
                self.questions[qid] = QuestionObj(qid, txt, ans, typ, diff, ["mock"])

    def select_next_question(
        self, difficulty: str, prev_tags: List[str] = None
    ) -> QuestionObj:
        # 1. Filter by Difficulty & Unused
        candidates = [
            q
            for q in self.questions.values()
            if q.difficulty.lower() == difficulty.lower() and q.id not in self.used_ids
        ]

        # 2. Relax difficulty if no exact match
        if not candidates:
            candidates = [
                q for q in self.questions.values() if q.id not in self.used_ids
            ]

        # 3. Terminal State
        if not candidates:
            return QuestionObj(
                "0", "No questions left. End of interview.", "", "Fact", "Easy", []
            )

        # 4. Selection (Simple Random for now)
        best_candidate = random.choice(candidates)
        self.used_ids.add(best_candidate.id)
        return best_candidate

    def grade_answer(self, q_id: str, user_text: str) -> TurnResult:
        target_q = self.questions.get(q_id)
        if not target_q:
            return TurnResult(q_id, user_text, 0.0, False, "Error")

        # Heuristic: Skip detection
        if any(
            w in user_text.lower() for w in ["skip", "don't know", "pass", "unsure"]
        ):
            return TurnResult(
                q_id, user_text, 0.0, False, f"The answer was: {target_q.answer}"
            )

        score = 0.5
        if self.grader:
            try:
                with self.ml_lock:
                    raw_logits = self.grader.predict([(user_text, target_q.answer)])

                # Numerically Stable Softmax
                shifted_logits = raw_logits - np.max(raw_logits)
                exp_scores = np.exp(shifted_logits)
                probs = exp_scores / np.sum(exp_scores)

                contradiction = probs[0]
                entailment = probs[1]

                # Weighted Score
                score = entailment - (contradiction * 0.5)
                score = max(0.0, min(1.0, score))

            except Exception as e:
                logger.error(f"Grading Error: {e}")
                score = 0.5

        is_correct = score >= PASS_THRESHOLD
        feedback = "Correct." if is_correct else f"Actually: {target_q.answer}"
        return TurnResult(q_id, user_text, float(score), is_correct, feedback)


# ---------------- STATE MACHINE CORE ----------------
class InterviewStateMachine:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.logic = LogicEngine(DB_PATH)

        if VOICE_AVAILABLE:
            self.stt = SpeechToText()
            self.tts = TextToSpeech()
        else:
            self.stt = None
            self.tts = None

        self.state = InterviewState.INIT
        self.ctx = InterviewContext()
        self.is_running = True

    # --- I/O Helpers ---
    async def _speak(self, text: str):
        logger.info(f"🤖 BOT: {text}")
        if self.tts:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self.tts.generate_wav_bytes, text)
        else:
            # Console fallback
            print(f"\n[BOT]: {text}\n")
            await asyncio.sleep(0.1)

    async def _listen(self) -> str:
        logger.info("🎤 USER: (Listening...)")

        # KEYBOARD INPUT FOR TESTING
        loop = asyncio.get_running_loop()
        user_input = await loop.run_in_executor(None, input, ">> TYPE ANSWER: ")

        if not user_input.strip():
            return "I don't know."

        return user_input

    # --- MAIN LOOP ---
    async def run(self):
        logger.info("--- SESSION START ---")

        while self.is_running:
            # 1. INIT
            if self.state == InterviewState.INIT:
                await self._speak("Tell me about yourself.")
                await self._listen()
                await self._speak(f"Thanks. Let's start the technical interview.")
                self.state = InterviewState.SELECT_Q

            # 2. SELECT QUESTION
            elif self.state == InterviewState.SELECT_Q:
                prev_tags = (
                    self.ctx.current_question.tags
                    if self.ctx.current_question
                    else None
                )

                self.ctx.current_question = self.logic.select_next_question(
                    self.ctx.difficulty_label, prev_tags
                )

                if self.ctx.current_question.id == "0":
                    self.state = InterviewState.TERMINAL
                else:
                    self.state = InterviewState.ASK

            # 3. ASK
            elif self.state == InterviewState.ASK:
                await self._speak(self.ctx.current_question.text)
                self.state = InterviewState.LISTEN

            # 4. LISTEN
            elif self.state == InterviewState.LISTEN:
                answer = await self._listen()
                self.ctx.last_user_text = answer

                if "stop interview" in answer.lower():
                    self.state = InterviewState.TERMINAL
                else:
                    self.state = InterviewState.GRADE

            # 5. GRADE
            elif self.state == InterviewState.GRADE:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.logic.grade_answer,
                    self.ctx.current_question.id,
                    self.ctx.last_user_text,
                )

                self.ctx.last_result = result
                self.ctx.history.append(result)
                logger.info(f"📝 GRADE: {result.is_correct} ({result.score:.2f})")

                self.state = InterviewState.FEEDBACK

            # 6. FEEDBACK
            elif self.state == InterviewState.FEEDBACK:
                if not self.ctx.last_result.is_correct:
                    await self._speak(self.ctx.last_result.feedback)
                elif random.random() > 0.7:
                    await self._speak("Good.")

                self.state = InterviewState.ADAPT

            # 7. ADAPT (EMA Policy)
            elif self.state == InterviewState.ADAPT:
                alpha = 0.3
                current_perf = 1.0 if self.ctx.last_result.is_correct else 0.0

                self.ctx.difficulty_score = (alpha * current_perf) + (
                    (1 - alpha) * self.ctx.difficulty_score
                )

                if self.ctx.difficulty_score > 0.75:
                    self.ctx.difficulty_label = "Hard"
                elif self.ctx.difficulty_score < 0.35:
                    self.ctx.difficulty_label = "Easy"
                else:
                    self.ctx.difficulty_label = "Medium"

                logger.info(
                    f"⚙️ ADAPT: Score={self.ctx.difficulty_score:.2f} -> {self.ctx.difficulty_label}"
                )
                self.state = InterviewState.SELECT_Q

            # 8. TERMINAL
            elif self.state == InterviewState.TERMINAL:
                await self._speak("Interview complete. Goodbye.")
                self.is_running = False


if __name__ == "__main__":
    session = InterviewStateMachine()
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        logger.info("Session Interrupted.")
