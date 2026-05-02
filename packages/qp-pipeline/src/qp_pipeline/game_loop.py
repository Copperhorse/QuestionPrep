"""
game_loop.py — State-Machine Driven Interview Orchestrator

"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import uuid
from collections import defaultdict  # You can move this import to the top of the file
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qp_core.DBManager import DBManager

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sentence_transformers import util as st_util

    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.rule import Rule
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH = project_root / "data" / "rag_staging.db"
CHROMA_DIR = project_root / "data" / "chroma_store"

# ── Models ────────────────────────────────────────────────────────────────────
BGE_MODEL = "BAAI/bge-small-en-v1.5"
# NLI cross-encoder — label order for this model: [contradiction, entailment, neutral]
# https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768
CE_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
CE_LABEL_CONTRADICTION = 0
CE_LABEL_ENTAILMENT = 1
CE_LABEL_NEUTRAL = 2

# ── Scoring thresholds ────────────────────────────────────────────────────────
LEXICAL_PASS_THRESHOLD = 0.75  # Stage 1: skip remaining stages if this is met
CE_MIN_BI_SCORE = 0.35  # Stage 3: don't bother running CE below this
CONTRADICTION_THRESHOLD = 0.55  # p(contradiction) above this → apply hard cap
ENTAILMENT_THRESHOLD = 0.50  # p(entailment) above this → trust bi-encoder
SIMILARITY_FLOOR = 0.35  # Below this → "does not align" feedback
CONFIDENCE_HINT = 0.60  # Above this → "aligns well" feedback
QUOTE_MIN_CHARS = 40  # require substantive quotes
NEUTRAL_FLOOR: float = float(os.environ.get("SCORING_NEUTRAL_FLOOR", "0.35"))
MIN_LEXICAL_WORDS = 4  # answers shorter than this cannot pass via lexical alone

# Length-ratio guard — flags suspicious answer lengths for penalty/override
LENGTH_RATIO_MIN = 0.85  # below this → likely cutoff / incomplete
LENGTH_RATIO_MAX = 1.50  # above this → likely rambling / hallucination
# ── Logging ──────────────────────────────────────────────────────────────────


logger = logging.getLogger("GameLoop")


def _quote_is_grounded(quote: str, content: str) -> tuple[bool, str]:
    """
    Returns (is_grounded, reason).
    Two-step check:
      1. Minimum length — short quotes are not specific enough to be useful.
      2. Exact or near-exact substring — the quote must appear in the content
         with high fidelity (not just share common words).
    """
    quote = quote.strip()
    if len(quote) < QUOTE_MIN_CHARS:
        return False, f"Quote too short ({len(quote)} chars, min {QUOTE_MIN_CHARS})"
    # Try exact first (fast)
    if quote.lower() in content.lower():
        return True, "exact"
    # Fuzzy fallback for minor OCR/formatting differences
    score = fuzz.partial_ratio(quote.lower(), content.lower())
    if score >= LEXICAL_PASS_THRESHOLD:
        return True, f"fuzzy ({score:.0f}%)"
    return False, f"not found ({score:.0f}%)"


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


class InterviewState(Enum):
    INIT = auto()
    IN_PROGRESS = auto()
    TERMINAL = auto()


@dataclass
class QuestionObj:
    id: str
    text: str
    answer: str
    type: str
    difficulty: str
    tags: List[str]


@dataclass
class NLIResult:
    """Raw output from the cross-encoder stage."""

    p_entailment: float
    p_neutral: float
    p_contradiction: float

    @property
    def verdict(self) -> str:
        """Dominant label as a string."""
        scores = {
            "entailment": self.p_entailment,
            "neutral": self.p_neutral,
            "contradiction": self.p_contradiction,
        }
        return max(scores, key=scores.__getitem__)

    @property
    def is_contradictory(self) -> bool:
        return self.p_contradiction >= CONTRADICTION_THRESHOLD

    @property
    def is_entailed(self) -> bool:
        return self.p_entailment >= ENTAILMENT_THRESHOLD


@dataclass
class TurnResult:
    question_id: str
    user_text: str
    similarity: float  # final adjusted score
    bi_score: float  # raw bi-encoder cosine before NLI adjustment
    lexical_score: float  # bidirectional fuzzy lexical score
    length_ratio: float  # len(user_text) / len(reference)
    confidence: float  # = similarity (kept for API compatibility)
    feedback: str
    grader: str = "lexical"
    nli: Optional[NLIResult] = None  # None when CE wasn't run


@dataclass
class InterviewContext:
    current_question: Optional[QuestionObj] = None
    history: List[TurnResult] = field(default_factory=list)
    difficulty_score: float = 0.5
    difficulty_label: str = "Medium"


# ══════════════════════════════════════════════════════════════════════════════
# CHROMA VERIFIER  (diagnostic — only used in --debug mode)
# ══════════════════════════════════════════════════════════════════════════════


class ChromaVerifier:
    COLLECTION = "qa_pairs"

    def __init__(self, persist_directory: str = str(CHROMA_DIR)):
        self._collection = None
        self._model: Optional[SentenceTransformer] = None
        self._ready = False
        if not CHROMA_AVAILABLE or not Path(persist_directory).exists():
            return
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            self._collection = client.get_collection(self.COLLECTION)
            self._ready = self._collection.count() > 0
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return self._ready

    def count(self) -> int:
        if not self._ready:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def _get_model(self) -> Optional[SentenceTransformer]:
        if not BGE_AVAILABLE:
            return None
        if self._model is None:
            try:
                self._model = SentenceTransformer(BGE_MODEL, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(BGE_MODEL)
        return self._model

    def query(self, text: str, n: int = 3) -> List[Dict]:
        if not self._ready:
            return []
        model = self._get_model()
        if not model:
            return []
        try:
            embedding = model.encode(text, normalize_embeddings=True).tolist()
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(n, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            hits = []
            for qid, doc, meta, dist in zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                hits.append(
                    {
                        "question_id": qid,
                        "document": doc,
                        "score": round(1.0 - dist, 4),
                        "metadata": meta,
                    }
                )
            return hits
        except Exception as e:
            logger.warning(f"ChromaVerifier.query failed: {e}")
            return []


# ══════════════════════════════════════════════════════════════════════════════
# LOGIC ENGINE  — all three scoring stages live here
# ══════════════════════════════════════════════════════════════════════════════


class LogicEngine:
    """
    Thread-safe scoring engine.

    B19: _load_lock is held only during model initialisation (lazy, once each).
         encode() and predict() run without any lock — both are thread-safe
         once the model weights are in memory.

    Class-level model singletons are shared across all LogicEngine instances
    (i.e., all concurrent interview sessions) so each model is only loaded once.
    """

    # ── Bi-encoder (stage 2) ──────────────────────────────────────────────────
    _bi_encoder: Optional[SentenceTransformer] = None
    _bi_lock = threading.Lock()
    _bi_loaded = False

    # ── Cross-encoder NLI (stage 3) ───────────────────────────────────────────
    _ce_model: Optional[CrossEncoder] = None
    _ce_lock = threading.Lock()
    _ce_loaded = False
    _ce_skipped = False  # True if download was interrupted or failed

    @classmethod
    def _get_bi_encoder(cls) -> Optional[SentenceTransformer]:
        """
        Lazy-load BGE exactly once, thread-safely.

        _bi_loaded is only set True when the model is fully loaded.
        If interrupted (Ctrl+C) or download fails, it stays False so the
        next call retries rather than returning a None model silently.
        """
        if not cls._bi_loaded:
            with cls._bi_lock:
                if not cls._bi_loaded:
                    if BGE_AVAILABLE:
                        try:
                            logger.info(f"Loading bi-encoder: {BGE_MODEL}")

                            try:
                                cls._bi_encoder = SentenceTransformer(
                                    BGE_MODEL, local_files_only=True
                                )
                            except Exception:
                                logger.info(
                                    "Bi-encoder not fully cached — downloading…"
                                )
                                cls._bi_encoder = SentenceTransformer(BGE_MODEL)
                            cls._bi_loaded = True  # only mark loaded on success
                        except KeyboardInterrupt:
                            logger.warning("Bi-encoder download interrupted by user.")
                            raise  # let it propagate — user wants to quit
                        except Exception as e:
                            logger.error(f"Failed to load bi-encoder: {e}")
                            # _bi_loaded stays False — next call will retry
                    else:
                        cls._bi_loaded = True  # BGE not available, don't retry
        return cls._bi_encoder

    @classmethod
    def _get_ce_model(cls) -> Optional[CrossEncoder]:
        """
        Lazy-load the NLI cross-encoder exactly once, thread-safely.

        The CE is optional — if its download is interrupted (Ctrl+C) or fails,
        we degrade gracefully to bi-encoder-only scoring rather than crashing.

        _ce_loaded is only set True when the model is fully loaded OR when the
        user has explicitly skipped it. It stays False on failure/interruption
        so the next call could retry (e.g. if the user re-runs the script).

        _ce_skipped is set True on interruption so the session knows to report
        "bi-encoder only" in the banner rather than silently showing no NLI data.
        """
        if not cls._ce_loaded and not cls._ce_skipped:
            with cls._ce_lock:
                if not cls._ce_loaded and not cls._ce_skipped:
                    if BGE_AVAILABLE:
                        try:
                            logger.info(f"Loading cross-encoder: {CE_NLI_MODEL}")
                            # Same try-local-first pattern: avoids the network HEAD
                            # check when cached, falls back to download if the cache
                            # is partial (interrupted previous download).
                            try:
                                cls._ce_model = CrossEncoder(
                                    CE_NLI_MODEL, local_files_only=True
                                )
                            except Exception:
                                logger.info(
                                    "Cross-encoder not fully cached — downloading…"
                                )
                                cls._ce_model = CrossEncoder(CE_NLI_MODEL)
                            cls._ce_loaded = True  # only mark loaded on success
                        except KeyboardInterrupt:
                            # User pressed Ctrl+C during download.
                            # Don't crash — degrade to bi-encoder-only and continue.
                            cls._ce_skipped = True
                            cls._ce_model = None
                            logger.warning(
                                "Cross-encoder download interrupted. "
                                "Continuing with bi-encoder scoring only."
                            )
                            # Do NOT re-raise — the session should continue
                        except Exception as e:
                            cls._ce_skipped = True
                            cls._ce_model = None
                            logger.error(
                                f"Cross-encoder failed to load: {e}  "
                                "Continuing with bi-encoder scoring only."
                            )
                    else:
                        cls._ce_loaded = True
        return cls._ce_model

    def __init__(self, db_path, user_id: str, file_id: Optional[str] = None):
        self.db = DBManager(db_path)
        self.user_id = user_id
        self.file_id = file_id  # Save it
        self.questions: Dict[str, QuestionObj] = {}
        self.by_difficulty: Dict[str, List[str]] = defaultdict(list)
        self._load_questions()

    def _load_questions(self):
        # Route logic based on whether we have a file_id
        if self.file_id:
            rows = self.db.get_questions_for_file(self.file_id)
            if not rows:
                logger.warning(f"No questions for file {self.file_id}")
                return
        else:
            rows = self.db.get_questions_for_user(self.user_id)
            if not rows:
                logger.warning(
                    f"No questions for user {self.user_id[:8]} — "
                    "ingest a PDF, enrich it, index it, and assign it to this user."
                )
                return
        for r in rows:
            tags = r.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (ValueError, TypeError):
                    tags = []
            q = QuestionObj(
                id=r["question_id"],
                text=r["question_text"],
                answer=r["answer_text"],
                type=r["question_type"],
                difficulty=r.get("difficulty") or "Medium",
                tags=tags,
            )
            self.questions[q.id] = q
            self.by_difficulty[q.difficulty.lower()].append(q.id)

    def select_next_question(
        self, difficulty: str, used_ids: Set[str]
    ) -> Optional[QuestionObj]:
        candidates = [
            self.questions[qid]
            for qid in self.by_difficulty.get(difficulty.lower(), [])
            if qid not in used_ids
        ]
        if not candidates:
            candidates = [q for q in self.questions.values() if q.id not in used_ids]
        return random.choice(candidates) if candidates else None

    # ── Stage 2: Bi-encoder ───────────────────────────────────────────────────

    def _bi_encode_score(self, user_text: str, reference: str) -> float:
        """
        Encode both texts independently and return cosine similarity [0, 1].

        Good for: paraphrase detection, synonym tolerance, lexical variation.
        Weakness: "on-topic but wrong" answers score high because the embedding
                  space clusters by subject, not by logical content.
        """
        model = self._get_bi_encoder()
        if not model:
            return 0.0
        # No lock needed — SentenceTransformer.encode() is thread-safe once loaded
        embs = model.encode([user_text, reference], convert_to_tensor=True)
        score = float(st_util.cos_sim(embs[0], embs[1]))
        return max(0.0, min(1.0, score))

    # ── Stage 3: Cross-encoder NLI ────────────────────────────────────────────

    def _nli_score(self, user_text: str, reference: str) -> Optional[NLIResult]:
        """
        Feed both (reference→user) and (user→reference) to the NLI cross-encoder
        in a single batched predict() call, then apply priority logic:

          Priority 1 — Contradiction: checked on FWD direction only.
            The reverse direction throws false-positive contradictions on paraphrases,
            so we never use it for contradiction detection.
          Priority 2 — Entailment: rewarded from EITHER direction.
            A detailed answer that *contains* the reference entails it in the REV
            direction; a paraphrase entails in the FWD direction.  Taking the
            stronger entailment signal from either catches both.
          Priority 3 — Neutral: default to FWD.

        Returns None if the model is not available.

        Note on label order for cross-encoder/nli-deberta-v3-xsmall:
          index 0 = contradiction
          index 1 = entailment
          index 2 = neutral
        """
        model = self._get_ce_model()
        if not model:
            return None

        # Batch-predict both directions at once (one forward pass overhead)
        logits = model.predict(
            [
                [reference, user_text],  # FWD: ref is premise, user is hypothesis
                [user_text, reference],  # REV: user is premise, ref is hypothesis
            ]
        )

        probs_fwd = self._softmax(logits[0])
        probs_rev = self._softmax(logits[1])

        # Priority 1: contradiction — FWD direction only (stricter, fewer false positives)
        if probs_fwd[CE_LABEL_CONTRADICTION] >= CONTRADICTION_THRESHOLD:
            best_probs = probs_fwd

        # Priority 2: entailment — reward from either direction
        elif (
            probs_fwd[CE_LABEL_ENTAILMENT] >= ENTAILMENT_THRESHOLD
            or probs_rev[CE_LABEL_ENTAILMENT] >= ENTAILMENT_THRESHOLD
        ):
            best_probs = (
                probs_fwd
                if probs_fwd[CE_LABEL_ENTAILMENT] >= probs_rev[CE_LABEL_ENTAILMENT]
                else probs_rev
            )

        # Priority 3: neutral — fall back to FWD
        else:
            best_probs = probs_fwd

        return NLIResult(
            p_contradiction=float(best_probs[CE_LABEL_CONTRADICTION]),
            p_entailment=float(best_probs[CE_LABEL_ENTAILMENT]),
            p_neutral=float(best_probs[CE_LABEL_NEUTRAL]),
        )

    @staticmethod
    def _softmax(logits) -> np.ndarray:
        e = np.exp(np.array(logits, dtype=np.float64) - np.max(logits))
        return e / e.sum()

    @staticmethod
    def _lexical_score(user_text: str, reference: str) -> float:
        if len(user_text.split()) < MIN_LEXICAL_WORDS:
            return 0.0  # too short to be a real answer; force to semantic stage
        # Bidirectional: min of both directions catches prefix cutoffs and
        # hallucinations that happen to embed the full reference as a substring.
        score_fwd = fuzz.partial_ratio(user_text.lower(), reference.lower()) / 100.0
        score_rev = fuzz.partial_ratio(reference.lower(), user_text.lower()) / 100.0
        return min(score_fwd, score_rev)

    # ── NLI adjustment formula ────────────────────────────────────────────────

    @staticmethod
    def _apply_nli_adjustment(bi_score: float, nli: NLIResult) -> Tuple[float, str]:
        """
        Adjust the bi-encoder score using cross-encoder NLI probabilities.

        Returns (adjusted_score, explanation_for_feedback).

        Three cases:

        1. CONTRADICTION (p_contra ≥ 0.55)
           Hard cap regardless of bi-encoder score.
           cap = bi_score * (1 - p_contra)
           e.g. bi=0.80, p_contra=0.70 → 0.80 * 0.30 = 0.24

           Rationale: a contradictory answer is wrong even if it uses the same
           vocabulary as the reference. The cap is proportional so a weak
           contradiction (0.56) only lightly penalises, while a strong one (0.90)
           nearly zeros the score.

        2. ENTAILMENT (p_entail ≥ 0.50)
           Trust the bi-encoder score. Apply a small reward proportional to how
           confident the entailment is, capped at 1.0.
           adjusted = min(1.0, bi_score * (1 + 0.15 * (p_entail - 0.50)))
           e.g. bi=0.70, p_entail=0.80 → 0.70 * 1.045 = 0.73

           Rationale: if the cross-encoder says the user's answer entails the
           reference, the bi-encoder score is likely accurate — the answers cover
           the same conceptual ground.

        3. NEUTRAL (everything else)
           Scale by entailment confidence.
           adjusted = bi_score * (0.50 + 0.50 * p_entail)
           e.g. bi=0.67, p_entail=0.25 → 0.67 * 0.625 = 0.42
        """
        p_e = nli.p_entailment
        p_c = nli.p_contradiction

        if nli.is_contradictory:
            adjusted = bi_score * (1.0 - p_c)
            reason = "contradicts"
        elif nli.is_entailed:
            boost = 1.0 + 0.15 * (p_e - ENTAILMENT_THRESHOLD)
            adjusted = min(1.0, bi_score * boost)
            reason = "entails"
        else:
            scale = NEUTRAL_FLOOR + (1.0 - NEUTRAL_FLOOR) * p_e
            adjusted = bi_score * scale
            reason = "neutral"

        return max(0.0, min(1.0, adjusted)), reason

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def analyze_response(self, q: QuestionObj, user_text: str) -> TurnResult:
        # 1. Lexical (bidirectional — always run, no shortcut)
        lexical_score = self._lexical_score(user_text, q.answer)

        # Length ratio guard
        len_ratio = len(user_text) / max(len(q.answer), 1)
        length_suspicious = not (LENGTH_RATIO_MIN <= len_ratio <= LENGTH_RATIO_MAX)

        # Quote grounding check (is the full reference contained in the user text?)
        quote_verdict, _quote_reason = _quote_is_grounded(
            quote=q.answer.lower(), content=user_text.lower()
        )

        # 2. Bi-encoder (always run — no lexical shortcut)
        bi_score = self._bi_encode_score(user_text, q.answer)

        # 3. Cross-encoder NLI (run if bi_score is high enough to add signal)
        nli_result = None
        if bi_score >= CE_MIN_BI_SCORE:
            nli_result = self._nli_score(user_text, q.answer)

        grader_used = "bi-encoder"
        similarity = bi_score

        if nli_result is not None:
            max_nli_prob = max(
                nli_result.p_entailment,
                nli_result.p_neutral,
                nli_result.p_contradiction,
            )

            if max_nli_prob >= 0.60:
                if nli_result.p_neutral == max_nli_prob and bi_score >= 0.85:
                    similarity = bi_score
                    grader_used = "bi-encoder (CE neutral override)"
                else:
                    similarity, _ = self._apply_nli_adjustment(bi_score, nli_result)
                    grader_used = "bi+ce"
            else:
                grader_used = "bi-encoder (CE ignored < 0.60)"

        # 4. Lexical override — ONLY for true verbatim copies
        #    (lexical ≥ 0.95 AND length is sane AND reference is grounded in the answer)
        is_verbatim = lexical_score >= 0.95 and not length_suspicious and quote_verdict
        if is_verbatim:
            if (
                nli_result is None
                or "ignored" in grader_used
                or (nli_result is not None and nli_result.is_entailed)
            ):
                similarity = max(similarity, lexical_score)
                grader_used = "lexical+bi+ce" if "ce" in grader_used else "lexical+bi"
            # If CE says contradiction with high confidence, don't let lexical override

        # 5. Length penalty — proportionally punishes incomplete / cut-off answers
        if len_ratio < LENGTH_RATIO_MIN:
            penalty = len_ratio / LENGTH_RATIO_MIN
            similarity = similarity * penalty
            grader_used += " (len penalty)"

        # 6. Feedback
        feedback = self._build_feedback(
            similarity, bi_score, lexical_score, len_ratio, nli_result
        )

        return TurnResult(
            question_id=q.id,
            user_text=user_text,
            similarity=similarity,
            bi_score=bi_score,
            lexical_score=lexical_score,
            length_ratio=len_ratio,
            confidence=similarity,
            feedback=feedback,
            grader=grader_used,
            nli=nli_result,
        )

    @staticmethod
    def _build_feedback(
        final_score: float,
        bi_score: float,
        lexical_score: float,
        len_ratio: float,
        nli: Optional[NLIResult],
    ) -> str:
        """
        Generate specific feedback using all available signals.

        Length-ratio feedback is checked first: an incomplete or bloated answer
        gets guidance before NLI-based feedback, since the score is already
        penalised and the feedback should explain why.
        """
        # Length guard feedback (checked before NLI — score was already penalised)
        if len_ratio < LENGTH_RATIO_MIN:
            return (
                "Your answer appears to be cut off or significantly shorter than expected. "
                "Please provide a complete explanation."
            )
        if len_ratio > LENGTH_RATIO_MAX:
            return (
                "Your answer is substantially longer than the reference. "
                "Please check that you have not introduced incorrect details."
            )
        if nli is None:
            # Fallback: generic feedback from score alone
            if final_score < SIMILARITY_FLOOR:
                return "Your answer does not align well with the source material."
            elif final_score >= CONFIDENCE_HINT:
                return "Your explanation aligns well with the reference concepts."
            else:
                return (
                    "Your answer captures some relevant ideas. "
                    "Consider emphasising the mechanisms described in the source."
                )

        # CE is available — give specific feedback based on the NLI verdict
        if nli.is_contradictory:
            return (
                f"Your answer contradicts the reference material "
                f"(contradiction confidence: {nli.p_contradiction:.0%}). "
                "Re-read the source passage and focus on what it explicitly states."
            )

        if nli.is_entailed:
            if final_score >= CONFIDENCE_HINT:
                return (
                    "Your answer correctly captures the reference concept. "
                    "The cross-encoder confirms strong logical alignment."
                )
            else:
                return (
                    "Your answer is logically aligned with the reference "
                    "but lacks some of the specific detail or terminology used."
                )

        # Neutral — the most instructive case
        if bi_score >= 0.60 and final_score < 0.50:
            return (
                "Your answer is on the right topic but does not directly address "
                "what the reference states. You are discussing related concepts "
                "without covering the core claim — try to answer more precisely."
            )
        elif final_score >= CONFIDENCE_HINT:
            return "Your answer addresses the concept with reasonable coverage."
        else:
            return (
                "Your answer touches on relevant ideas but the cross-encoder "
                "found weak logical alignment with the reference. "
                "Consider how your answer specifically supports the reference claim."
            )


# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW SESSION  (used by both CLI and FastAPI)
# ══════════════════════════════════════════════════════════════════════════════


class InterviewSession:
    # Update init to accept file_id
    def __init__(
        self, session_id: str, user_id: str, db_path=None, file_id: Optional[str] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        # Pass file_id to the LogicEngine
        self.logic = LogicEngine(db_path or DB_PATH, user_id, file_id)
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
        if self.state == InterviewState.TERMINAL:
            return {"error": "Session complete. No more questions available."}

        q = self.ctx.current_question
        if not q:
            return {"error": "No active question."}

        result = self.logic.analyze_response(q, user_text)
        self.ctx.history.append(result)

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
                "bi_score": result.bi_score,
                "lexical_score": result.lexical_score,
                "length_ratio": result.length_ratio,
                "confidence": result.confidence,
                "feedback": result.feedback,
                "grader": result.grader,
                "nli_verdict": result.nli.verdict if result.nli else None,
                "nli_scores": {
                    "entailment": round(result.nli.p_entailment, 3),
                    "neutral": round(result.nli.p_neutral, 3),
                    "contradiction": round(result.nli.p_contradiction, 3),
                }
                if result.nli
                else None,
            },
            "next_question": (
                self.ctx.current_question.text if self.ctx.current_question else None
            ),
            "is_terminal": is_terminal,
        }


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL INTERVIEW
# ══════════════════════════════════════════════════════════════════════════════

DIFF_COLOUR = {"Easy": "green", "Medium": "yellow", "Hard": "red"}

NLI_COLOUR = {
    "entailment": "green",
    "neutral": "yellow",
    "contradiction": "red",
}
NLI_ICON = {
    "entailment": "✓",
    "neutral": "~",
    "contradiction": "✗",
}


class TerminalInterview:
    def __init__(self, user_id: str, db_path=None, debug: bool = False):
        self.user_id = user_id
        self.debug = debug
        self.db_path = Path(db_path or DB_PATH)
        self.session = InterviewSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            db_path=str(self.db_path),
        )
        self.chroma = ChromaVerifier() if debug else None
        self._q_num = 0

    # ── Output helpers ────────────────────────────────────────────────────────

    def _p(self, *a, **kw):
        console.print(*a, **kw) if RICH_AVAILABLE else print(*a)

    def _rule(self, title=""):
        if RICH_AVAILABLE:
            console.print(Rule(title, style="dim"))
        else:
            print(f"\n{'─' * 60}  {title}")

    def _spinner(self, label):
        if RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn(f"[dim]{label}[/]"),
                transient=True,
                console=console,
            )

        class _N:
            def __enter__(self):
                print(f"  {label}")
                return self

            def __exit__(self, *_):
                pass

            def add_task(self, *a, **kw):
                return None

        return _N()

    @staticmethod
    def _is_cached(model_name: str) -> bool:
        """
        Return True if the model is already in the HuggingFace local cache.
        Uses huggingface_hub.try_to_load_from_cache() which does a pure local
        path check — no network request, no download.
        Falls back to False if huggingface_hub is not installed.
        """
        if not HF_HUB_AVAILABLE:
            return False
        result = try_to_load_from_cache(model_name, filename="config.json")
        return (
            result is not None
            and str(result) != "huggingface_hub.file_download._CACHED_NO_EXIST"
        )

    def _load_model(self, model_name: str, label: str):
        """
        Context manager that shows the right UI for model loading:

        • Already cached → transient Rich spinner (loads in ~1-3s, no output needed)
        • Not cached     → step out of Rich entirely so HuggingFace's own tqdm
                           download bars can render cleanly to stderr, then print
                           a completion message when done

        The 'with' block should call the actual model constructor.
        """
        cached = self._is_cached(model_name)

        if cached:
            # Fast path — weights already on disk, just mmap them
            return self._spinner(f"{label} (loading from cache)…")

        # Slow path — first download. Rich would swallow tqdm, so don't use it.
        class _DownloadCtx:
            def __enter__(self_):
                if RICH_AVAILABLE:
                    # Temporarily pause Rich's live display so tqdm can write freely
                    console.print(
                        f"[yellow]▶ Downloading[/] [bold]{model_name}[/] "
                        f"[dim](first run — this may take a minute)[/]"
                    )
                else:
                    print(f"  Downloading {model_name} (first run)…")
                return self_

            def __exit__(self_, *_):
                if RICH_AVAILABLE:
                    console.print(
                        f"[green]✓ Downloaded[/] [bold]{model_name}[/]  "
                        f"[dim]Cached for future runs.[/]"
                    )
                else:
                    print(f"  ✓ {model_name} downloaded and cached.")

        return _DownloadCtx()

    # ── Banner ────────────────────────────────────────────────────────────────

    def _banner(self):
        questions = self.session.logic.questions
        total = len(questions)
        db = DBManager(str(self.db_path))
        user = db.get_user_by_id(self.user_id)
        uname = user["username"] if user else self.user_id[:8]

        if not RICH_AVAILABLE:
            print(f"\n  QuestionPrep — Terminal Interview")
            print(f"  User: {uname}   Questions: {total}")
            if self.debug and self.chroma:
                st = (
                    f"ready — {self.chroma.count()} embeddings"
                    if self.chroma.ready
                    else "NOT READY"
                )
                print(f"  Chroma: {st}")
            print(f"  Graders: lexical → bi-encoder (BGE) → cross-encoder NLI")
            return

        easy = sum(1 for q in questions.values() if q.difficulty == "Easy")
        medium = sum(1 for q in questions.values() if q.difficulty == "Medium")
        hard = sum(1 for q in questions.values() if q.difficulty == "Hard")

        body = (
            f"[bold]Welcome, {uname}[/]\n\n"
            f"[dim]Questions loaded:[/] [bold]{total}[/]  "
            f"([green]Easy {easy}[/]  [yellow]Medium {medium}[/]  [red]Hard {hard}[/])\n\n"
            f"[dim]Scoring pipeline:[/]\n"
            f"  [dim]Stage 1[/]  Lexical          [dim]rapidfuzz partial_ratio[/]\n"
            f"  [dim]Stage 2[/]  Bi-encoder BGE   [dim]{BGE_MODEL}[/]\n"
            f"  [dim]Stage 3[/]  Cross-encoder NLI [dim]{CE_NLI_MODEL}[/]\n"
        )

        if self.debug and self.chroma:
            if self.chroma.ready:
                body += f"\n[green]✓ Chroma store ready — {self.chroma.count()} embedding(s)[/]\n"
            else:
                body += "\n[red]✗ Chroma store empty or not found[/]\n"

        console.print(
            Panel(
                body,
                title="[bold]QuestionPrep — Terminal Interview[/]",
                subtitle="[dim]'skip' to skip  |  'quit' to end[/]",
                expand=False,
            )
        )

    # ── Question display ──────────────────────────────────────────────────────

    def _show_question(self, q: QuestionObj):
        self._q_num += 1
        colour = DIFF_COLOUR.get(q.difficulty, "white")
        next_d = self.session.ctx.difficulty_label

        if RICH_AVAILABLE:
            tags_str = "  ".join(f"[dim cyan]{t}[/]" for t in q.tags[:5]) or "[dim]—[/]"
            self._rule()
            console.print(
                Panel(
                    f"[bold]{q.text}[/]",
                    title=f"[dim]Q{self._q_num}[/]  [{colour}]{q.difficulty}[/]  [dim]{q.type}[/]",
                    subtitle=tags_str,
                    border_style=colour,
                    expand=False,
                    padding=(1, 2),
                )
            )
            console.print(
                f"  [dim]Next difficulty target:[/] "
                f"[bold {DIFF_COLOUR.get(next_d, 'white')}]{next_d}[/]\n"
            )
        else:
            print(f"\n─── Q{self._q_num} [{q.difficulty} / {q.type}] ───")
            print(f"  {q.text}\n")

    # ── Feedback display ──────────────────────────────────────────────────────

    def _show_feedback(self, tr: TurnResult, q: QuestionObj, chroma_hits: List[Dict]):
        pct = tr.similarity * 100
        colour = "green" if pct >= 65 else "yellow" if pct >= 45 else "red"
        filled = round(30 * tr.similarity)
        bar = "█" * filled + "░" * (30 - filled)

        if RICH_AVAILABLE:
            # ── Score line ────────────────────────────────────────────────────
            score_line = (
                f"[{colour}]{bar}[/]  [{colour}]{pct:.0f}%[/]  "
                f"[dim]grader: {tr.grader}[/]"
            )

            # ── NLI breakdown (always shown when CE ran, not just in debug) ───
            nli_line = ""
            if tr.nli:
                verdict = tr.nli.verdict
                vc = NLI_COLOUR[verdict]
                vi = NLI_ICON[verdict]
                nli_line = (
                    f"\n\n[dim]NLI verdict:[/]  "
                    f"[{vc}]{vi} {verdict.upper()}[/]  "
                    f"[dim]("
                    f"[green]entail {tr.nli.p_entailment:.0%}[/]  "
                    f"[yellow]neutral {tr.nli.p_neutral:.0%}[/]  "
                    f"[red]contradict {tr.nli.p_contradiction:.0%}[/]"
                    f")[/]"
                )
                # Show bi-encoder vs final if they differ meaningfully
                if abs(tr.bi_score - tr.similarity) > 0.04:
                    nli_line += (
                        f"\n[dim]Score adjustment:[/]  "
                        f"[dim]bi-encoder {tr.bi_score:.0%}[/]  →  "
                        f"[{colour}]final {pct:.0f}%[/]"
                    )

            console.print(
                Panel(
                    f"{score_line}{nli_line}\n\n[italic]{tr.feedback}[/]",
                    title="[dim]Evaluation[/]",
                    border_style=colour,
                    expand=False,
                    padding=(1, 2),
                )
            )

            # ── Reference answer reveal ───────────────────────────────────────
            if Confirm.ask("  [dim]Reveal reference answer?[/]", default=False):
                console.print(
                    Panel(
                        f"[dim]{q.answer}[/]",
                        title="[dim]Reference Answer[/]",
                        expand=False,
                        padding=(1, 2),
                    )
                )

            # ── Chroma debug panel ────────────────────────────────────────────
            if self.debug and chroma_hits:
                self._chroma_panel(q, chroma_hits, tr.similarity)
            elif self.debug and self.chroma and self.chroma.ready and not chroma_hits:
                console.print("[dim]  (Chroma returned no hits)[/]\n")

        else:
            print(f"\n  [{bar}] {pct:.0f}%  ({tr.grader})")
            if tr.nli:
                print(
                    f"  NLI: {tr.nli.verdict.upper()}  "
                    f"(E={tr.nli.p_entailment:.0%}  "
                    f"N={tr.nli.p_neutral:.0%}  "
                    f"C={tr.nli.p_contradiction:.0%})"
                )
                if abs(tr.bi_score - tr.similarity) > 0.04:
                    print(
                        f"  Adjustment: bi-encoder {tr.bi_score:.0%} → final {pct:.0f}%"
                    )
            print(f"  {tr.feedback}")
            if input("  Reveal reference answer? (y/n): ").strip().lower() == "y":
                print(f"\n  Reference: {q.answer}\n")
            if self.debug and chroma_hits:
                self._chroma_plain(q, chroma_hits)

    # ── Chroma debug panel ────────────────────────────────────────────────────

    def _chroma_panel(self, q: QuestionObj, hits: List[Dict], final_score: float):
        table = Table(
            "Rank",
            "Chroma score",
            "Match?",
            "Stored document (first 90 chars)",
            title="[dim italic]Chroma retrieval — top hits for your answer[/]",
            show_lines=True,
            border_style="dim",
            title_style="dim italic",
        )
        for i, hit in enumerate(hits[:3], 1):
            is_match = hit["question_id"] == q.id
            match_cell = "[bold green]✓ correct[/]" if is_match else "[dim]—[/]"
            score_cell = (
                f"[green]{hit['score']:.3f}[/]"
                if hit["score"] >= 0.5
                else f"[dim]{hit['score']:.3f}[/]"
            )
            preview = (hit["document"] or "").replace("\n", " ")[:90] + "…"
            table.add_row(str(i), score_cell, match_cell, f"[dim]{preview}[/]")

        top_match = hits[0]["question_id"] == q.id if hits else False
        verdict = (
            "[green]✓ Chroma returned the correct Q&A as the top hit.[/]"
            if top_match
            else (
                f"[yellow]⚠ Correct Q&A was hit "
                f"#{next((i + 1 for i, h in enumerate(hits) if h['question_id'] == q.id), '?')}"
                f", not #1.  final={final_score:.3f}[/]"
            )
        )
        console.print()
        console.print(table)
        console.print(f"  {verdict}\n")

    def _chroma_plain(self, q: QuestionObj, hits: List[Dict]):
        print("\n  Chroma top hits:")
        for i, h in enumerate(hits[:3], 1):
            sym = "✓" if h["question_id"] == q.id else " "
            print(f"    {i}. [{sym}] score={h['score']:.3f}  {h['document'][:70]}…")
        print()

    # ── Session summary ───────────────────────────────────────────────────────

    def _summary(self):
        history = self.session.ctx.history
        if not history:
            self._p("[dim]No answers recorded.[/]")
            return

        avg = sum(r.similarity for r in history) / len(history)
        best = max(history, key=lambda r: r.similarity)
        worst = min(history, key=lambda r: r.similarity)

        if RICH_AVAILABLE:
            table = Table(
                "Q#",
                "Final",
                "Bi-enc",
                "NLI verdict",
                "Grader",
                "Feedback (preview)",
                title="Session Summary",
                show_lines=True,
            )
            for i, r in enumerate(history, 1):
                pct = r.similarity * 100
                bipct = r.bi_score * 100
                col = "green" if pct >= 65 else "yellow" if pct >= 45 else "red"
                bicol = "green" if bipct >= 65 else "yellow" if bipct >= 45 else "red"

                if r.nli:
                    vc = NLI_COLOUR[r.nli.verdict]
                    vi = NLI_ICON[r.nli.verdict]
                    nli_cell = f"[{vc}]{vi} {r.nli.verdict[:7]}[/]"
                else:
                    nli_cell = "[dim]—[/]"

                table.add_row(
                    str(i),
                    f"[{col}]{pct:.0f}%[/]",
                    f"[{bicol}]{bipct:.0f}%[/]",
                    nli_cell,
                    r.grader,
                    r.feedback[:50] + ("…" if len(r.feedback) > 50 else ""),
                )

            console.print()
            console.print(table)
            console.print(
                Panel(
                    f"[bold]Average final score:[/]  {avg * 100:.0f}%\n"
                    f"[bold]Best:[/]     Q{history.index(best) + 1}  "
                    f"({best.similarity * 100:.0f}%)\n"
                    f"[bold]Weakest:[/]  Q{history.index(worst) + 1}  "
                    f"({worst.similarity * 100:.0f}%)\n"
                    f"[bold]Answered:[/] {len(history)}  |  "
                    f"[bold]Final difficulty:[/] {self.session.ctx.difficulty_label}",
                    title="[bold]Overall[/]",
                    expand=False,
                    padding=(1, 2),
                )
            )
        else:
            print("\n  ── Summary ──")
            for i, r in enumerate(history, 1):
                nli_str = f"  [{r.nli.verdict[:1].upper()}]" if r.nli else ""
                print(
                    f"  Q{i}: {r.similarity * 100:.0f}% (bi:{r.bi_score * 100:.0f}%){nli_str}  {r.feedback[:55]}"
                )
            print(f"\n  Average: {avg * 100:.0f}%  |  Answered: {len(history)}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if not RICH_AVAILABLE:
            print("Tip: pip install rich  for a nicer terminal experience.\n")

        self._banner()

        if not self.session.logic.questions:
            self._p(
                "[red bold]No questions found for this user.[/]\n\n"
                "[dim]Steps to fix:[/]\n"
                "  1. Upload and ingest a PDF    [dim]POST /api/files/ingest[/]\n"
                "  2. Run enrichment             [dim]POST /api/questions/generate[/]\n"
                "  3. Run VectorIndexer          [dim]VectorIndexer().index_file(file_id)[/]\n"
                "  4. Assign the file to the user [dim]DBManager.assign_file_to_user[/]\n"
            )
            return

        with self._load_model(BGE_MODEL, "Bi-encoder BGE"):
            LogicEngine._get_bi_encoder()

        with self._load_model(CE_NLI_MODEL, "Cross-encoder NLI"):
            LogicEngine._get_ce_model()

        if LogicEngine._ce_skipped:
            if RICH_AVAILABLE:
                console.print(
                    "[yellow]⚠ Cross-encoder not loaded[/] — "
                    "scoring will use [bold]bi-encoder only[/] (Stage 1 + 2).\n"
                    "[dim]Re-run the script to retry the download.[/]\n"
                )
            else:
                print("  ⚠ Cross-encoder not loaded — bi-encoder scoring only.")
                print("  Re-run the script to retry the download.\n")

        if self.debug and self.chroma and not self.chroma.ready:
            self._p("[yellow]⚠ Chroma store empty — run VectorIndexer first.[/]\n")

        q_text = self.session.start_interview()
        if not q_text:
            self._p("[red]Could not load the first question.[/]")
            return

        while True:
            q = self.session.ctx.current_question
            self._show_question(q)

            try:
                if RICH_AVAILABLE:
                    answer = Prompt.ask(
                        "  [bold]Your answer[/]  [dim](or 'skip' / 'quit')[/]"
                    ).strip()
                else:
                    answer = input("  Your answer (or 'skip' / 'quit'): ").strip()
            except (KeyboardInterrupt, EOFError):
                self._p("\n[dim]Interrupted.[/]")
                break

            cmd = answer.lower()
            if cmd in ("quit", "q", "exit"):
                self._p("[dim]Ending session early.[/]")
                break

            if cmd in ("skip", "s", ""):
                self._p("[dim]Skipped.[/]")
                nxt = self.session.logic.select_next_question(
                    self.session.ctx.difficulty_label, self.session.used_ids
                )
                if not nxt:
                    break
                self.session.ctx.current_question = nxt
                self.session.used_ids.add(nxt.id)
                continue

            chroma_hits: List[Dict] = []
            if self.debug and self.chroma and self.chroma.ready:
                chroma_hits = self.chroma.query(answer, n=3)

            result_dict = self.session.evaluate_turn(answer)
            if "error" in result_dict:
                self._p(f"[red]{result_dict['error']}[/]")
                break

            tr = self.session.ctx.history[-1]
            self._show_feedback(tr, q, chroma_hits)

            if result_dict["is_terminal"]:
                self._rule()
                self._p("\n[bold green]  All questions answered — great work![/]\n")
                break

        self._rule("End of Session")
        self._summary()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def _pick_user(db_path: Path) -> Optional[str]:
    db = DBManager(str(db_path))
    users = db.list_users()
    if not users:
        msg = "No users found. Sign up first via POST /api/auth/signup."
        console.print(f"[red]{msg}[/]") if RICH_AVAILABLE else print(msg)
        return None

    if RICH_AVAILABLE:
        table = Table(
            "Index",
            "User ID (first 8)",
            "Username",
            "Email",
            title="Users",
            show_lines=False,
        )
        for i, u in enumerate(users, 1):
            table.add_row(str(i), u["user_id"][:8], u["username"], u["email"])
        console.print(table)
        choice = Prompt.ask(
            "Select a user [bold](number or user_id)[/]", default="1"
        ).strip()
    else:
        for i, u in enumerate(users, 1):
            print(f"  {i}. {u['username']}  ({u['user_id'][:8]})")
        choice = input("Select user (number or user_id): ").strip()

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(users):
            return users[idx]["user_id"]
        print("Invalid index.")
        return None

    for u in users:
        if u["user_id"].startswith(choice) or u["username"] == choice:
            return u["user_id"]

    print(f"User '{choice}' not found.")
    return None


def main():
    # ── Terminal logging setup ───────────────────────────────────────────────
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.DEBUG)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)  # INFO in terminal; FastAPI controls its own level
    logger.propagate = False  # don't double-emit to root logger

    parser = argparse.ArgumentParser(
        description="QuestionPrep — Terminal Interview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  python game_loop.py                       # interactive user-picker
  python game_loop.py --debug               # Chroma retrieval + full NLI breakdown
  python game_loop.py --user <user_id>      # skip the user-picker
  python game_loop.py --list-users          # list all users and exit
  python game_loop.py --db /path/to/db      # custom DB path
        """,
    )
    parser.add_argument("--user", help="User ID (skip the interactive picker)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show Chroma retrieval hits alongside each answer score",
    )
    parser.add_argument(
        "--list-users", action="store_true", help="Print all registered users and exit"
    )
    parser.add_argument("--db", default=str(DB_PATH), help="Path to rag_staging.db")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show DEBUG-level logs (model internals, score details)",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    if args.list_users:
        db = DBManager(str(db_path))
        users = db.list_users()
        if not users:
            print("No users found.")
        else:
            for u in users:
                print(f"  {u['user_id']}  {u['username']}  {u['email']}")
        sys.exit(0)

    user_id = args.user or _pick_user(db_path)
    if not user_id:
        sys.exit(1)

    TerminalInterview(user_id=user_id, db_path=db_path, debug=args.debug).run()


if __name__ == "__main__":
    main()
