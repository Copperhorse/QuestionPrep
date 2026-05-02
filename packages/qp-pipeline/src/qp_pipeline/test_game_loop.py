#!/usr/bin/env python3
"""
test_game_loop_fixed.py — Standalone evaluation of the FIXED scoring pipeline.

Run:  python test_game_loop_fixed.py
"""

import json
import logging
import os
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import fuzz

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from sentence_transformers import util as st_util

    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    CrossEncoder = None
    SentenceTransformer = None
    st_util = None

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("FixedPipeline")

# ── Constants ────────────────────────────────────────────────────────────────
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
CE_LABEL_CONTRADICTION = 0
CE_LABEL_ENTAILMENT = 1
CE_LABEL_NEUTRAL = 2

# Thresholds
LEXICAL_PASS_THRESHOLD = 0.75
CE_MIN_BI_SCORE = 0.35
CONTRADICTION_THRESHOLD = 0.55
ENTAILMENT_THRESHOLD = 0.50
SIMILARITY_FLOOR = 0.35
CONFIDENCE_HINT = 0.60
QUOTE_MIN_CHARS = 40
NEUTRAL_FLOOR: float = float(os.environ.get("SCORING_NEUTRAL_FLOOR", "0.35"))
MIN_LEXICAL_WORDS = 4

# Length-ratio guard (NEW)
LENGTH_RATIO_MIN = 0.85  # below this → likely cutoff
LENGTH_RATIO_MAX = 1.50  # above this → likely rambling / hallucination


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
    p_entailment: float
    p_neutral: float
    p_contradiction: float

    @property
    def verdict(self) -> str:
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
    similarity: float
    bi_score: float
    lexical_score: float
    length_ratio: float
    confidence: float
    feedback: str
    grader: str = "lexical"
    nli: Optional[NLIResult] = None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _quote_is_grounded(quote: str, content: str) -> tuple[bool, str]:
    quote = quote.strip()
    if len(quote) < QUOTE_MIN_CHARS:
        return False, f"Quote too short ({len(quote)} chars, min {QUOTE_MIN_CHARS})"
    if quote.lower() in content.lower():
        return True, "exact"
    score = fuzz.partial_ratio(quote.lower(), content.lower())
    if score >= LEXICAL_PASS_THRESHOLD * 100:
        return True, f"fuzzy ({score:.0f}%)"
    return False, f"not found ({score:.0f}%)"


# ══════════════════════════════════════════════════════════════════════════════
# FIXED LOGIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class FixedLogicEngine:
    """
    Thread-safe scoring engine with the four non-SLM fixes applied.
    """

    _bi_encoder: Optional[SentenceTransformer] = None
    _bi_lock = threading.Lock()
    _bi_loaded = False

    _ce_model: Optional[CrossEncoder] = None
    _ce_lock = threading.Lock()
    _ce_loaded = False
    _ce_skipped = False

    @classmethod
    def _get_bi_encoder(cls) -> Optional[SentenceTransformer]:
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
                            cls._bi_loaded = True
                        except KeyboardInterrupt:
                            logger.warning("Bi-encoder download interrupted.")
                            raise
                        except Exception as e:
                            logger.error(f"Failed to load bi-encoder: {e}")
                    else:
                        cls._bi_loaded = True
        return cls._bi_encoder

    @classmethod
    def _get_ce_model(cls) -> Optional[CrossEncoder]:
        if not cls._ce_loaded and not cls._ce_skipped:
            with cls._ce_lock:
                if not cls._ce_loaded and not cls._ce_skipped:
                    if BGE_AVAILABLE:
                        try:
                            logger.info(f"Loading cross-encoder: {CE_NLI_MODEL}")
                            try:
                                cls._ce_model = CrossEncoder(
                                    CE_NLI_MODEL, local_files_only=True
                                )
                            except Exception:
                                logger.info(
                                    "Cross-encoder not fully cached — downloading…"
                                )
                                cls._ce_model = CrossEncoder(CE_NLI_MODEL)
                            cls._ce_loaded = True
                        except KeyboardInterrupt:
                            cls._ce_skipped = True
                            cls._ce_model = None
                            logger.warning(
                                "Cross-encoder download interrupted. Continuing with bi-encoder only."
                            )
                        except Exception as e:
                            cls._ce_skipped = True
                            cls._ce_model = None
                            logger.error(
                                f"Cross-encoder failed: {e}. Continuing with bi-encoder only."
                            )
                    else:
                        cls._ce_loaded = True
        return cls._ce_model

    # ── FIX 1: Bidirectional lexical scoring ────────────────────────────────
    @staticmethod
    def _lexical_score(user_text: str, reference: str) -> float:
        if len(user_text.split()) < MIN_LEXICAL_WORDS:
            return 0.0
        # OLD: one-directional — partial_ratio(user, ref)
        # NEW: bidirectional — catches prefix cutoffs and hallucinations
        score_fwd = fuzz.partial_ratio(user_text.lower(), reference.lower()) / 100.0
        score_rev = fuzz.partial_ratio(reference.lower(), user_text.lower()) / 100.0
        return min(score_fwd, score_rev)

    # ── Stage 2: Bi-encoder ─────────────────────────────────────────────────
    def _bi_encode_score(self, user_text: str, reference: str) -> float:
        model = self._get_bi_encoder()
        if not model:
            return 0.0
        embs = model.encode([user_text, reference], convert_to_tensor=True)
        score = float(st_util.cos_sim(embs[0], embs[1]))
        return max(0.0, min(1.0, score))

        # ── FIX 2: CE input order swapped ───────────────────────────────────────

    # ── FIX 2: CE Bidirectional Batching ────────────────────────────────────
    def _nli_score(self, user_text: str, reference: str) -> Optional[NLIResult]:
        model = self._get_ce_model()
        if not model:
            return None

        # Batch predict both directions at once
        logits = model.predict(
            [
                [reference, user_text],  # FWD: Ref -> User
                [user_text, reference],  # REV: User -> Ref
            ]
        )

        probs_fwd = self._softmax(logits[0])
        probs_rev = self._softmax(logits[1])

        # Priority 1: Contradiction
        # STRICT RULE: Only use the FWD direction to check for contradiction.
        # The reverse direction throws false-positive contradictions on paraphrases.
        if probs_fwd[CE_LABEL_CONTRADICTION] >= CONTRADICTION_THRESHOLD:
            best_probs = probs_fwd

        # Priority 2: Entailment
        # Reward entailment in EITHER direction (catches highly detailed answers)
        elif (
            probs_fwd[CE_LABEL_ENTAILMENT] >= ENTAILMENT_THRESHOLD
            or probs_rev[CE_LABEL_ENTAILMENT] >= ENTAILMENT_THRESHOLD
        ):
            # Take the one with the strongest entailment signal
            best_probs = (
                probs_fwd
                if probs_fwd[CE_LABEL_ENTAILMENT] > probs_rev[CE_LABEL_ENTAILMENT]
                else probs_rev
            )

        # Priority 3: Neutral
        else:
            # Default to FWD if neither strongly entails or contradicts
            best_probs = probs_fwd

        return NLIResult(
            p_contradiction=float(best_probs[CE_LABEL_CONTRADICTION]),
            p_entailment=float(best_probs[CE_LABEL_ENTAILMENT]),
            p_neutral=float(best_probs[CE_LABEL_NEUTRAL]),
        )

    # ── FIX 3 & 4: Full pipeline with length guard + no lexical shortcut ────
    def analyze_response(self, q: QuestionObj, user_text: str) -> TurnResult:
        # 1. Lexical (bidirectional)
        lexical_score = self._lexical_score(user_text, q.answer)

        # Length ratio guard
        len_ratio = len(user_text) / max(len(q.answer), 1)
        length_suspicious = not (LENGTH_RATIO_MIN <= len_ratio <= LENGTH_RATIO_MAX)

        # Quote grounding check (reference in user text)
        quote_verdict, quote_reason = _quote_is_grounded(
            quote=q.answer.lower(), content=user_text.lower()
        )

        # 2. Bi-encoder (always run — no lexical shortcut)
        bi_score = self._bi_encode_score(user_text, q.answer)

        # 3. Cross-encoder NLI (always run if available and bi_score is high enough)
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
                # --- NEW: The Neutral Override ---
                # If the bi-encoder strongly believes it's correct, but the CE is just "Neutral",
                # trust the bi-encoder. Don't let a neutral NLI scale down a great answer.
                if nli_result.p_neutral == max_nli_prob and bi_score >= 0.85:
                    similarity = bi_score
                    grader_used = "bi-encoder (CE neutral override)"
                else:
                    similarity, _ = self._apply_nli_adjustment(bi_score, nli_result)
                    grader_used = "bi+ce"
            else:
                grader_used = "bi-encoder (CE ignored < 0.60)"
        # 4. Lexical override — ONLY for true verbatim copies
        is_verbatim = lexical_score >= 0.95 and not length_suspicious and quote_verdict
        if is_verbatim:
            if nli_result is None or "ignored" in grader_used or nli_result.is_entailed:
                similarity = max(similarity, lexical_score)
                if "ce" in grader_used:
                    grader_used = "lexical+bi+ce"
                else:
                    grader_used = "lexical+bi"
            elif nli_result.is_contradictory and max_nli_prob >= 0.60:
                pass

        # 5. Length Guard Penalty (NEW: mathematically punishes cut-offs)
        if len_ratio < LENGTH_RATIO_MIN:
            # Proportional penalty: e.g., if ratio is 0.4 and min is 0.85,
            # similarity is multiplied by ~0.47, tanking the score of unfinished sentences.
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
    def _softmax(logits) -> np.ndarray:
        e = np.exp(np.array(logits, dtype=np.float64) - np.max(logits))
        return e / e.sum()

    @staticmethod
    def _apply_nli_adjustment(bi_score: float, nli: NLIResult) -> Tuple[float, str]:
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

        # ── FIX 3 & 4: Full pipeline with length guard + no lexical shortcut ────
        # ── FIX 3 & 4: Full pipeline with length guard + no lexical shortcut ────

    @staticmethod
    def _build_feedback(
        final_score: float,
        bi_score: float,
        lexical_score: float,
        len_ratio: float,
        nli: Optional[NLIResult],
    ) -> str:
        # Length guard feedback
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
            if final_score < SIMILARITY_FLOOR:
                return "Your answer does not align well with the source material."
            elif final_score >= CONFIDENCE_HINT:
                return "Your explanation aligns well with the reference concepts."
            else:
                return (
                    "Your answer captures some relevant ideas. "
                    "Consider emphasising the mechanisms described in the source."
                )

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

        # Neutral
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
# TEST DATA
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict] = []

# ── Domain 1: Self-Attention (Transformers) ─────────────────────────────────
REF_ATTENTION = (
    "Self-attention in transformers computes a weighted sum of all input tokens "
    "for each position. It uses Query, Key, and Value matrices derived from the "
    "input embeddings. The dot product of Query and Key determines the attention "
    "weights, which are then scaled and passed through a softmax function. These "
    "weights are multiplied by the Value matrix to produce the final output "
    "representation for each token."
)

TEST_CASES += [
    {
        "domain": "Self-Attention",
        "name": "1. Similar (paraphrase)",
        "user_answer": (
            "Transformers use self-attention to calculate a weighted combination of "
            "every input token at each position. The mechanism derives Query, Key, and "
            "Value vectors from the embeddings. It computes attention scores via the dot "
            "product of Query and Key, applies scaling and softmax, then uses these "
            "weights to aggregate the Value vectors into the output."
        ),
        "expected": "High score (8.5–9.5 range). Conceptually identical, different wording.",
    },
    {
        "domain": "Self-Attention",
        "name": "2. More detailed (correct extra info)",
        "user_answer": (
            "Self-attention in transformers computes a weighted sum of all input tokens "
            "for each position using Query, Key, and Value matrices derived from input "
            "embeddings. The dot product of Query and Key determines raw attention weights, "
            "which are scaled by the square root of the head dimension to prevent vanishing "
            "gradients, then passed through softmax. These normalized weights are multiplied "
            "by the Value matrix to produce the final output. In multi-head attention, this "
            "process runs in parallel across several heads with different learned projections, "
            "allowing the model to attend to information from different representation subspaces "
            "at different positions."
        ),
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Self-Attention",
        "name": "3. Contradicts reference",
        "user_answer": (
            "Self-attention ignores all other tokens and only processes the current token "
            "independently. It does not use Query, Key, or Value matrices; instead, it applies "
            "a fixed convolutional filter to each token separately. The weights are determined "
            "by a pre-trained lookup table rather than being computed dynamically via dot products."
        ),
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Self-Attention",
        "name": "4. Less detailed (vague but not wrong)",
        "user_answer": (
            "Self-attention calculates weights for tokens using dot products and then "
            "combines them to form the output."
        ),
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (Q/K/V, softmax, scaling).",
    },
    {
        "domain": "Self-Attention",
        "name": "5. Cutoff / incomplete sentence",
        "user_answer": (
            "Self-attention in transformers computes a weighted sum of all input tokens "
            "for each position. It uses Query, Key, and Value matrices derived from the "
            "input embeddings. The dot product of Query and Key determines the attention "
            "weights, which are then scaled and passed through a soft"
        ),
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Self-Attention",
        "name": "6. Totally unrelated",
        "user_answer": (
            "The water cycle involves evaporation, condensation, and precipitation. "
            "When the sun heats water in oceans and rivers, it rises into the atmosphere "
            "as vapor, cools down to form clouds, and eventually falls back to Earth as "
            "rain or snow."
        ),
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
    {
        "domain": "Self-Attention",
        "name": "7. Hallucinated / fabricated details",
        "user_answer": (
            "Self-attention in transformers computes a weighted sum of all input tokens "
            "for each position. It uses Query, Key, and Value matrices derived from the "
            "input embeddings. The dot product of Query and Key determines the attention "
            "weights, which are then scaled by the inverse cosine of the embedding dimension "
            "and passed through a ReLU activation instead of softmax. These weights are "
            "multiplied by the Value matrix and then fed into a recurrent LSTM cell to "
            "produce the final output representation for each token."
        ),
        "expected": "Low-to-moderate score (2–5). Starts correctly but introduces plausible-sounding false details.",
    },
    {
        "domain": "Self-Attention",
        "name": "8. Correct but different terminology",
        "user_answer": (
            "In transformer networks, intra-attention mechanisms generate a convex combination "
            "of source representations per time-step. The system projects hidden states into "
            "three distinct subspaces—denoted as Q, K, and V—via learned affine transformations. "
            "Affinity scores are obtained via inner-product similarity between the query and key "
            "projections, followed by temperature-scaled normalization via the Boltzmann (softmax) "
            "operator. The resulting convex coefficients are applied to the value projections to "
            "yield context-aware hidden states."
        ),
        "expected": "High score (8–9). Same concepts, different jargon.",
    },
    {
        "domain": "Self-Attention",
        "name": "9. Mostly correct with one critical error",
        "user_answer": (
            "Self-attention in transformers computes a weighted sum of all input tokens "
            "for each position. It uses Query, Key, and Value matrices derived from the "
            "input embeddings. The dot product of Query and Key determines the attention "
            "weights, which are then scaled and passed through a softmax function. These "
            "weights are added to the Value matrix to produce the final output representation "
            "for each token."
        ),
        "expected": "Moderate-to-low score (3–6). One word changed ('added' vs 'multiplied') makes the mechanism wrong.",
    },
    {
        "domain": "Self-Attention",
        "name": "10. Verbatim copy",
        "user_answer": REF_ATTENTION,
        "expected": "Very high score (9.5–10). Exact match; should trigger lexical pass and skip CE.",
    },
]

# ── Domain 2: Database Indexing (B-Tree) ────────────────────────────────────
REF_BTREE = (
    "A B-tree index in a database organizes data in a self-balancing tree structure "
    "where each node can contain multiple keys and pointers to child nodes. This design "
    "minimizes disk I/O by keeping the tree shallow, ensuring that lookups, insertions, "
    "and deletions all complete in O(log n) time. The nodes are kept sorted, and a fill "
    "factor governs how full each node can become before it splits."
)

TEST_CASES += [
    {
        "domain": "B-Tree Index",
        "name": "11. Similar (paraphrase)",
        "user_answer": (
            "Databases use B-tree indexes to arrange records in a balanced hierarchical structure. "
            "Every node stores several keys along with references to its children. Because the tree "
            "remains flat, disk reads are reduced, and search, insert, and delete operations all run "
            "in logarithmic time relative to the number of records. Keys within each node are maintained "
            "in order, and nodes split once they exceed a configured capacity threshold."
        ),
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "B-Tree Index",
        "name": "12. More detailed (correct extra info)",
        "user_answer": (
            "A B-tree index in a database organizes data in a self-balancing tree structure where each "
            "node can contain multiple keys and pointers to child nodes. This design minimizes disk I/O "
            "by keeping the tree shallow, ensuring that lookups, insertions, and deletions all complete "
            "in O(log n) time. The nodes are kept sorted, and a fill factor governs how full each node "
            "can become before it splits. In practice, most database engines use a B+ tree variant where "
            "only leaf nodes store actual record pointers or row IDs, while internal nodes serve purely "
            "as navigation keys. Leaf nodes are also linked together as a doubly-linked list, enabling "
            "efficient range scans and ordered traversal without revisiting parent nodes."
        ),
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "B-Tree Index",
        "name": "13. Contradicts reference",
        "user_answer": (
            "A B-tree index stores every record in a single root node without any child pointers, making "
            "it essentially a flat array. Lookups require a linear scan from the first element to the last, "
            "resulting in O(n) worst-case performance. The structure is intentionally unsorted to maximize "
            "write throughput, and nodes never split; instead, overflow data is written to a separate "
            "append-only log file."
        ),
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "B-Tree Index",
        "name": "14. Less detailed (vague but not wrong)",
        "user_answer": (
            "A B-tree is a tree structure used by databases to find data quickly. It keeps things sorted and balanced."
        ),
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (child pointers, fill factor, O(log n)).",
    },
    {
        "domain": "B-Tree Index",
        "name": "15. Cutoff / incomplete sentence",
        "user_answer": (
            "A B-tree index in a database organizes data in a self-balancing tree structure where each node "
            "can contain multiple keys and pointers to child nodes. This design minimizes disk I/O by keeping "
            "the tree shallow, ensuring that lookups, insertions, and deletions all complete in O(log n) time. "
            "The nodes are kept sorted, and a fill factor governs how full each node can become before it"
        ),
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "B-Tree Index",
        "name": "16. Totally unrelated",
        "user_answer": (
            "Photosynthesis occurs in the chloroplasts of plant cells, where chlorophyll absorbs sunlight "
            "to convert carbon dioxide and water into glucose and oxygen. This process is divided into the "
            "light-dependent reactions and the Calvin cycle, which together provide the energy and organic "
            "compounds necessary for plant growth."
        ),
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

# ── Domain 3: TCP Three-Way Handshake ───────────────────────────────────────
REF_TCP = (
    "The TCP three-way handshake establishes a reliable connection between a client and a server. "
    "The client first sends a SYN packet with an initial sequence number. The server responds with a "
    "SYN-ACK packet, acknowledging the client's sequence number and providing its own. Finally, the "
    "client sends an ACK packet to acknowledge the server's sequence number, at which point the "
    "connection is established and data transfer can begin."
)

TEST_CASES += [
    {
        "domain": "TCP Handshake",
        "name": "17. Similar (paraphrase)",
        "user_answer": (
            "To set up a reliable TCP link, the client transmits a SYN segment containing its starting "
            "sequence number. The server replies with a SYN-ACK that both acknowledges the client's number "
            "and advertises its own initial sequence number. The client then returns an ACK to confirm the "
            "server's number, and once this exchange finishes, the two hosts may start exchanging payload data."
        ),
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "TCP Handshake",
        "name": "18. More detailed (correct extra info)",
        "user_answer": (
            "The TCP three-way handshake establishes a reliable connection between a client and a server. "
            "The client first sends a SYN packet with an initial sequence number. The server responds with a "
            "SYN-ACK packet, acknowledging the client's sequence number and providing its own. Finally, the "
            "client sends an ACK packet to acknowledge the server's sequence number, at which point the "
            "connection is established and data transfer can begin. During this exchange, both sides also "
            "negotiate window scaling options and maximum segment size (MSS) to optimize throughput. The "
            "initial sequence numbers are randomly generated to protect against sequence number prediction "
            "attacks, and the connection enters the ESTABLISHED state only after the final ACK is successfully "
            "transmitted."
        ),
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "TCP Handshake",
        "name": "19. Contradicts reference",
        "user_answer": (
            "The TCP three-way handshake begins with the server sending a FIN packet to the client to announce "
            "its intention to close the connection. The client replies with a RST packet to reset the link "
            "immediately. No sequence numbers are exchanged during this process, and the connection is marked "
            "as established as soon as the server receives the RST, without any further acknowledgment from the client."
        ),
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "TCP Handshake",
        "name": "20. Less detailed (vague but not wrong)",
        "user_answer": (
            "TCP uses a handshake where the client and server exchange three messages to start a connection."
        ),
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (SYN, SYN-ACK, ACK, sequence numbers).",
    },
    {
        "domain": "TCP Handshake",
        "name": "21. Cutoff / incomplete sentence",
        "user_answer": (
            "The TCP three-way handshake establishes a reliable connection between a client and a server. "
            "The client first sends a SYN packet with an initial sequence number. The server responds with a "
            "SYN-ACK packet, acknowledging the client's sequence number and providing its own. Finally, the "
            "client sends an"
        ),
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "TCP Handshake",
        "name": "22. Totally unrelated",
        "user_answer": (
            "The French Revolution began in 1789 with the storming of the Bastille, driven by widespread "
            "social inequality, fiscal crisis, and Enlightenment ideals. It led to the abolition of the monarchy, "
            "the Reign of Terror under Robespierre, and eventually the rise of Napoleon Bonaparte, fundamentally "
            "reshaping the political landscape of Europe."
        ),
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

# ── Domain 4: Gradient Descent ──────────────────────────────────────────────
REF_GRAD = (
    "Gradient descent is an optimization algorithm used to minimize a neural network's loss function. "
    "It works by computing the gradient of the loss with respect to each weight parameter using backpropagation, "
    "then updating the weights in the opposite direction of the gradient. The size of each update is controlled "
    "by a learning rate hyperparameter. This process is repeated iteratively over batches of training data until "
    "the loss converges to a local minimum."
)

TEST_CASES += [
    {
        "domain": "Gradient Descent",
        "name": "23. Similar (paraphrase)",
        "user_answer": (
            "In neural network training, gradient descent serves as the core optimizer for reducing loss. "
            "The algorithm calculates the partial derivative of the loss function for every trainable weight via "
            "backpropagation, then shifts each weight in the negative gradient direction. A learning rate dictates "
            "the step magnitude, and the entire procedure cycles repeatedly across mini-batches until the loss "
            "stabilizes near a local optimum."
        ),
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "Gradient Descent",
        "name": "24. More detailed (correct extra info)",
        "user_answer": (
            "Gradient descent is an optimization algorithm used to minimize a neural network's loss function. "
            "It works by computing the gradient of the loss with respect to each weight parameter using backpropagation, "
            "then updating the weights in the opposite direction of the gradient. The size of each update is controlled "
            "by a learning rate hyperparameter. This process is repeated iteratively over batches of training data until "
            "the loss converges to a local minimum. Modern implementations typically use stochastic gradient descent (SGD) "
            "with momentum, which accumulates a velocity vector to dampen oscillations in ravines and accelerate convergence "
            "in consistent directions. Adaptive variants such as Adam additionally maintain per-parameter estimates of the "
            "first and second moments of the gradient, allowing the effective learning rate to adjust automatically for each weight."
        ),
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Gradient Descent",
        "name": "25. Contradicts reference",
        "user_answer": (
            "Gradient descent maximizes the loss function by adding the gradient directly to each weight after every "
            "forward pass. Backpropagation is not used; instead, random perturbations are applied to weights, and only "
            "perturbations that increase the loss are kept. The learning rate determines how many layers are frozen on each "
            "step, and training continues until the loss reaches its global maximum."
        ),
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Gradient Descent",
        "name": "26. Less detailed (vague but not wrong)",
        "user_answer": (
            "Gradient descent updates neural network weights using gradients to reduce loss over time."
        ),
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (backprop, learning rate, local minimum, batches).",
    },
    {
        "domain": "Gradient Descent",
        "name": "27. Cutoff / incomplete sentence",
        "user_answer": (
            "Gradient descent is an optimization algorithm used to minimize a neural network's loss function. "
            "It works by computing the gradient of the loss with respect to each weight parameter using backpropagation, "
            "then updating the weights in the opposite direction of the gradient. The size of each update is controlled "
            "by a learning rate hyperparameter. This process is repeated iteratively over batches of training data until "
            "the loss converges to a"
        ),
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Gradient Descent",
        "name": "28. Totally unrelated",
        "user_answer": (
            "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, "
            "Australia. It spans over 2,300 kilometers and comprises thousands of individual reefs and islands. It is home "
            "to an immense diversity of marine life, including over 1,500 fish species, and is visible from outer space."
        ),
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

# ── Domain 5: Raft Consensus (Leader Election) ──────────────────────────────
REF_RAFT = (
    "In the Raft consensus protocol, leader election begins when a follower receives no valid heartbeat "
    "from the current leader within its election timeout period. The follower then increments its current term, "
    "transitions to candidate state, and votes for itself. It sends RequestVote RPCs to all other servers. "
    "A candidate wins the election if it receives votes from a majority of servers in the cluster for that term. "
    "Once elected, the new leader sends heartbeat messages to establish authority and prevent new elections."
)

TEST_CASES += [
    {
        "domain": "Raft Leader Election",
        "name": "29. Similar (paraphrase)",
        "user_answer": (
            "Within the Raft protocol, a follower that fails to detect a heartbeat from the leader before its election "
            "timer expires becomes a candidate. It advances its term counter, casts a vote for itself, and dispatches "
            "RequestVote requests to every other node. If the candidate gathers a strict majority of affirmative votes "
            "for its term, it assumes leadership and immediately begins broadcasting heartbeats to suppress further elections."
        ),
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "30. More detailed (correct extra info)",
        "user_answer": (
            "In the Raft consensus protocol, leader election begins when a follower receives no valid heartbeat from the "
            "current leader within its election timeout period. The follower then increments its current term, transitions "
            "to candidate state, and votes for itself. It sends RequestVote RPCs to all other servers. A candidate wins the "
            "election if it receives votes from a majority of servers in the cluster for that term. Once elected, the new "
            "leader sends heartbeat messages to establish authority and prevent new elections. To reduce split votes, each "
            "server uses a randomized election timeout between 150 and 300 milliseconds. Additionally, Raft's voting logic "
            "includes a safety check: a voter denies its vote if the candidate's log is less up-to-date than its own, ensuring "
            "that a newly elected leader always contains all committed entries."
        ),
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "31. Contradicts reference",
        "user_answer": (
            "In Raft, leader election is triggered when the current leader voluntarily sends a resignation broadcast to all "
            "followers. Followers respond with a LeaderSurrender acknowledgment and immediately enter candidate state simultaneously. "
            "The first candidate to transmit a heartbeat becomes the leader automatically, regardless of how many votes it has received. "
            "There is no concept of a majority; the network latency alone decides the winner."
        ),
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "32. Less detailed (vague but not wrong)",
        "user_answer": (
            "Raft picks a new leader when the old one stops responding. A server votes for itself and asks others for votes. "
            "Whoever gets the most votes becomes leader."
        ),
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (majority, RequestVote RPC, heartbeats, term increment).",
    },
    {
        "domain": "Raft Leader Election",
        "name": "33. Cutoff / incomplete sentence",
        "user_answer": (
            "In the Raft consensus protocol, leader election begins when a follower receives no valid heartbeat from the "
            "current leader within its election timeout period. The follower then increments its current term, transitions "
            "to candidate state, and votes for itself. It sends RequestVote RPCs to all other servers. A candidate wins the "
            "election if it receives votes from a majority of servers in the cluster for that"
        ),
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "34. Totally unrelated",
        "user_answer": (
            "Impressionism was a 19th-century art movement characterized by small, thin brush strokes, an emphasis on accurate "
            "depiction of light in its changing qualities, and ordinary subject matter. Artists such as Claude Monet and "
            "Pierre-Auguste Renoir often painted outdoors to capture the transient effects of sunlight and atmosphere on "
            "landscapes and daily life."
        ),
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════


def run_tests():
    print("=" * 90)
    print("FIXED PIPELINE TEST SUITE  (4 non-SLM fixes applied)")
    print("=" * 90)
    print("Fixes: 1) Bidirectional lexical   2) CE order [ref, user]")
    print("       3) Length-ratio guard      4) No lexical shortcut")
    print()

    engine = FixedLogicEngine()

    # Pre-load models so timing is fair
    print("Loading models (first run may download)…")
    engine._get_bi_encoder()
    engine._get_ce_model()
    print("Models ready.\n")

    results = []
    domain_map = {
        "Self-Attention": REF_ATTENTION,
        "B-Tree Index": REF_BTREE,
        "TCP Handshake": REF_TCP,
        "Gradient Descent": REF_GRAD,
        "Raft Leader Election": REF_RAFT,
    }

    for case in TEST_CASES:
        ref = domain_map[case["domain"]]
        q = QuestionObj(
            id=f"test_{case['name'].split('.')[0]}",
            text=f"Explain {case['domain']}.",
            answer=ref,
            type="open",
            difficulty="Medium",
            tags=[case["domain"].lower().replace(" ", "-")],
        )

        print(f"\n{'─' * 90}")
        print(f"CASE {case['name']}  |  Domain: {case['domain']}")
        print(f"Expected: {case['expected']}")
        preview = (
            case["user_answer"][:180] + "…"
            if len(case["user_answer"]) > 180
            else case["user_answer"]
        )
        print(f"User: {preview}")

        tr = engine.analyze_response(q, case["user_answer"])

        print(f"  Final sim : {tr.similarity:.4f}  ({tr.similarity * 100:.1f}%)")
        print(f"  Bi-enc    : {tr.bi_score:.4f}  ({tr.bi_score * 100:.1f}%)")
        print(f"  Lexical   : {tr.lexical_score:.4f}  ({tr.lexical_score * 100:.1f}%)")
        print(f"  Len ratio : {tr.length_ratio:.3f}")
        print(f"  Grader    : {tr.grader}")
        if tr.nli:
            print(
                f"  NLI       : {tr.nli.verdict.upper()}  "
                f"(E={tr.nli.p_entailment:.3f}  N={tr.nli.p_neutral:.3f}  C={tr.nli.p_contradiction:.3f})"
            )
        else:
            print(f"  NLI       : (not run)")
        print(f"  Feedback  : {tr.feedback}")

        results.append(
            {
                "case": case["name"],
                "domain": case["domain"],
                "expected": case["expected"],
                "final_score": round(tr.similarity, 4),
                "bi_score": round(tr.bi_score, 4),
                "lexical_score": round(tr.lexical_score, 4),
                "length_ratio": round(tr.length_ratio, 4),
                "grader": tr.grader,
                "nli_verdict": tr.nli.verdict if tr.nli else None,
                "nli_entailment": round(tr.nli.p_entailment, 4) if tr.nli else None,
                "nli_neutral": round(tr.nli.p_neutral, 4) if tr.nli else None,
                "nli_contradiction": round(tr.nli.p_contradiction, 4)
                if tr.nli
                else None,
                "feedback": tr.feedback,
            }
        )

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Case':<45} {'Final':>7} {'Bi-enc':>7} {'Lex':>7} {'Ratio':>6} {'Grader':>14} {'NLI':>10}"
    )
    print("-" * 90)
    for r in results:
        nli_str = r["nli_verdict"].upper()[:3] if r["nli_verdict"] else "N/A"
        print(
            f"{r['case']:<45} "
            f"{r['final_score']:>6.2f} "
            f"{r['bi_score']:>6.2f} "
            f"{r['lexical_score']:>6.2f} "
            f"{r['length_ratio']:>5.2f} "
            f"{r['grader']:>14} "
            f"{nli_str:>10}"
        )

    # Save
    out_path = Path("test_results_fixed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path.resolve()}")

    return results


if __name__ == "__main__":
    run_tests()
