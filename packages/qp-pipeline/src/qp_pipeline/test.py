"""
qa_chunk_evaluator_logged.py

Strict Q/A evaluator with:
- Explicit chunk
- Hard gates
- Deterministic scoring
- Full logging visibility

Python 3.9+
"""

import logging
import re
from typing import Dict, List

# =========================
# 0. LOGGING SETUP
# =========================

LOG_LEVEL = logging.DEBUG  # change to INFO to reduce verbosity

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(name)s:%(message)s")

logger = logging.getLogger("QAChunkEvaluator")

# =========================
# 1. SOURCE CHUNK
# =========================

CHUNK = """
U²-Net is a convolutional neural network designed for salient object detection.
It uses a nested U-structure (U-in-U) to capture multi-scale features efficiently.
U²-Net is widely used for background removal because it can preserve fine details
such as hair and object boundaries while maintaining low computational cost.
"""

logger.info("Loaded source chunk (%d characters)", len(CHUNK))

# =========================
# 2. Q/A PAIRS (REPLACE ME)
# =========================

QA_PAIRS: List[Dict[str, str]] = [
    {
        "question": "What is U²-Net primarily designed for?",
        "answer": "U²-Net is designed for salient object detection.",
    },
    {
        "question": "Why is U²-Net effective for background removal?",
        "answer": "It preserves fine details like hair and object boundaries while remaining computationally efficient.",
    },
    {
        "question": "Does U²-Net use transformers internally?",
        "answer": "Yes, it is based on transformer attention layers.",
    },
]

logger.info("Loaded %d Q/A pair(s)", len(QA_PAIRS))

# =========================
# 3. TEXT UTILITIES
# =========================


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())
    if not a_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens) / len(a_tokens)
    logger.debug("Token overlap: %.3f | A='%s'", overlap, a)
    return overlap


# =========================
# 4. HARD GATES
# =========================

QUESTION_GATE = 0.15
ANSWER_GATE = 0.20


def gate_answerable(question: str, chunk: str) -> bool:
    score = token_overlap(question, chunk)
    logger.debug("Question gate score: %.3f (threshold %.2f)", score, QUESTION_GATE)
    return score >= QUESTION_GATE


def gate_grounded(answer: str, chunk: str) -> bool:
    score = token_overlap(answer, chunk)
    logger.debug("Answer grounding score: %.3f (threshold %.2f)", score, ANSWER_GATE)
    return score >= ANSWER_GATE


# =========================
# 5. SCORING
# =========================


def score_question(question: str, chunk: str) -> float:
    overlap = token_overlap(question, chunk)
    length_score = min(len(question.split()) / 12, 1.0)
    score = (0.6 * overlap + 0.4 * length_score) * 10
    logger.debug(
        "Question score: overlap=%.3f length=%.3f final=%.2f",
        overlap,
        length_score,
        score,
    )
    return round(score, 2)


def score_answer(answer: str, question: str, chunk: str) -> float:
    grounding = token_overlap(answer, chunk)
    relevance = token_overlap(answer, question)
    score = (0.6 * grounding + 0.4 * relevance) * 10
    logger.debug(
        "Answer score: grounding=%.3f relevance=%.3f final=%.2f",
        grounding,
        relevance,
        score,
    )
    return round(score, 2)


# =========================
# 6. EVALUATION LOOP
# =========================


def evaluate(chunk: str, qa_pairs: List[Dict[str, str]]):
    logger.info("Starting Q/A evaluation")
    results = []
    total_score = 0.0
    passed_count = 0

    for idx, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        a = qa["answer"]

        logger.info("Evaluating Q/A pair #%d", idx)
        logger.debug("Question: %s", q)
        logger.debug("Answer: %s", a)

        # ---- QUESTION GATE ----
        if not gate_answerable(q, chunk):
            logger.warning("FAIL Q%d: Question not answerable from chunk", idx)
            results.append((idx, "FAIL", "question_not_answerable"))
            continue

        # ---- ANSWER GATE ----
        if not gate_grounded(a, chunk):
            logger.warning("FAIL Q%d: Answer not grounded in chunk", idx)
            results.append((idx, "FAIL", "answer_not_grounded"))
            continue

        q_score = score_question(q, chunk)
        a_score = score_answer(a, q, chunk)
        avg = round((q_score + a_score) / 2, 2)

        logger.info(
            "PASS Q%d: q_score=%.2f a_score=%.2f avg=%.2f", idx, q_score, a_score, avg
        )

        total_score += avg
        passed_count += 1
        results.append((idx, avg, q_score, a_score))

    average = round(total_score / passed_count, 2) if passed_count else 0.0

    logger.info(
        "Evaluation complete: %d passed / %d total | average=%.2f",
        passed_count,
        len(qa_pairs),
        average,
    )

    return results, average


# =========================
# 7. MAIN
# =========================

if __name__ == "__main__":
    logger.info("🐍 Starting Q/A Chunk Evaluation Pipeline")
    results, avg = evaluate(CHUNK, QA_PAIRS)

    print("\n=== RESULTS ===")
    for r in results:
        print(r)

    print("\n=== AVERAGE SCORE ===")
    print(avg)

    if avg < 7.0:
        logger.error("❌ BELOW QUALITY BAR — pipeline should be rejected")
    else:
        logger.info("✅ PASSES QUALITY BAR")
