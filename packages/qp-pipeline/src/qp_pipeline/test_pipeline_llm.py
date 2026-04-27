#!/usr/bin/env python3
"""
test_pipeline_llm_reasoning.py — LLM-based grading with a reasoning model (Gemma-4B-E2B).

Leverages the model's reasoning capability by asking it to think step-by-step
before producing the final JSON score. Handles reasoning traces in the output.

Run:  python test_pipeline_llm_reasoning.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from json_repair import repair_json

    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

# ── Config ───────────────────────────────────────────────────────────────────
LLM_API_URL = "http://localhost:8080/v1"
LLM_API_KEY = "no-key"
LLM_MODEL = "gemma-4-E2B-it"  # Reasoning model
LLM_TEMPERATURE = 0.3  # Slightly higher for reasoning models
LLM_MAX_TOKENS = 512  # More tokens for reasoning trace + JSON

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("LLMReasoningPipeline")


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA — 34 cases across 5 domains
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = []

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
        "user_answer": "Transformers use self-attention to calculate a weighted combination of every input token at each position. The mechanism derives Query, Key, and Value vectors from the embeddings. It computes attention scores via the dot product of Query and Key, applies scaling and softmax, then uses these weights to aggregate the Value vectors into the output.",
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "Self-Attention",
        "name": "2. More detailed (correct extra info)",
        "user_answer": "Self-attention in transformers computes a weighted sum of all input tokens for each position using Query, Key, and Value matrices derived from input embeddings. The dot product of Query and Key determines raw attention weights, which are scaled by the square root of the head dimension to prevent vanishing gradients, then passed through softmax. These normalized weights are multiplied by the Value matrix to produce the final output. In multi-head attention, this process runs in parallel across several heads with different learned projections, allowing the model to attend to information from different representation subspaces at different positions.",
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Self-Attention",
        "name": "3. Contradicts reference",
        "user_answer": "Self-attention ignores all other tokens and only processes the current token independently. It does not use Query, Key, or Value matrices; instead, it applies a fixed convolutional filter to each token separately. The weights are determined by a pre-trained lookup table rather than being computed dynamically via dot products.",
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Self-Attention",
        "name": "4. Less detailed (vague but not wrong)",
        "user_answer": "Self-attention calculates weights for tokens using dot products and then combines them to form the output.",
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (Q/K/V, softmax, scaling).",
    },
    {
        "domain": "Self-Attention",
        "name": "5. Cutoff / incomplete sentence",
        "user_answer": "Self-attention in transformers computes a weighted sum of all input tokens for each position. It uses Query, Key, and Value matrices derived from the input embeddings. The dot product of Query and Key determines the attention weights, which are then scaled and passed through a soft",
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Self-Attention",
        "name": "6. Totally unrelated",
        "user_answer": "The water cycle involves evaporation, condensation, and precipitation. When the sun heats water in oceans and rivers, it rises into the atmosphere as vapor, cools down to form clouds, and eventually falls back to Earth as rain or snow.",
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
    {
        "domain": "Self-Attention",
        "name": "7. Hallucinated / fabricated details",
        "user_answer": "Self-attention in transformers computes a weighted sum of all input tokens for each position. It uses Query, Key, and Value matrices derived from the input embeddings. The dot product of Query and Key determines the attention weights, which are then scaled by the inverse cosine of the embedding dimension and passed through a ReLU activation instead of softmax. These weights are multiplied by the Value matrix and then fed into a recurrent LSTM cell to produce the final output representation for each token.",
        "expected": "Low-to-moderate score (2–5). Starts correctly but introduces plausible-sounding false details.",
    },
    {
        "domain": "Self-Attention",
        "name": "8. Correct but different terminology",
        "user_answer": "In transformer networks, intra-attention mechanisms generate a convex combination of source representations per time-step. The system projects hidden states into three distinct subspaces—denoted as Q, K, and V—via learned affine transformations. Affinity scores are obtained via inner-product similarity between the query and key projections, followed by temperature-scaled normalization via the Boltzmann (softmax) operator. The resulting convex coefficients are applied to the value projections to yield context-aware hidden states.",
        "expected": "High score (8–9). Same concepts, different jargon.",
    },
    {
        "domain": "Self-Attention",
        "name": "9. Mostly correct with one critical error",
        "user_answer": "Self-attention in transformers computes a weighted sum of all input tokens for each position. It uses Query, Key, and Value matrices derived from the input embeddings. The dot product of Query and Key determines the attention weights, which are then scaled and passed through a softmax function. These weights are added to the Value matrix to produce the final output representation for each token.",
        "expected": "Moderate-to-low score (3–6). One word changed ('added' vs 'multiplied') makes the mechanism wrong.",
    },
    {
        "domain": "Self-Attention",
        "name": "10. Verbatim copy",
        "user_answer": REF_ATTENTION,
        "expected": "Very high score (9.5–10). Exact match.",
    },
]

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
        "user_answer": "Databases use B-tree indexes to arrange records in a balanced hierarchical structure. Every node stores several keys along with references to its children. Because the tree remains flat, disk reads are reduced, and search, insert, and delete operations all run in logarithmic time relative to the number of records. Keys within each node are maintained in order, and nodes split once they exceed a configured capacity threshold.",
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "B-Tree Index",
        "name": "12. More detailed (correct extra info)",
        "user_answer": "A B-tree index in a database organizes data in a self-balancing tree structure where each node can contain multiple keys and pointers to child nodes. This design minimizes disk I/O by keeping the tree shallow, ensuring that lookups, insertions, and deletions all complete in O(log n) time. The nodes are kept sorted, and a fill factor governs how full each node can become before it splits. In practice, most database engines use a B+ tree variant where only leaf nodes store actual record pointers or row IDs, while internal nodes serve purely as navigation keys. Leaf nodes are also linked together as a doubly-linked list, enabling efficient range scans and ordered traversal without revisiting parent nodes.",
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "B-Tree Index",
        "name": "13. Contradicts reference",
        "user_answer": "A B-tree index stores every record in a single root node without any child pointers, making it essentially a flat array. Lookups require a linear scan from the first element to the last, resulting in O(n) worst-case performance. The structure is intentionally unsorted to maximize write throughput, and nodes never split; instead, overflow data is written to a separate append-only log file.",
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "B-Tree Index",
        "name": "14. Less detailed (vague but not wrong)",
        "user_answer": "A B-tree is a tree structure used by databases to find data quickly. It keeps things sorted and balanced.",
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (child pointers, fill factor, O(log n)).",
    },
    {
        "domain": "B-Tree Index",
        "name": "15. Cutoff / incomplete sentence",
        "user_answer": "A B-tree index in a database organizes data in a self-balancing tree structure where each node can contain multiple keys and pointers to child nodes. This design minimizes disk I/O by keeping the tree shallow, ensuring that lookups, insertions, and deletions all complete in O(log n) time. The nodes are kept sorted, and a fill factor governs how full each node can become before it",
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "B-Tree Index",
        "name": "16. Totally unrelated",
        "user_answer": "Photosynthesis occurs in the chloroplasts of plant cells, where chlorophyll absorbs sunlight to convert carbon dioxide and water into glucose and oxygen. This process is divided into the light-dependent reactions and the Calvin cycle, which together provide the energy and organic compounds necessary for plant growth.",
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

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
        "user_answer": "To set up a reliable TCP link, the client transmits a SYN segment containing its starting sequence number. The server replies with a SYN-ACK that both acknowledges the client's number and advertises its own initial sequence number. The client then returns an ACK to confirm the server's number, and once this exchange finishes, the two hosts may start exchanging payload data.",
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "TCP Handshake",
        "name": "18. More detailed (correct extra info)",
        "user_answer": "The TCP three-way handshake establishes a reliable connection between a client and a server. The client first sends a SYN packet with an initial sequence number. The server responds with a SYN-ACK packet, acknowledging the client's sequence number and providing its own. Finally, the client sends an ACK packet to acknowledge the server's sequence number, at which point the connection is established and data transfer can begin. During this exchange, both sides also negotiate window scaling options and maximum segment size (MSS) to optimize throughput. The initial sequence numbers are randomly generated to protect against sequence number prediction attacks, and the connection enters the ESTABLISHED state only after the final ACK is successfully transmitted.",
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "TCP Handshake",
        "name": "19. Contradicts reference",
        "user_answer": "The TCP three-way handshake begins with the server sending a FIN packet to the client to announce its intention to close the connection. The client replies with a RST packet to reset the link immediately. No sequence numbers are exchanged during this process, and the connection is marked as established as soon as the server receives the RST, without any further acknowledgment from the client.",
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "TCP Handshake",
        "name": "20. Less detailed (vague but not wrong)",
        "user_answer": "TCP uses a handshake where the client and server exchange three messages to start a connection.",
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (SYN, SYN-ACK, ACK, sequence numbers).",
    },
    {
        "domain": "TCP Handshake",
        "name": "21. Cutoff / incomplete sentence",
        "user_answer": "The TCP three-way handshake establishes a reliable connection between a client and a server. The client first sends a SYN packet with an initial sequence number. The server responds with a SYN-ACK packet, acknowledging the client's sequence number and providing its own. Finally, the client sends an",
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "TCP Handshake",
        "name": "22. Totally unrelated",
        "user_answer": "The French Revolution began in 1789 with the storming of the Bastille, driven by widespread social inequality, fiscal crisis, and Enlightenment ideals. It led to the abolition of the monarchy, the Reign of Terror under Robespierre, and eventually the rise of Napoleon Bonaparte, fundamentally reshaping the political landscape of Europe.",
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

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
        "user_answer": "In neural network training, gradient descent serves as the core optimizer for reducing loss. The algorithm calculates the partial derivative of the loss function for every trainable weight via backpropagation, then shifts each weight in the negative gradient direction. A learning rate dictates the step magnitude, and the entire procedure cycles repeatedly across mini-batches until the loss stabilizes near a local optimum.",
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "Gradient Descent",
        "name": "24. More detailed (correct extra info)",
        "user_answer": "Gradient descent is an optimization algorithm used to minimize a neural network's loss function. It works by computing the gradient of the loss with respect to each weight parameter using backpropagation, then updating the weights in the opposite direction of the gradient. The size of each update is controlled by a learning rate hyperparameter. This process is repeated iteratively over batches of training data until the loss converges to a local minimum. Modern implementations typically use stochastic gradient descent (SGD) with momentum, which accumulates a velocity vector to dampen oscillations in ravines and accelerate convergence in consistent directions. Adaptive variants such as Adam additionally maintain per-parameter estimates of the first and second moments of the gradient, allowing the effective learning rate to adjust automatically for each weight.",
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Gradient Descent",
        "name": "25. Contradicts reference",
        "user_answer": "Gradient descent maximizes the loss function by adding the gradient directly to each weight after every forward pass. Backpropagation is not used; instead, random perturbations are applied to weights, and only perturbations that increase the loss are kept. The learning rate determines how many layers are frozen on each step, and training continues until the loss reaches its global maximum.",
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Gradient Descent",
        "name": "26. Less detailed (vague but not wrong)",
        "user_answer": "Gradient descent updates neural network weights using gradients to reduce loss over time.",
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (backprop, learning rate, local minimum, batches).",
    },
    {
        "domain": "Gradient Descent",
        "name": "27. Cutoff / incomplete sentence",
        "user_answer": "Gradient descent is an optimization algorithm used to minimize a neural network's loss function. It works by computing the gradient of the loss with respect to each weight parameter using backpropagation, then updating the weights in the opposite direction of the gradient. The size of each update is controlled by a learning rate hyperparameter. This process is repeated iteratively over batches of training data until the loss converges to a",
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Gradient Descent",
        "name": "28. Totally unrelated",
        "user_answer": "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia. It spans over 2,300 kilometers and comprises thousands of individual reefs and islands. It is home to an immense diversity of marine life, including over 1,500 fish species, and is visible from outer space.",
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

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
        "user_answer": "Within the Raft protocol, a follower that fails to detect a heartbeat from the leader before its election timer expires becomes a candidate. It advances its term counter, casts a vote for itself, and dispatches RequestVote requests to every other node. If the candidate gathers a strict majority of affirmative votes for its term, it assumes leadership and immediately begins broadcasting heartbeats to suppress further elections.",
        "expected": "High score (8.5–9.5). Conceptually identical, different wording.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "30. More detailed (correct extra info)",
        "user_answer": "In the Raft consensus protocol, leader election begins when a follower receives no valid heartbeat from the current leader within its election timeout period. The follower then increments its current term, transitions to candidate state, and votes for itself. It sends RequestVote RPCs to all other servers. A candidate wins the election if it receives votes from a majority of servers in the cluster for that term. Once elected, the new leader sends heartbeat messages to establish authority and prevent new elections. To reduce split votes, each server uses a randomized election timeout between 150 and 300 milliseconds. Additionally, Raft's voting logic includes a safety check: a voter denies its vote if the candidate's log is less up-to-date than its own, ensuring that a newly elected leader always contains all committed entries.",
        "expected": "High score (9–10). Adds accurate, relevant depth without contradicting.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "31. Contradicts reference",
        "user_answer": "In Raft, leader election is triggered when the current leader voluntarily sends a resignation broadcast to all followers. Followers respond with a LeaderSurrender acknowledgment and immediately enter candidate state simultaneously. The first candidate to transmit a heartbeat becomes the leader automatically, regardless of how many votes it has received. There is no concept of a majority; the network latency alone decides the winner.",
        "expected": "Very low score (0–2). Every core claim is factually opposite.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "32. Less detailed (vague but not wrong)",
        "user_answer": "Raft picks a new leader when the old one stops responding. A server votes for itself and asks others for votes. Whoever gets the most votes becomes leader.",
        "expected": "Moderate score (4–6). Not wrong, but omits critical components (majority, RequestVote RPC, heartbeats, term increment).",
    },
    {
        "domain": "Raft Leader Election",
        "name": "33. Cutoff / incomplete sentence",
        "user_answer": "In the Raft consensus protocol, leader election begins when a follower receives no valid heartbeat from the current leader within its election timeout period. The follower then increments its current term, transitions to candidate state, and votes for itself. It sends RequestVote RPCs to all other servers. A candidate wins the election if it receives votes from a majority of servers in the cluster for that",
        "expected": "Low score (2–4). Literally unfinished; cannot be evaluated as complete.",
    },
    {
        "domain": "Raft Leader Election",
        "name": "34. Totally unrelated",
        "user_answer": "Impressionism was a 19th-century art movement characterized by small, thin brush strokes, an emphasis on accurate depiction of light in its changing qualities, and ordinary subject matter. Artists such as Claude Monet and Pierre-Auguste Renoir often painted outdoors to capture the transient effects of sunlight and atmosphere on landscapes and daily life.",
        "expected": "Near zero (0–1). Off-topic; no semantic overlap with the reference.",
    },
]

DOMAIN_REFS = {
    "Self-Attention": REF_ATTENTION,
    "B-Tree Index": REF_BTREE,
    "TCP Handshake": REF_TCP,
    "Gradient Descent": REF_GRAD,
    "Raft Leader Election": REF_RAFT,
}


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST JSON EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════


def robust_json_extract(text: str) -> Dict[str, Any]:
    """
    Extract a JSON dict from text that may contain reasoning traces, markdown,
    or other noise. Tries multiple strategies from most to least reliable.
    """
    if not text or not text.strip():
        return {}

    cleaned = text.strip()

    # Strategy 0: Strip known reasoning tags
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    # Strategy 1: Find the LAST { ... } pair (reasoning models put JSON at the end)
    # Use greedy match from the last opening brace to the last closing brace
    last_open = cleaned.rfind("{")
    last_close = cleaned.rfind("}")
    if last_open != -1 and last_close != -1 and last_close > last_open:
        candidate = cleaned[last_open : last_close + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "score" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find ALL { ... } substrings and try each (longest first)
    candidates = re.findall(r"\{[\s\S]*?\}", cleaned)
    candidates.sort(key=len, reverse=True)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and "score" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Strategy 3: Try to repair with json_repair
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict) and "score" in repaired:
                return repaired
            # json_repair might return a list of dicts
            if isinstance(repaired, list):
                for item in repaired:
                    if isinstance(item, dict) and "score" in item:
                        return item
        except Exception:
            pass

    # Strategy 4: Try the entire cleaned text as JSON
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "score" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 5: Try to extract key-value pairs manually as last resort
    score_match = re.search(r'"score"\s*:\s*(\d+)', cleaned)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', cleaned)
    if score_match:
        return {
            "score": int(score_match.group(1)),
            "reason": reason_match.group(1) if reason_match else "extracted via regex",
        }

    logger.error(f"JSON Parse Error — raw text: {text[:400]}...")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT — REASONING MODEL
# ══════════════════════════════════════════════════════════════════════════════


class ReasoningLLMGrader:
    def __init__(
        self,
        base_url: str = LLM_API_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def grade(self, reference: str, user_answer: str) -> Dict[str, Any]:
        sys_prompt = (
            "You are an expert technical interviewer grading student answers. "
            "Compare the STUDENT ANSWER to the REFERENCE ANSWER and assign a score 0–10. "
            "Think carefully, then output ONLY a valid JSON object. No text after the JSON."
        )

        user_prompt = f"""Score the student answer against the reference answer.

Think step by step:
1. Is the student answer related to the reference topic at all?
2. Is the student answer a complete sentence, or does it cut off mid-thought?
3. Does the student answer contradict the reference in any way?
4. Does the student answer capture the core mechanisms described in the reference?
5. Is the student answer vague, or does it include specific details?
6. Does the student answer use different but correct terminology?
7. Does the student answer add extra correct information?

After your reasoning, provide ONLY the JSON object.

### SCORING RUBRIC (0–10):
- **10**: Perfect. Correct, complete, and well-explained. May include accurate extra detail.
- **8–9**: Correct and complete, but slightly less precise or missing minor details.
- **6–7**: Partially correct; captures the main idea but omits important mechanisms or nuances.
- **4–5**: Mostly incorrect or severely incomplete; shows fundamental misunderstanding or misses most key points.
- **2–3**: Entirely wrong, contradictory, or completely unrelated.
- **0–1**: No meaningful content, or the answer is an incomplete sentence / cutoff mid-thought.

### CRITICAL RULES:
1. If the student answer **contradicts** the reference at any point, score ≤ 3 regardless of fluency.
2. If the student answer is an **incomplete sentence** or cuts off mid-thought, score ≤ 2.
3. If the student answer is **totally unrelated** to the reference, score 0–1.
4. If the student answer is **vague** but not wrong, score in the 4–6 range.
5. If the student answer uses **different but correct terminology**, do not penalize.
6. Reward **additional correct detail** that does not contradict the reference.

### REFERENCE ANSWER:
{reference}

### STUDENT ANSWER:
{user_answer}

Respond with ONLY this JSON format (absolutely no text before or after):
{{"score": <integer 0-10>, "reason": "<one-sentence justification>"}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                extra_body={
                    "top_k": 50,
                    "repetition_penalty": 1.05,
                },
            )
            content = response.choices[0].message.content
            parsed = robust_json_extract(content)

            score = parsed.get("score")
            reason = parsed.get("reason", "")

            if score is None:
                logger.warning("LLM returned no score — defaulting to 0")
                score = 0
            try:
                score = int(score)
                score = max(0, min(10, score))
            except (ValueError, TypeError):
                logger.warning(
                    f"LLM returned non-integer score: {score!r} — defaulting to 0"
                )
                score = 0

            return {"score": score, "reason": reason, "raw": content}

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"score": 0, "reason": f"LLM error: {e}", "raw": ""}


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════


def run_pipeline():
    print("=" * 90)
    print("LLM-BASED ANSWER GRADING — REASONING MODEL")
    print(f"Model: {LLM_MODEL}  |  Endpoint: {LLM_API_URL}")
    print("=" * 90)
    print()

    grader = ReasoningLLMGrader()

    # Quick health check
    print("Checking LLM health...")
    try:
        test_resp = grader.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        print(f"  LLM responded: {test_resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  ERROR: Cannot reach LLM at {LLM_API_URL} — {e}")
        sys.exit(1)
    print()

    results = []
    total_time = 0.0

    for i, case in enumerate(TEST_CASES, 1):
        ref = DOMAIN_REFS[case["domain"]]
        print(f"\n{'─' * 90}")
        print(f"[{i}/{len(TEST_CASES)}] {case['name']}  |  Domain: {case['domain']}")
        print(f"Expected: {case['expected']}")
        preview = (
            case["user_answer"][:160] + "…"
            if len(case["user_answer"]) > 160
            else case["user_answer"]
        )
        print(f"User: {preview}")

        t0 = time.perf_counter()
        grade = grader.grade(ref, case["user_answer"])
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        print(f"  LLM score : {grade['score']}/10")
        print(f"  Reason    : {grade['reason']}")
        print(f"  Latency   : {elapsed:.2f}s")

        results.append(
            {
                "case": case["name"],
                "domain": case["domain"],
                "expected": case["expected"],
                "user_answer": case["user_answer"],
                "llm_score": grade["score"],
                "llm_reason": grade["reason"],
                "llm_raw": grade["raw"],
                "latency_sec": round(elapsed, 3),
            }
        )

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Case':<45} {'Score':>7} {'Expected':>25} {'Time':>8}")
    print("-" * 90)
    for r in results:
        expected_short = r["expected"].split(".")[0]
        print(
            f"{r['case']:<45} {r['llm_score']:>5}/10 {expected_short:>25} {r['latency_sec']:>7.2f}s"
        )

    avg_time = total_time / len(results) if results else 0
    print(f"\nTotal time: {total_time:.1f}s  |  Avg per case: {avg_time:.2f}s")

    # Save
    out_path = Path("test_results_llm_reasoning.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path.resolve()}")

    return results


if __name__ == "__main__":
    run_pipeline()
