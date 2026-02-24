"""
test_full_mini_pipeline.py
Minimum Viable Full Test:
- Pass 1: Real question + quote generation (from your Enricher.py)
- Quote guard
- Pass 2: Natural human-like reference answers (with few-shot + double-prompt)
- Strict gate + scoring evaluator

Fixes applied:
  1. NOT_ENOUGH_INFORMATION is now a standalone-only response (no tailing)
  2. Pass 2 prompt has few-shot examples showing specific vs. vague answers
  3. Question is repeated twice in the user turn to reinforce focus
  4. Tighter max_tokens + explicit "do not pad" instruction
  5. Over-enumeration discouraged via explicit single-answer rule
"""

import logging
import re
from typing import Dict, List

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("MiniPipelineTest")

# ========================= CONFIG =========================
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "lfm-2.5-1.2b"
client = OpenAI(base_url=LLAMA_API_URL, api_key="no-key")

# ========================= CHUNK (your real chunk) =========================
CHUNK = """
## 2.1.2 Data Searching

While the previous data systems are platforms for sharing datasets, as a next logical step, we now explore systems that are mainly designed for searching datasets. This setting is common within large companies or on the Web.

Data Lake Data searching systems have become more popular with the advent of data lakes [19], [75] in corporate environments where many datasets are generated internally, but they are not easily discoverable by other teams or individuals within the company. Providing a way to search datasets and analyze them has significant business value because the teams or individuals do not have to make redundant efforts to re-generate the datasets for their machine learning tasks. Most of the recent data lake systems have come from the industry. In many cases, it is not feasible for all the dataset owners to publish datasets through one system. Instead, a post-hoc approach becomes necessary where datasets are

processed for searching after they are created, and no effort is required on the dataset owner's side.

As an early solution for data lakes, IBM proposed a system [19] that enables datasets to be curated and then searched. IBM estimates that 70% of the time spent on analytic projects is concerned with discovering, cleaning, and integrating datasets that are scattered among many business applications. Thus, IBM takes the stance of creating, filling, maintaining, and governing the data lake where these processes are collectively called data wrangling . When analyzing data, users do not perform the analytics directly on the data lake, but extract data sets and store them separately. Before this step, the users can do a preliminary exploration of datasets, e.g., visualizing them to determine if the dataset is useful and does not contain anomalies that need further investigation. While supporting data curation in the data lake saves users from processing raw data, it does limit the scalability of how many datasets can be indexed.

More recently, scalability has become a pressing issue for handling data lakes that consists of most datasets in a large company. Google Data Search (GOODS) [20] is a system that catalogues the metadata of tens of billions of datasets from various storage systems within Google. GOODS infers various metadata including owner information and provenance information (by looking up job logs), analyzes the contents of the datasets, and collects input from users. At the core is a central catalog, which contains the metadata and is indexed for data searching. Due to Google's scale, there are many technical challenges including scaling to the number of datasets, supporting a variety of data formats where the costs for extracting metadata may differ, updating the catalog entries due to the frequent churn of datasets, dealing with uncertainty in metadata discovery, computing dataset importance for search ranking, and recovering dataset semantics that are missing. To find datasets, users can use keywords queries on the GOODS frontend and view profile pages of the datasets that appear in the search results. In addition, users can track the provenance of a dataset to see which datasets were used to create the given dataset and those that rely on it.

Finally, expressive queries are also important for searching a data lake. While GOODS scales, one downside is that it only supports simple keyword queries. This approach is similar to keyword search in databases [76], [77], but the purpose is to find datasets instead of tuples. The DATA CIVILIZER system [21], [22] complements GOODS by focusing more on the discovery aspect of datasets. Specifically, DATA CIVILIZER consists of a module for building a linkage graph of data. Assuming that datasets have schema, the nodes in the linkage graph are columns of tables while edges are relationships like primary key-foreign key (PKFK) relationships. A data discovery module then supports a rich set of discovery queries on the linkage graph, which can help users more easily discover the relevant datasets. DATARAMAN [23] specializes in extracting structured data from semi-structured log datasets in data lakes automatically by learning patterns. AURUM [78], [79] supports data discovery queries on semantically-linked datasets.

Web As the Web contains large numbers of structured datasets, there have been significant efforts to automati- cally extract the useful ones [32]-[34]. One of the most successful systems is WebTables [24], [25], which automatically extracts structured data that is published online in the form of HTML tables. For example, WebTables extracts all Wikipedia infoboxes. Initially, about 14.1 billion HTML tables are collected from the Google search web crawl. Then a classifier is applied to determine which tables can be viewed as relational database tables. Each relational table consists of a schema that describes the columns and a set of tuples. In comparison to the above data lake systems, WebTables collects structured data from the Web.

As Web data tends to be much more diverse than say those in a corporate environment, the table extraction techniques have been extended in multiple ways as well. One direction is to extend table extraction beyond identifying HTML tags by extracting relational data in the form of vertical tables and lists and leveraging knowledge bases [27], [28]. Table searching also evolved where, in addition to keyword searching, row-subset queries, entity-attribute queries, and column search were introduced [29]. Finally, techniques for enhancing the tables [30], [31] were proposed where entities or attribute values are added to make the tables more complete.

Recently, a service called Google Dataset Search [26] was launched for searching repositories of datasets on the Web. The motivation is that there are thousands of data repositories on the Web that contain millions of datasets that are not easy to search. Dataset Search lets dataset providers describe their datasets using various metadata (e.g., author, publication date, how the data was collected, and terms for using the data) so that they become more searcheable. In comparison to the fully-automatic WebTables, dataset providers may need to do some manual work, but have the opportunity to make their datasets more searcheable. In comparison to GOODS, Dataset Search targets the Web instead of a data lake.

"""


# ========================= PASS 1: QUESTION + QUOTE GENERATION =========================
def generate_question_candidates(chunk_content: str) -> List[Dict]:
    per_diff = 2
    difficulties = {
        "Easy": "factual recall — ask about specific definitions, names, or stated facts",
        "Medium": "conceptual / mechanism — ask how or why something works",
        "Hard": "analytical / critical — ask about limitations, trade-offs, or comparisons",
    }

    sys_prompt = "You are an Expert Technical Interviewer. Output ONLY valid JSON."
    all_candidates = []

    for diff, instruction in difficulties.items():
        user_prompt = f"""
### TARGET TEXT CHUNK (You MUST extract questions and quotes ONLY from here):
{chunk_content}

### TASK:
Generate up to {per_diff} '{diff}' questions ({instruction}).
Questions must be EXCLUSIVELY answerable from the TARGET TEXT CHUNK.
Every "source_quote" MUST be copied verbatim from the chunk (≥25 characters).

Output ONLY:
{{"qa_pairs": [ {{"question": "...", "source_quote": "...", "difficulty": "{diff}"}}, ... ] }}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content.strip()
            import json

            data = json.loads(result)
            for item in data.get("qa_pairs", [])[:per_diff]:
                item["difficulty"] = diff
                all_candidates.append(item)
        except Exception as e:
            logger.error(f"Pass 1 failed for {diff}: {e}")

    logger.info(f"Pass 1 generated {len(all_candidates)} candidate questions")
    return all_candidates


# ========================= PASS 2: HUMAN-LIKE ANSWER =========================

# Few-shot examples that demonstrate the difference between vague and grounded answers.
# These are static and based on the same domain so the model internalises the style.
FEW_SHOT_EXAMPLES = """
### EXAMPLES OF GOOD vs. BAD ANSWERS:

QUESTION: What does IBM estimate about time spent on analytic projects?
BAD (vague, padded): IBM found that a lot of time is spent on data-related tasks, which slows down analytics teams significantly. This is a major problem in the industry.
GOOD (specific, grounded): IBM estimates that 70% of the time spent on analytic projects is concerned with discovering, cleaning, and integrating datasets.

---

QUESTION: What type of queries does GOODS support?
BAD (over-lists other systems): GOODS supports keyword queries, but DATA CIVILIZER supports richer ones, and WebTables has its own search. AURUM also supports discovery queries. NOT_ENOUGH_INFORMATION
GOOD (direct, single answer): GOODS only supports simple keyword queries — users type keywords on the GOODS frontend and view profile pages of matching datasets.

---

QUESTION: What is a linkage graph in DATA CIVILIZER?
BAD (generic): DATA CIVILIZER uses graph structures to help organize data relationships in a useful way for discovery purposes.
GOOD (text-grounded): In DATA CIVILIZER, a linkage graph has columns of tables as nodes and edges representing relationships like primary key-foreign key (PKFK) relationships.
"""


def extract_content_nouns(text: str) -> set:
    """
    Extract meaningful content words (4+ chars, non-stopword) from text.
    Used to catch hallucinated noun phrases that don't appear in the source chunk.
    Intentionally simple — no NLTK dependency, no POS tagging.
    """
    STOPWORDS = {
        "that",
        "this",
        "with",
        "from",
        "they",
        "them",
        "their",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "also",
        "such",
        "which",
        "when",
        "where",
        "what",
        "than",
        "then",
        "into",
        "more",
        "some",
        "only",
        "each",
        "other",
        "these",
        "those",
        "about",
        "over",
        "while",
        "being",
        "after",
        "before",
        "between",
        "through",
        "within",
        "without",
        "during",
        "however",
        "therefore",
        "because",
        "using",
        "include",
        "includes",
        "including",
        "allows",
        "provides",
        "enables",
        "offer",
        "offers",
        "require",
        "requires",
        "focus",
        "focuses",
    }
    words = re.findall(r"\b[a-z]{4,}\b", normalize(text))
    return {w for w in words if w not in STOPWORDS}


def check_hallucination(answer: str, chunk: str) -> tuple[bool, list]:
    """
    Flag answers where meaningful content words appear in the answer
    but NOT in the source chunk. Returns (is_clean, flagged_words).
    Threshold: if >20% of answer's content words are absent from chunk → flag.
    """
    answer_nouns = extract_content_nouns(answer)
    chunk_nouns = extract_content_nouns(chunk)
    missing = answer_nouns - chunk_nouns
    if not answer_nouns:
        return True, []
    ratio = len(missing) / len(answer_nouns)
    return ratio <= 0.20, list(missing)


def generate_human_reference(question: str, chunk: str) -> str:
    # No "sound natural" or "helpful colleague" framing.
    # Faithful extraction > fluent paraphrasing for this use case.
    sys_prompt = (
        "You are a precise technical answer extractor. "
        "You answer questions using ONLY words and phrases from the provided source text. "
        "You do not infer, interpolate, or add any information not explicitly stated. "
        "You never combine a real answer with NOT_ENOUGH_INFORMATION — it is one or the other."
    )

    # Partial prompt repetition: repeat only the rules + question, NOT the chunk.
    # Full repetition doubled prefill cost to ~3,500 tokens on this hybrid SSM/attention
    # model (LFM lfm-2.5) because SWA invalidates KV cache between requests.
    # Repeating only the lightweight tail preserves the re-reading signal cheaply.
    rules_and_question = f"""### RULES — FOLLOW EXACTLY:
- Answer in 1–3 sentences using words and phrases directly from the SOURCE TEXT.
- Do NOT paraphrase beyond minimal grammar. Copy key terms verbatim where possible.
- Answer ONLY the question asked. Do NOT mention other systems unless the question asks for a comparison.
- Do NOT add any claim, qualifier, or word that does not appear in the SOURCE TEXT.
- If the answer is genuinely not in the text, output ONLY: NOT_ENOUGH_INFORMATION
- Never output NOT_ENOUGH_INFORMATION after a real answer. It is one or the other.

### QUESTION:
{question}"""

    user_prompt = f"""### SOURCE TEXT (your answer must stay within this text):
{chunk}

{FEW_SHOT_EXAMPLES}

{rules_and_question}

Let me repeat that:

{rules_and_question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low = echo source text, not creative paraphrase
            max_tokens=180,
        )
        raw = response.choices[0].message.content.strip()

        # Guard 1: strip trailing NOT_ENOUGH_INFORMATION if a real answer also exists
        if "NOT_ENOUGH_INFORMATION" in raw:
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            non_flag_lines = [l for l in lines if l != "NOT_ENOUGH_INFORMATION"]
            if non_flag_lines:
                logger.debug("Stripped trailing NOT_ENOUGH_INFORMATION from answer.")
                raw = " ".join(non_flag_lines)
            else:
                return "NOT_ENOUGH_INFORMATION"

        # Guard 2: hallucination check — reject answers where >20% of content
        # words are absent from the source chunk (catches fluency-driven invention)
        is_clean, flagged = check_hallucination(raw, chunk)
        if not is_clean:
            logger.warning(f"⚠️  Hallucination flag — words not in chunk: {flagged}")
            return "HALLUCINATION_FLAGGED"

        return raw
    except Exception as e:
        logger.error(f"Pass 2 failed: {e}")
        return "ERROR"


# ========================= EVALUATOR =========================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def token_overlap(a: str, b: str) -> float:
    a_tokens = set(normalize(a).split())
    b_tokens = set(normalize(b).split())
    return len(a_tokens & b_tokens) / len(a_tokens) if a_tokens else 0.0


QUESTION_GATE = 0.16
ANSWER_GATE = 0.23


def score_pair(q: str, a: str, chunk: str) -> float:
    grounding = token_overlap(a, chunk) * 10
    relevance = token_overlap(a, q) * 10
    natural_bonus = 1.0 if 8 <= len(a.split()) <= 48 else 0.8
    return round(0.60 * grounding + 0.25 * relevance + 0.15 * natural_bonus, 2)


# ========================= MAIN TEST =========================
if __name__ == "__main__":
    logger.info("🚀 Starting FULL Mini Pipeline Test (Pass 1 + Pass 2 + Eval)")

    # === PASS 1 ===
    candidates = generate_question_candidates(CHUNK)

    valid_pairs = []
    for cand in candidates:
        q = cand.get("question", "").strip()
        quote = cand.get("source_quote", "").strip()
        if len(quote) < 25:
            continue
        score = token_overlap(quote, CHUNK) * 100
        if score >= 75:
            valid_pairs.append({"question": q, "source_quote": quote})

    logger.info(f"After quote guard: {len(valid_pairs)} valid QA pairs")

    # === PASS 2 + EVALUATION ===
    results = []
    scores = []
    skipped_not_enough = 0
    skipped_hallucination = 0
    skipped_gates = 0

    for i, pair in enumerate(valid_pairs, 1):
        q = pair["question"]
        logger.info(f"\n=== Pair {i} ===")
        logger.info(f"Q: {q}")

        answer = generate_human_reference(q, CHUNK)
        print(f"→ Answer: {answer}\n")

        if answer == "NOT_ENOUGH_INFORMATION":
            logger.warning("⏭  Skipped — model says NOT_ENOUGH_INFORMATION")
            skipped_not_enough += 1
            continue

        if answer == "HALLUCINATION_FLAGGED":
            logger.warning("🚫 Skipped — hallucination guard triggered")
            skipped_hallucination += 1
            continue

        if not token_overlap(q, CHUNK) >= QUESTION_GATE:
            logger.warning("❌ Question gate failed")
            skipped_gates += 1
            continue
        if not token_overlap(answer, CHUNK) >= ANSWER_GATE:
            logger.warning("❌ Grounding gate failed")
            skipped_gates += 1
            continue

        score = score_pair(q, answer, CHUNK)
        logger.info(f"✅ PASS - Score: {score:.2f}/10")
        results.append((i, score))
        scores.append(score)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    passed = len(scores)
    avg = round(sum(scores) / passed, 2) if passed else 0.0

    for r in results:
        print(r)
    print(f"\nPassed:              {passed}/{len(valid_pairs)}")
    print(f"Average Score:       {avg}/10")
    print(f"Hallucination drops: {skipped_hallucination}")
    print(f"NOT_ENOUGH drops:    {skipped_not_enough}")
    print(f"Gate failures:       {skipped_gates}")

    if avg >= 7.0:
        logger.info("🎉 EXCELLENT — Full pipeline is working great!")
    elif avg >= 5.5:
        logger.info("👍 Good enough for production")
    else:
        logger.warning("⚠️  Still needs tweaking")
