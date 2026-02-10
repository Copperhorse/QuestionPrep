import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from openai import OpenAI
from qp_core.DBManager import DBManager

# ---------------- CONFIG ----------------
DB_PATH = project_root / "data" / "rag_staging.db"
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "llama-model"
MAX_WORKERS = 4  # Keep at 4; the Validator is local and nearly instant
MAX_CONTEXT_CHARS = 10000

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Enricher")


# ---------------- LLM CLIENT ----------------
class LLMClient:
    def __init__(self, base_url, api_key="no-key"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_metadata(self, text: str, section: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a Technical Knowledge Architect. Your task is to convert technical text "
            "into a structured JSON knowledge graph. You must avoid generic or short answers."
        )

        # PART A: THE INTENT-BASED PROMPT
        user_prompt = f"""
### INPUT DATA:
**SECTION:** {section}
**TEXT:** {text[:4000]}

### TASK: Generate 3 QA pairs based on these INTENT CATEGORIES:
1. **COMPONENT (Structural):** Identify a specific module or framework part.
   - *Constraint:* Define the part and its relation to the whole system.
2. **PROCESS (Causal):** Trace how data/logic flows through a mechanism.
   - *Constraint:* Use causal connectors (e.g., "results in", "leads to", "triggers").
3. **CONSTRAINT (Failure-Mode):** Identify a limitation or trade-off.
   - *Constraint:* Explain WHY this limitation exists.

### QUALITY RULES:
- Answers MUST be 2-3 sentences (~30 words).
- Focus EXCLUSIVELY on technical content. Ignore analogies (chefs, cars, etc.).
- Use JSON format only.

### TARGET JSON:
"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                presence_penalty=0.3,  # Higher penalty to encourage longer, varied responses
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None


# ---------------- ENRICHMENT MANAGER ----------------
class EnrichmentManager:
    def __init__(self):
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)

    # PART B: THE HEURISTIC VALIDATOR (THE GOVERNOR)
    def validate_qa_quality(
        self, qa_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        validated = []
        causal_keywords = [
            "because",
            "results in",
            "leads to",
            "due to",
            "enables",
            "prevents",
            "requires",
            "triggers",
        ]
        forbidden_analogies = [
            "chef",
            "kitchen",
            "cooking",
            "restaurant",
            "recipe",
            "ingredient",
        ]

        for qa in qa_pairs:
            q_text = qa.get("question", qa.get("question_text", "")).lower()
            a_text = qa.get("answer", qa.get("answer_text", ""))
            q_type = qa.get("type", "Component")

            # Rule 1: The "No Single Line" Rule
            word_count = len(a_text.split())
            if word_count < 15:
                logger.warning(f"Rejected: Answer too short ({word_count} words)")
                continue

            # Rule 2: Causal Enforcement for 'Process' and 'Constraint'
            if q_type in ["Process", "Constraint"]:
                if not any(word in a_text.lower() for word in causal_keywords):
                    logger.warning(f"Rejected: {q_type} answer lacks causal logic")
                    continue

            # Rule 3: Anti-Analogy (The "No Chef" Rule)
            if any(
                word in q_text or word in a_text.lower() for word in forbidden_analogies
            ):
                logger.warning(f"Rejected: Analogy detected ('{q_text[:20]}...')")
                continue

            # Standardize keys for DB
            validated.append(
                {
                    "question_type": q_type,
                    "question_text": qa.get("question", qa.get("question_text")),
                    "answer_text": a_text,
                    "difficulty": "Medium",  # Defaulting since we use Intent now
                }
            )

        return validated

    def process_file_sequentially(self, file_id: str):
        chunks = self.db.get_chunks_for_file_ordered(file_id)
        for chunk in chunks:
            # Skip if already done
            if chunk.get("existing_summary"):
                continue

            # 1. Propose
            metadata = self.llm.generate_metadata(
                chunk["content"], chunk["section_header"]
            )

            if metadata:
                # 2. Govern (Validate)
                raw_qa = metadata.get("qa_pairs", [])
                clean_qa = self.validate_qa_quality(raw_qa)

                # 3. Persist only if quality is met
                if clean_qa:
                    self.db.save_enrichment(chunk["chunk_id"], metadata)
                    self.db.save_questions(chunk["chunk_id"], clean_qa)
                    logger.info(
                        f"✔ Chunk {chunk['chunk_index']}: Saved {len(clean_qa)} valid QA pairs."
                    )
                else:
                    logger.warning(
                        f"✘ Chunk {chunk['chunk_index']}: No QA pairs passed validation."
                    )

    def run(self):
        while True:
            files = self.db.get_pending_files(limit=5)
            if not files:
                break
            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = {
                    ex.submit(self.process_file_sequentially, fid): fid for fid in files
                }
                for future in as_completed(futures):
                    future.result()


if __name__ == "__main__":
    manager = EnrichmentManager()
    # Basic server check
    try:
        manager.llm.client.models.list()
        manager.run()
    except Exception as e:
        print(f"Server not ready: {e}")
