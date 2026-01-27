"""
Enricher.py
Production-Grade Enrichment Pipeline (Liquid Model Optimized)
Features:
- Flexible Logic ("Insight") mapped to Strict DB Schema ("Critical")
- Anti-Tautology Instructions
- Liquid Model Settings (10k Context, 4096 Tokens)
"""

import json
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Deque, Dict, List, Set

# --- IMPORT FIX: Add parent directory to path so Utils import works ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from openai import OpenAI

from Utils.DBManager import DBManager

# ---------------- CONFIG ----------------
DB_PATH = os.path.join(parent_dir, "rag_staging.db")
LLAMA_API_URL = "http://localhost:8080/v1"
MODEL_NAME = "llama-model"
MAX_WORKERS = 4

# LIQUID MODEL OPTIMIZATIONS
MAX_CONTEXT_CHARS = 10000

# Validation Thresholds
MIN_SUMMARY_CHARS = 50
MIN_TAGS = 3
MIN_TRIPLETS = 2

PREDICATE_WHITELIST = {
    "is_a",
    "part_of",
    "contains",
    "causes",
    "prevents",
    "optimizes",
    "requires",
    "enables",
    "produces",
    "uses",
    "calls",
    "inherits_from",
    "defined_as",
    "has_property",
    "critiques",
    "contrasts_with",
    "limits",
}

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Enricher")


# ---------------- LLM CLIENT ----------------
class LLMClient:
    def __init__(self, base_url, api_key="no-key"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_metadata(
        self, text: str, context_text: str, known_entities: Set[str], retries=2
    ) -> Dict[str, Any]:
        """
        Flexible prompt that maps nuance to strict DB schemas.
        """
        # Format Known Entities
        entity_list_str = (
            ", ".join(sorted(list(known_entities))[:25]) if known_entities else "None"
        )

        system_prompt = (
            "You are an Expert Technical Interviewer. "
            "Your goal is to Extract Knowledge. "
            "Avoid robotic answers. Use natural, complete sentences."
        )

        user_prompt = f"""
        ### SECTION CONTEXT:
        {context_text if context_text else "(None)"}

        ### KNOWN ENTITIES:
        {entity_list_str}

        ### TEXT:
        {text[:4000]}

        ### TASKS:

        1. **Summary (2-3 sentences):**
           - Start with the SUBJECT.
           - Capture the main technical content.

        2. **Graph Data:**
           - Extract 5-8 Noun Tags.
           - Extract Triplets explicitly stated in text.
           - Predicates: {json.dumps(list(PREDICATE_WHITELIST))}

        3. **Interview Questions (3 Pairs):**
           * **Anti-Tautology:** Do NOT say "X is important because it is key." Explain WHY.

           - **Pair 1 (Fact):** Ask for a definition ("What is X?").
             * *Type Label:* "Fact"

           - **Pair 2 (Mechanism):** Ask how a process works or relationships ("How does X work?").
             * *Type Label:* "Mechanism"

           - **Pair 3 (Critical):** Ask about limitation, risk, OR significance ("Why is X significant?").
             * *Type Label:* "Critical"

        ### REQUIRED JSON OUTPUT:
        {{
            "summary": "The text describes...",
            "tags": ["Tag1", "Tag2"],
            "triplets": [
                {{"subject": "Entity", "predicate": "uses", "object": "Tool"}}
            ],
            "qa_pairs": [
                {{
                    "difficulty": "Easy",
                    "type": "Fact",
                    "question": "What is...?",
                    "answer": "It is..."
                }},
                {{
                    "difficulty": "Medium",
                    "type": "Mechanism",
                    "question": "How does...?",
                    "answer": "It works by..."
                }},
                {{
                    "difficulty": "Hard",
                    "type": "Critical",
                    "question": "What is the significance of...?",
                    "answer": "It is significant because..."
                }}
            ]
        }}
        """

        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"LLM Error (Attempt {attempt}): {e}")
                time.sleep(1)

        return None


# ---------------- ENRICHMENT MANAGER ----------------
class EnrichmentManager:
    def __init__(self):
        self.db = DBManager(DB_PATH)
        self.llm = LLMClient(LLAMA_API_URL)

    def _build_context(self, summaries: Deque[str], entities: Set[str]) -> str:
        """Constructs the context string."""
        lines, size = [], 0
        for s in reversed(summaries):
            if size + len(s) > MAX_CONTEXT_CHARS:
                break
            lines.insert(0, f"- {s}")
            size += len(s)
        return "\n".join(lines)

    def _sanitize_qa_pairs(
        self, qa_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ensures Q&A pairs match strict DB constraints."""
        valid_types = {"Fact", "Mechanism", "Critical"}
        # Mapping loosely generated types to strict DB types
        type_map = {
            "Concept": "Fact",
            "Definition": "Fact",
            "Explanation": "Mechanism",
            "Process": "Mechanism",
            "Insight": "Critical",
            "Trade-off": "Critical",
            "Limitation": "Critical",
        }

        sanitized = []
        for qa in qa_pairs:
            q_type = qa.get("type", "Fact")
            # 1. Map known variants
            if q_type in type_map:
                q_type = type_map[q_type]
            # 2. Fallback if still invalid
            if q_type not in valid_types:
                q_type = "Fact"  # Safe default

            qa["type"] = q_type
            sanitized.append(qa)
        return sanitized

    def process_file_sequentially(self, file_id: str):
        logger.info(f"📂 Starting File: {file_id[:8]}...")
        chunks = self.db.get_chunks_for_file_ordered(file_id)

        section = None
        summaries: Deque[str] = deque()
        entities: Set[str] = set()

        processed_count = 0

        for chunk in chunks:
            # 1. Section Boundary Check
            if chunk["section_header"] != section:
                summaries.clear()
                entities.clear()
                section = chunk["section_header"]

            # 2. Resume Logic
            if chunk.get("existing_summary"):
                summaries.append(chunk["existing_summary"])
                if chunk.get("existing_tags"):
                    try:
                        tags = json.loads(chunk["existing_tags"])
                        entities.update(tags)
                    except:
                        pass
                continue

            # 3. Enrich
            context = self._build_context(summaries, entities)
            metadata = self.llm.generate_metadata(chunk["content"], context, entities)

            # 4. Save & Update State
            if metadata:
                # A. Save Base Enrichment
                self.db.save_enrichment(chunk["chunk_id"], metadata)

                # B. Save Questions (With Sanitization)
                qa_pairs = metadata.get("qa_pairs", [])
                if qa_pairs:
                    clean_qa = self._sanitize_qa_pairs(qa_pairs)
                    self.db.save_questions(chunk["chunk_id"], clean_qa)

                # C. Update State
                if metadata.get("summary"):
                    summaries.append(metadata["summary"])
                if metadata.get("tags"):
                    entities.update(metadata["tags"])

                processed_count += 1
                logger.info(
                    f"✔ Chunk {chunk['chunk_index']} Enriched ({len(qa_pairs)} Qs)"
                )
            else:
                logger.warning(f"✘ Chunk {chunk['chunk_index']} Failed")

        logger.info(f"✅ File {file_id[:8]} Finished. ({processed_count} new chunks)")

    def run(self):
        while True:
            files = self.db.get_pending_files(limit=10)
            if not files:
                logger.info("Pipeline Complete. No pending files.")
                break

            logger.info(f"--- Processing Batch of {len(files)} Files ---")

            with ThreadPoolExecutor(MAX_WORKERS) as ex:
                futures = {
                    ex.submit(self.process_file_sequentially, fid): fid for fid in files
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Crash in file thread: {e}")


if __name__ == "__main__":
    manager = EnrichmentManager()
    print("Checking Llama Server...")
    try:
        manager.llm.client.models.list()
        print("Server Ready. Starting Pipeline.")
        manager.run()
    except Exception as e:
        print(f"Server Error: {e}")
        print("Make sure run_pipeline.sh is running!")
