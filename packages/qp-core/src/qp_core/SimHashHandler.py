"""
SimHashHandler.py
Handles SimHash generation and duplicate detection
"""

import logging
import re
from typing import Any, Dict, List, Tuple, Union

from simhash import Simhash, SimhashIndex


class SimHashHandler:
    """Handles SimHash generation and duplicate detection for documents."""

    def __init__(self, k=3, log_level=logging.INFO):
        """
        Initialize SimHash handler.
        Args:
            k (int): Distance threshold for duplicate detection (default: 3)
            log_level: Logging level
        """
        self.k = k
        self.simhash_index = SimhashIndex([], k=k)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized SimHashHandler with k={k}")

    def generate_simhash(self, text: str, use_words: bool = False) -> str:
        """Generate SimHash for text content."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for simhash generation")
            return "0" * 16

        try:
            if use_words:
                features = text.lower().split()
            else:
                features = self._get_ngram_features(text)

            sim = Simhash(features)
            simhash_hex = format(sim.value, "016x")
            return simhash_hex

        except Exception as e:
            self.logger.error(f"Error generating simhash: {str(e)}")
            raise

    def _get_ngram_features(self, text: str, width: int = 3) -> List[str]:
        """Generate character n-gram features for SimHash."""
        text = text.lower()
        text = re.sub(r"[^\w]+", "", text)
        return [text[i : i + width] for i in range(max(len(text) - width + 1, 1))]

    def add_to_index(self, file_id: str, simhash_hex: str):
        """Add a simhash to the index for future duplicate detection."""
        try:
            simhash_value = int(simhash_hex, 16)
            simhash_obj = Simhash(value=simhash_value)
            self.simhash_index.add(file_id, simhash_obj)
            self.logger.debug(
                f"Added simhash {simhash_hex} to index for file {file_id}"
            )
        except Exception as e:
            self.logger.error(f"Error adding to index: {str(e)}")
            raise

    def load_index_from_data(self, data: Union[List[Tuple[str, str]], Dict[str, str]]):
        """
        Load multiple simhashes into the index.
        Supports both Dict {simhash: file_id} and List [(file_id, simhash)].
        """
        try:
            index_data = []

            # ✅ FIX: Handle Dictionary Input (from DB loader)
            if isinstance(data, dict):
                # Dict is {simhash: file_id}
                for simhash_hex, file_id in data.items():
                    simhash_value = int(simhash_hex, 16)
                    index_data.append((file_id, Simhash(value=simhash_value)))

            # Handle List Input [(file_id, simhash)]
            else:
                for file_id, simhash_hex in data:
                    simhash_value = int(simhash_hex, 16)
                    index_data.append((file_id, Simhash(value=simhash_value)))

            self.simhash_index = SimhashIndex(index_data, k=self.k)
            self.logger.info(f"Loaded {len(index_data)} simhashes into index")

        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise

    def check_duplicate(self, text: str, use_words: bool = False) -> Dict[str, Any]:
        """Check if document is a duplicate."""
        try:
            # Generate simhash
            simhash_hex = self.generate_simhash(text, use_words=use_words)
            simhash_value = int(simhash_hex, 16)
            simhash_obj = Simhash(value=simhash_value)

            # Check for near duplicates in index
            duplicate_file_ids = self.simhash_index.get_near_dups(simhash_obj)

            is_duplicate = len(duplicate_file_ids) > 0

            if is_duplicate:
                self.logger.info(
                    f"Duplicate detected: {len(duplicate_file_ids)} similar file(s)"
                )

            return {
                "is_duplicate": is_duplicate,
                "duplicate_file_ids": duplicate_file_ids,
                "simhash": simhash_hex,
            }

        except Exception as e:
            self.logger.error(f"Error checking duplicates: {str(e)}")
            return {"is_duplicate": False, "duplicate_file_ids": [], "simhash": None}
