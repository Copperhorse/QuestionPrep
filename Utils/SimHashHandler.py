"""
simhash_handler.py
Handles SimHash generation and duplicate detection
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from simhash import Simhash, SimhashIndex


class SimHashHandler:
    """Handles SimHash generation and duplicate detection for documents."""

    def __init__(self, k=5, log_level=logging.INFO):
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
        """
        Generate SimHash for text content.

        Args:
            text (str): Text content to hash (full document recommended)
            use_words (bool): If True, use word-based features
                            If False, use character n-grams (default, more precise)

        Returns:
            str: Hexadecimal SimHash string (64-bit)
        """
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
            self.logger.debug(f"Generated simhash: {simhash_hex}")
            return simhash_hex

        except Exception as e:
            self.logger.error(f"Error generating simhash: {str(e)}")
            raise

    def _get_ngram_features(self, text: str, width: int = 3) -> List[str]:
        """
        Generate character n-gram features for SimHash.

        Args:
            text (str): Input text
            width (int): N-gram width (default: 3)

        Returns:
            list: List of n-gram features
        """
        text = text.lower()
        text = re.sub(r"[^\w]+", "", text)
        return [text[i : i + width] for i in range(max(len(text) - width + 1, 1))]

    def add_to_index(self, file_id: str, simhash_hex: str):
        """
        Add a simhash to the index for future duplicate detection.

        Args:
            file_id (str): Unique file identifier
            simhash_hex (str): Hexadecimal simhash string
        """
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

    def load_index_from_data(self, data: List[Tuple[str, str]]):
        """
        Load multiple simhashes into the index.

        Args:
            data (list): List of tuples [(file_id, simhash_hex), ...]
        """
        try:
            index_data = []
            for file_id, simhash_hex in data:
                simhash_value = int(simhash_hex, 16)
                index_data.append((file_id, Simhash(value=simhash_value)))

            self.simhash_index = SimhashIndex(index_data, k=self.k)
            self.logger.info(f"Loaded {len(data)} simhashes into index")

        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise

    def check_duplicate(self, text: str, use_words: bool = False) -> Dict[str, Any]:
        """
        Check if document is a duplicate.

        Args:
            text (str): Full document text
            use_words (bool): Feature extraction mode

        Returns:
            dict: {
                'is_duplicate': bool,
                'duplicate_file_ids': list,
                'simhash': str
            }
        """
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
                self.logger.debug(f"Similar to file IDs: {duplicate_file_ids}")

            return {
                "is_duplicate": is_duplicate,
                "duplicate_file_ids": duplicate_file_ids,
                "simhash": simhash_hex,
            }

        except Exception as e:
            self.logger.error(f"Error checking duplicates: {str(e)}")
            return {"is_duplicate": False, "duplicate_file_ids": [], "simhash": None}

    def calculate_distance(self, simhash1: str, simhash2: str) -> int:
        """
        Calculate Hamming distance between two SimHashes.

        Args:
            simhash1 (str): First SimHash (hex string)
            simhash2 (str): Second SimHash (hex string)

        Returns:
            int: Hamming distance (0 = identical, 64 = completely different)
        """
        try:
            hash1 = Simhash(value=int(simhash1, 16))
            hash2 = Simhash(value=int(simhash2, 16))
            distance = hash1.distance(hash2)
            self.logger.debug(f"Distance between hashes: {distance}")
            return distance
        except Exception as e:
            self.logger.error(f"Error calculating distance: {str(e)}")
            raise

    def get_index_size(self) -> int:
        """
        Get the number of entries in the index.

        Returns:
            int: Number of indexed documents
        """
        # SimhashIndex doesn't expose size directly, so we track it
        return len(self.simhash_index.bucket)


# Usage example
if __name__ == "__main__":
    handler = SimHashHandler(k=3)

    # Generate simhash
    doc1 = "This is a simple document for testing."
    simhash1 = handler.generate_simhash(doc1)
    print(f"SimHash: {simhash1}")

    # Add to index
    handler.add_to_index("file_001", simhash1)

    # Check duplicate
    doc2 = "This is a sample document for training"  # Very similar
    result = handler.check_duplicate(doc2)
    print(f"Is duplicate: {result['is_duplicate']}")
    print(f"Similar to: {result['duplicate_file_ids']}")

    # Calculate distance
    simhash2 = result["simhash"]
    distance = handler.calculate_distance(simhash1, simhash2)
    print(f"Hamming distance: {distance}")
