"""
IDGenerator.py
Handles UUID generation for files and chunks
"""

import logging
import uuid
from typing import List


class IDGenerator:
    """Generates unique identifiers for files and chunks."""

    def __init__(self, log_level=logging.INFO):
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

    def generate_file_id(self) -> str:
        """Generate a unique UUID for a file."""
        file_id = str(uuid.uuid4())
        self.logger.debug(f"Generated file ID: {file_id}")
        return file_id

    def generate_chunk_ids(self, num_chunks: int) -> List[str]:
        """Generate multiple UUIDs for chunks."""
        chunk_ids = [str(uuid.uuid4()) for _ in range(num_chunks)]
        self.logger.debug(f"Generated {num_chunks} chunk IDs")
        return chunk_ids
