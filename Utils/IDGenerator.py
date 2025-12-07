"""
id_generator.py
Handles UUID generation for files and chunks
"""

import logging
import uuid
from typing import List


class IDGenerator:
    """Generates unique identifiers for files and chunks."""

    def __init__(self, log_level=logging.INFO):
        """
        Initialize ID generator.

        Args:
            log_level: Logging level
        """
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
        """
        Generate a unique UUID for a file.

        Returns:
            str: UUID string
        """
        file_id = str(uuid.uuid4())
        self.logger.debug(f"Generated file ID: {file_id}")
        return file_id

    def generate_chunk_ids(self, num_chunks: int) -> List[str]:
        """
        Generate multiple UUIDs for chunks.

        Args:
            num_chunks (int): Number of chunk IDs to generate

        Returns:
            list: List of UUID strings
        """
        chunk_ids = [str(uuid.uuid4()) for _ in range(num_chunks)]
        self.logger.debug(f"Generated {num_chunks} chunk IDs")
        return chunk_ids

    def generate_batch_ids(self, num_files: int, chunks_per_file: List[int]) -> dict:
        """
        Generate IDs for multiple files and their chunks in batch.

        Args:
            num_files (int): Number of files
            chunks_per_file (list): List of chunk counts for each file

        Returns:
            dict: {
                'file_ids': [uuid1, uuid2, ...],
                'chunk_ids': [[chunk_uuid1, chunk_uuid2], [chunk_uuid3, ...]]
            }
        """
        if len(chunks_per_file) != num_files:
            raise ValueError("chunks_per_file length must match num_files")

        file_ids = [self.generate_file_id() for _ in range(num_files)]
        chunk_ids = [self.generate_chunk_ids(count) for count in chunks_per_file]

        self.logger.info(
            f"Generated {num_files} file IDs and {sum(chunks_per_file)} chunk IDs"
        )

        return {"file_ids": file_ids, "chunk_ids": chunk_ids}


# Usage example
if __name__ == "__main__":
    generator = IDGenerator()

    # Generate single file ID
    file_id = generator.generate_file_id()
    print(f"File ID: {file_id}")

    # Generate chunk IDs
    chunk_ids = generator.generate_chunk_ids(5)
    print(f"Chunk IDs: {chunk_ids}")

    # Generate batch
    batch = generator.generate_batch_ids(num_files=2, chunks_per_file=[3, 5])
    print(f"Batch: {batch}")
