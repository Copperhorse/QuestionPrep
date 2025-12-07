"""
csv_manager.py
Handles saving data to CSV files
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class CSVManager:
    """Manages CSV file operations for files and chunks."""

    def __init__(self, output_dir="output", log_level=logging.INFO):
        """
        Initialize CSV manager.

        Args:
            output_dir (str): Directory to save CSV files
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

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

        self.logger.info(f"CSV output directory: {self.output_dir.absolute()}")

    def save_file_metadata(
        self,
        file_id: str,
        file_path: str,
        simhash: str,
        metadata: Dict[str, Any],
        content_length: int,
        csv_filename: str = "files.csv",
    ) -> bool:
        """
        Save file metadata to CSV.

        Args:
            file_id (str): Unique file identifier (UUID)
            file_path (str): Original file path
            simhash (str): SimHash of the document
            metadata (dict): File metadata from PyMuPDF
            content_length (int): Length of markdown content
            csv_filename (str): Output CSV filename

        Returns:
            bool: True if successful
        """
        try:
            csv_path = self.output_dir / csv_filename
            file_exists = csv_path.exists()

            row = {
                "file_id": file_id,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "simhash": simhash,
                "format": metadata.get("format", ""),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "mod_date": metadata.get("modDate", ""),
                "trapped": metadata.get("trapped", ""),
                "encryption": str(metadata.get("encryption", "")),
                "processed_at": datetime.now().isoformat(),
                "content_length": content_length,
            }

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()
                    self.logger.debug(f"Created new CSV: {csv_path}")

                writer.writerow(row)

            self.logger.info(f"Saved file metadata: {file_id} to {csv_filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving file metadata: {str(e)}")
            return False

    def save_chunks(
        self,
        file_id: str,
        chunk_ids: List[str],
        chunks: List[Dict[str, Any]],
        csv_filename: str = "chunks.csv",
    ) -> bool:
        """
        Save chunk data to CSV.

        Args:
            file_id (str): Parent file UUID
            chunk_ids (list): Pre-generated chunk UUIDs
            chunks (list): List of chunks with content and metadata
            csv_filename (str): Output CSV filename

        Returns:
            bool: True if successful
        """
        try:
            if len(chunk_ids) != len(chunks):
                raise ValueError("Number of chunk_ids must match number of chunks")

            csv_path = self.output_dir / csv_filename
            file_exists = csv_path.exists()

            rows = []

            for idx, (chunk_id, chunk) in enumerate(zip(chunk_ids, chunks)):
                content = chunk.get("content", "")
                meta = chunk.get("metadata", {})

                row = {
                    "chunk_id": chunk_id,
                    "file_id": file_id,
                    "chunk_index": chunk.get("chunk_index", idx),
                    "content": content,
                    "section_header": meta.get("section_header", ""),
                    "section_level": meta.get("section_level", ""),
                    "parent_section": meta.get("parent_section", ""),
                    "content_type": meta.get("content_type", ""),
                    "estimated_tokens": chunk.get("estimated_tokens", 0),
                    "prev_chunk_id": chunk_ids[idx - 1] if idx > 0 else "",
                    "next_chunk_id": chunk_ids[idx + 1]
                    if idx < len(chunk_ids) - 1
                    else "",
                    "created_at": datetime.now().isoformat(),
                }

                rows.append(row)

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())

                    if not file_exists:
                        writer.writeheader()
                        self.logger.debug(f"Created new CSV: {csv_path}")

                    writer.writerows(rows)

            self.logger.info(f"Saved {len(chunks)} chunks to {csv_filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            return False

    def load_existing_simhashes(
        self, csv_filename: str = "files.csv"
    ) -> List[Tuple[str, str]]:
        """
        Load existing file_id and simhash pairs from CSV.
        Useful for initializing SimHashHandler index.

        Args:
            csv_filename (str): CSV filename to load from

        Returns:
            list: List of tuples [(file_id, simhash), ...]
        """
        csv_path = self.output_dir / csv_filename

        if not csv_path.exists():
            self.logger.info(f"No existing CSV found: {csv_filename}")
            return []

        try:
            data = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_id = row.get("file_id")
                    simhash = row.get("simhash")
                    if file_id and simhash:
                        data.append((file_id, simhash))

            self.logger.info(f"Loaded {len(data)} simhashes from {csv_filename}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading simhashes: {str(e)}")
            return []

    def get_file_count(self, csv_filename: str = "files.csv") -> int:
        """
        Get the number of files in the CSV.

        Args:
            csv_filename (str): CSV filename

        Returns:
            int: Number of files
        """
        csv_path = self.output_dir / csv_filename

        if not csv_path.exists():
            return 0

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = sum(1 for _ in reader)
            return count
        except Exception as e:
            self.logger.error(f"Error counting files: {str(e)}")
            return 0

    def get_chunk_count(self, csv_filename: str = "chunks.csv") -> int:
        """
        Get the number of chunks in the CSV.

        Args:
            csv_filename (str): CSV filename

        Returns:
            int: Number of chunks
        """
        csv_path = self.output_dir / csv_filename

        if not csv_path.exists():
            return 0

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = sum(1 for _ in reader)
            return count
        except Exception as e:
            self.logger.error(f"Error counting chunks: {str(e)}")
            return 0


# Usage example
if __name__ == "__main__":
    manager = CSVManager(output_dir="output")

    # Example file metadata
    file_metadata = {
        "format": "PDF 1.7",
        "title": "Sample Document",
        "author": "John Doe",
        "creator": "Microsoft Word",
        "creationDate": "2024-01-01",
    }

    # Save file metadata
    success = manager.save_file_metadata(
        file_id="uuid-123",
        file_path="/path/to/file.pdf",
        simhash="a5f3c8d1b2e47690",
        metadata=file_metadata,
        content_length=5000,
    )
    print(f"File saved: {success}")

    # Example chunks
    chunks = [
        {
            "content": "Chunk 1 content",
            "metadata": {"section_header": "Introduction", "section_level": 1},
            "estimated_tokens": 50,
            "chunk_index": 0,
        },
        {
            "content": "Chunk 2 content",
            "metadata": {"section_header": "Methods", "section_level": 1},
            "estimated_tokens": 75,
            "chunk_index": 1,
        },
    ]

    chunk_ids = ["chunk-uuid-1", "chunk-uuid-2"]

    # Save chunks
    success = manager.save_chunks(
        file_id="uuid-123", chunk_ids=chunk_ids, chunks=chunks
    )
    print(f"Chunks saved: {success}")

    # Load existing simhashes
    simhashes = manager.load_existing_simhashes()
    print(f"Loaded {len(simhashes)} simhashes")

    # Get counts
    print(f"Total files: {manager.get_file_count()}")
    print(f"Total chunks: {manager.get_chunk_count()}")
