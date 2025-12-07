"""
chunk_evaluator.py
Evaluates chunk quality and filters out low-value content
"""

import logging
import re
from typing import Any, Dict, List


class ChunkEvaluator:
    """
    Evaluates chunks for quality and usefulness.
    Filters out metadata, TOC, references, and other non-content chunks.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Initialize the chunk evaluator.

        Args:
            log_level: Logging level
        """
        # Header-based exclusions
        self.metadata_excludes = [
            "table of contents",
            "references",
            "bibliography",
            "acknowledgment",
            "acknowledgement",
            "appendix",
            "author",
            "index",
            "glossary",
        ]

        # Content prefix exclusions
        self.content_prefix_excludes = [
            "the image is",
            "<!-- image -->",
            "figure:",
            "fig.",
        ]

        # Content anywhere exclusions
        self.content_anywhere_excludes = [
            "copyright",
            "doi:",
            "contact@",
            "http://",
            "https://",
            "www.",
            "all rights reserved",
        ]

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

    def _alpha_ratio(self, text: str) -> float:
        """
        Calculate the ratio of alphabetic characters in text.

        Args:
            text (str): Text to analyze

        Returns:
            float: Ratio of alphabetic characters (0.0 to 1.0)
        """
        if not text:
            return 0.0
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / max(1, len(text))

    def _is_useful_table(self, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Detect if this is a data table (useful) vs formatting table (not useful).

        Args:
            text (str): Chunk content
            metadata (dict): Chunk metadata

        Returns:
            bool: True if useful table, False otherwise
        """
        lines = text.split("\n")

        # Count table rows (lines with |)
        table_rows = sum(1 for line in lines if "|" in line)

        # Get header
        header = metadata.get("section_header", "").lower()

        # Table of contents indicators
        if any(x in header for x in ["table of contents", "contents"]):
            return False

        # If it has substantial table content (>5 rows) and meaningful header, keep it
        if table_rows > 5 and header and header not in ["", "##", "###"]:
            return True

        return False

    def _is_table_of_contents(self, text: str) -> bool:
        """
        Detect if chunk is a table of contents.

        Args:
            text (str): Chunk content

        Returns:
            bool: True if TOC detected
        """
        text_lower = text.lower()
        lines = text.splitlines()

        # Detect TOC-like structure: many lines with dots and trailing page numbers
        toc_patterns = sum(
            1 for line in lines if re.search(r"\.{3,}\s*\d{1,4}$", line.strip())
        )

        toc_keywords = "table of contents" in text_lower[:400]

        return toc_patterns > 3 or toc_keywords

    def evaluate_chunk(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single chunk for quality and usefulness.

        Args:
            content (str): Chunk content
            metadata (dict): Chunk metadata

        Returns:
            dict: {
                'should_use': bool,
                'reason': str,
                'content_type': str,
                'quality_score': float (0-100)
            }
        """
        text = (content or "").strip()
        text_lower = text.lower()

        # Extract header text from metadata
        header_text = ""
        for key in (
            "section_header",
            "Header 3",
            "Header 2",
            "Header 1",
            "h3",
            "h2",
            "h1",
        ):
            if metadata.get(key):
                header_text = (metadata.get(key) or "").lower()
                break

        # 1. Header-based exclusions
        for exclude in self.metadata_excludes:
            if exclude in header_text:
                self.logger.debug(f"Excluded: Header contains '{exclude}'")
                return {
                    "should_use": False,
                    "reason": f"Header contains '{exclude}'",
                    "content_type": "metadata",
                    "quality_score": 0,
                }

        # 2. Empty or too short
        if len(text) < 30 or text_lower in ["", "<!-- image -->"]:
            self.logger.debug("Excluded: Empty or image-only")
            return {
                "should_use": False,
                "reason": "Empty or image-only",
                "content_type": "metadata",
                "quality_score": 0,
            }

        # 3. Content prefix exclusions
        for exclude in self.content_prefix_excludes:
            if text_lower.startswith(exclude):
                self.logger.debug(f"Excluded: Starts with '{exclude}'")
                return {
                    "should_use": False,
                    "reason": f"Starts with '{exclude}'",
                    "content_type": "other",
                    "quality_score": 0,
                }

        # 4. Content anywhere exclusions (check first 400 chars)
        for exclude in self.content_anywhere_excludes:
            if exclude in text_lower[:400]:
                self.logger.debug(f"Excluded: Contains '{exclude}'")
                return {
                    "should_use": False,
                    "reason": f"Contains '{exclude}'",
                    "content_type": "metadata",
                    "quality_score": 0,
                }

        # 5. Table of contents detection
        if self._is_table_of_contents(text):
            self.logger.debug("Excluded: Detected Table of Contents")
            return {
                "should_use": False,
                "reason": "Detected Table of Contents",
                "content_type": "toc",
                "quality_score": 0,
            }

        # 6. Low alphabetic content check
        alpha_ratio = self._alpha_ratio(text)
        if alpha_ratio < 0.30:
            # Check if it's a useful table
            if self._is_useful_table(text, metadata):
                self.logger.debug("Accepted: Useful data table")
                return {
                    "should_use": True,
                    "reason": "Useful data table",
                    "content_type": "table",
                    "quality_score": 70,
                }
            else:
                self.logger.debug(
                    f"Excluded: Low alphabetic content ({alpha_ratio:.2%})"
                )
                return {
                    "should_use": False,
                    "reason": f"Low alphabetic content ({alpha_ratio:.2%})",
                    "content_type": "metadata",
                    "quality_score": 20,
                }

        # 7. Calculate quality score for accepted chunks
        quality_score = self._calculate_quality_score(text, metadata, alpha_ratio)

        self.logger.debug(f"Accepted: Quality score {quality_score}")
        return {
            "should_use": True,
            "reason": "OK",
            "content_type": "main_content",
            "quality_score": quality_score,
        }

    def _calculate_quality_score(
        self, text: str, metadata: Dict[str, Any], alpha_ratio: float
    ) -> float:
        """
        Calculate a quality score for accepted chunks (0-100).

        Args:
            text (str): Chunk content
            metadata (dict): Chunk metadata
            alpha_ratio (float): Alphabetic character ratio

        Returns:
            float: Quality score (0-100)
        """
        score = 50  # Base score

        # Bonus for good length (200-1000 chars optimal)
        text_len = len(text)
        if 200 <= text_len <= 1000:
            score += 20
        elif 100 <= text_len < 200 or 1000 < text_len <= 2000:
            score += 10

        # Bonus for high alphabetic ratio
        if alpha_ratio > 0.7:
            score += 15
        elif alpha_ratio > 0.5:
            score += 10

        # Bonus for having meaningful header
        header = metadata.get("section_header", "")
        if header and len(header) > 3:
            score += 10

        # Bonus for complete sentences
        if text.strip().endswith((".", "!", "?")):
            score += 5

        return min(100, score)

    def evaluate_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple chunks and return statistics.

        Args:
            chunks (list): List of chunks with 'content' and 'metadata'

        Returns:
            dict: {
                'total_chunks': int,
                'accepted_chunks': int,
                'rejected_chunks': int,
                'rejection_reasons': dict,
                'content_type_distribution': dict,
                'average_quality_score': float,
                'evaluated_chunks': list  # Original chunks with evaluation added
            }
        """
        self.logger.info(f"Evaluating {len(chunks)} chunks...")

        total = len(chunks)
        accepted = 0
        rejected = 0
        rejection_reasons = {}
        content_types = {}
        quality_scores = []
        evaluated_chunks = []

        for chunk in chunks:
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            # Evaluate chunk
            evaluation = self.evaluate_chunk(content, metadata)

            # Add evaluation to chunk
            chunk_with_eval = chunk.copy()
            chunk_with_eval["evaluation"] = evaluation
            evaluated_chunks.append(chunk_with_eval)

            # Update statistics
            if evaluation["should_use"]:
                accepted += 1
                quality_scores.append(evaluation["quality_score"])
            else:
                rejected += 1
                reason = evaluation["reason"]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            content_type = evaluation["content_type"]
            content_types[content_type] = content_types.get(content_type, 0) + 1

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        stats = {
            "total_chunks": total,
            "accepted_chunks": accepted,
            "rejected_chunks": rejected,
            "acceptance_rate": (accepted / total * 100) if total > 0 else 0,
            "rejection_reasons": rejection_reasons,
            "content_type_distribution": content_types,
            "average_quality_score": round(avg_quality, 2),
            "evaluated_chunks": evaluated_chunks,
        }

        self.logger.info(
            f"Evaluation complete: {accepted}/{total} accepted ({stats['acceptance_rate']:.1f}%)"
        )

        return stats

    def filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter chunks, keeping only those that pass evaluation.

        Args:
            chunks (list): List of chunks to filter

        Returns:
            list: Filtered chunks (only accepted ones)
        """
        evaluation = self.evaluate_chunks(chunks)
        filtered = [
            chunk
            for chunk in evaluation["evaluated_chunks"]
            if chunk["evaluation"]["should_use"]
        ]

        self.logger.info(f"Filtered: {len(filtered)}/{len(chunks)} chunks retained")
        return filtered


# Usage example
if __name__ == "__main__":
    evaluator = ChunkEvaluator()

    # Example chunks
    test_chunks = [
        {
            "content": "This is a meaningful paragraph with actual content that discusses important topics.",
            "metadata": {"section_header": "Introduction", "section_level": 1},
        },
        {
            "content": "Table of Contents\nChapter 1 .................. 5\nChapter 2 .................. 10",
            "metadata": {"section_header": "Contents", "section_level": 1},
        },
        {
            "content": "<!-- image -->",
            "metadata": {"section_header": "", "section_level": 0},
        },
        {
            "content": "Copyright © 2024. All rights reserved.",
            "metadata": {"section_header": "Legal", "section_level": 1},
        },
        {
            "content": "This section explains the methodology used in the research.",
            "metadata": {"section_header": "Methodology", "section_level": 2},
        },
    ]

    # Evaluate all chunks
    stats = evaluator.evaluate_chunks(test_chunks)

    print("\n" + "=" * 80)
    print("EVALUATION STATISTICS")
    print("=" * 80)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Accepted: {stats['accepted_chunks']}")
    print(f"Rejected: {stats['rejected_chunks']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.1f}%")
    print(f"Average quality score: {stats['average_quality_score']}")

    print("\nRejection reasons:")
    for reason, count in stats["rejection_reasons"].items():
        print(f"  - {reason}: {count}")

    print("\nContent type distribution:")
    for ctype, count in stats["content_type_distribution"].items():
        print(f"  - {ctype}: {count}")

    # Filter chunks
    filtered = evaluator.filter_chunks(test_chunks)
    print(f"\nRetained {len(filtered)} high-quality chunks")
