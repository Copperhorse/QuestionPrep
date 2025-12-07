#!/usr/bin/env python3
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pymupdf4llm
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction and chunking"""

    max_chunk_tokens: int = 512
    min_tokens_for_chunk: int = 50
    mode: str = "single"  # "single" or "page"
    ignore_code: bool = False
    table_strategy: str = "lines"  # "lines_strict", "lines", or "text"
    pages_delimiter: str = "\n\f"


def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)"""
    return len(text) // 4


def split_by_sentences(text: str, max_tokens: int) -> List[str]:
    """Split text by sentences to stay under token limit"""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


class MarkdownCleaner:
    """Clean and normalize markdown text from PDF extraction"""

    @staticmethod
    def clean(text: str) -> str:
        """Apply all cleaning operations"""
        text = MarkdownCleaner._remove_excessive_newlines(text)
        text = MarkdownCleaner._fix_broken_headers(text)
        text = MarkdownCleaner._clean_tables(text)
        text = MarkdownCleaner._remove_artifacts(text)
        text = MarkdownCleaner._normalize_lists(text)
        text = MarkdownCleaner._fix_spacing(text)
        return text.strip()

    @staticmethod
    def _remove_excessive_newlines(text: str) -> str:
        """Replace 3+ newlines with 2 newlines"""
        return re.sub(r"\n{3,}", "\n\n", text)

    @staticmethod
    def _fix_broken_headers(text: str) -> str:
        """Fix headers that may have extra spaces or formatting issues"""
        # Fix headers with spaces between # and text
        text = re.sub(r"^(#{1,6})\s{2,}", r"\1 ", text, flags=re.MULTILINE)
        # Ensure headers have space after #
        text = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _clean_tables(text: str) -> str:
        """Clean up table formatting issues"""
        # Remove tables with only dashes/pipes (malformed)
        lines = text.split("\n")
        cleaned = []
        for i, line in enumerate(lines):
            # Skip lines that are only pipes and dashes/spaces
            if re.match(r"^[\|\-\s]+$", line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _remove_artifacts(text: str) -> str:
        """Remove common PDF extraction artifacts"""
        # Remove page numbers at start/end of lines
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove isolated single characters on lines
        text = re.sub(r"^\s*[a-zA-Z]\s*$", "", text, flags=re.MULTILINE)
        # Remove URLs that are broken across lines
        text = re.sub(r"http://\s+", "http://", text)
        text = re.sub(r"https://\s+", "https://", text)
        return text

    @staticmethod
    def _normalize_lists(text: str) -> str:
        """Normalize bullet points and numbered lists"""
        # Ensure consistent spacing for bullets
        text = re.sub(r"^[\*\-]\s+", "- ", text, flags=re.MULTILINE)
        # Fix numbered lists
        text = re.sub(r"^(\d+)\.\s{2,}", r"\1. ", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _fix_spacing(text: str) -> str:
        """Fix spacing issues"""
        # Remove trailing whitespace
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        # Ensure space after punctuation
        text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
        return text


class PyMuPDFEnhancedChunker:
    """
    Combined PyMuPDF4LLM extraction with enhanced markdown chunking.
    Extracts PDF to markdown, cleans it, creates intelligent chunks,
    and optionally saves chunks in several formats.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.cleaner = MarkdownCleaner()
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

    def load_and_chunk(
        self,
        file_path: str,
        password: Optional[str] = None,
        save_dir: Optional[str] = None,
        save_formats: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main method: Load PDF, extract markdown, clean, and chunk.
        Optionally save chunks to disk.

        Args:
            file_path: Path to PDF file or URL
            password: Optional password for encrypted PDFs
            save_dir: Directory where chunks will be saved. If None, no saving occurs.
            save_formats: Sequence of formats to save. Supported: 'jsonl', 'json', 'text', 'indiv'
                          - 'jsonl' writes one JSON object per line
                          - 'json' writes a single JSON array
                          - 'text' writes a single markdown file with all chunks appended
                          - 'indiv' writes individual chunk files (e.g., chunk_001.md)

        Returns:
            List of chunks with metadata and content
        """
        try:
            # Step 1: Load PDF using PyMuPDF4LLM
            logger.info(f"Loading PDF from {file_path}")
            loader = PyMuPDF4LLMLoader(
                file_path=file_path,
                password=password,
                mode=self.config.mode,
                ignore_code=self.config.ignore_code,
                table_strategy=self.config.table_strategy,
                pages_delimiter=self.config.pages_delimiter,
            )
            docs = loader.load()

            if not docs:
                raise ValueError("No content extracted from PDF")

            # Step 2: Get markdown content
            markdown_text = docs[0].page_content
            pdf_metadata = docs[0].metadata

            logger.info(f"Extracted {len(markdown_text)} characters from PDF")

            # Step 3: Clean the markdown
            cleaned_markdown = self.cleaner.clean(markdown_text)
            logger.info("Cleaned markdown text")

            # Step 4: Chunk the cleaned markdown
            chunks = self._chunk_markdown(cleaned_markdown, pdf_metadata)

            logger.info(f"Created {len(chunks)} chunks")

            # Optional: save chunks
            if save_dir and save_formats:
                try:
                    self._save_chunks(
                        chunks,
                        save_dir=save_dir,
                        formats=list(save_formats),
                        source_path=file_path,
                    )
                    logger.info(f"Chunks saved to {save_dir}")
                except Exception as save_exc:
                    logger.exception("Failed to save chunks")
                    # Do not raise; chunk creation succeeded even if saving failed.

            return chunks

        except Exception as e:
            logger.exception("Failed to load and chunk PDF")
            raise RuntimeError(f"PDF processing failed: {str(e)}") from e

    def _chunk_markdown(
        self, markdown_text: str, pdf_metadata: Dict
    ) -> List[Dict[str, Any]]:
        """Chunk cleaned markdown into semantic sections"""
        try:
            # 1. Split by headers
            header_docs = self.splitter.split_text(markdown_text)

            # 2. Token-aware normalization
            token_chunks = self._token_split_and_normalize(header_docs)

            # 3. Join short chunks
            merged = self._join_short_chunks(token_chunks)

            # 4. Assign sequential IDs, links, and enrich with PDF metadata
            for i, chunk in enumerate(merged, start=1):
                chunk["chunk_id"] = str(i)
                chunk["prev_chunk_id"] = str(i - 1) if i > 1 else None
                chunk["next_chunk_id"] = str(i + 1) if i < len(merged) else None

                # Merge PDF metadata with chunk metadata
                chunk["metadata"].update(
                    {
                        # Document-level metadata
                        "document_title": pdf_metadata.get("title", ""),
                        "document_author": pdf_metadata.get("author", ""),
                        "document_source": pdf_metadata.get("source", ""),
                        "total_pages": pdf_metadata.get("total_pages", 0),
                        "creation_date": pdf_metadata.get("creationdate")
                        or pdf_metadata.get("creationDate", ""),
                        "producer": pdf_metadata.get("producer", ""),
                        "format": pdf_metadata.get("format", ""),
                        # Full metadata for reference
                        "pdf_metadata": pdf_metadata,
                    }
                )

            return merged

        except Exception as e:
            logger.exception("Chunking failed")
            raise RuntimeError("Markdown chunking failed") from e

    def _token_split_and_normalize(self, docs) -> List[Dict[str, Any]]:
        """
        Split documents into chunks respecting token limits and header hierarchy.
        """
        final: List[Dict[str, Any]] = []
        prev_headers = {"Header 1": None, "Header 2": None, "Header 3": None}
        idx = 1
        char_position = 0

        for doc in docs:
            content = getattr(doc, "page_content", "") or ""
            md = getattr(doc, "metadata", {}) or {}

            # Update header context
            if "Header 1" in md:
                prev_headers["Header 1"] = md["Header 1"]
                prev_headers["Header 2"] = None
                prev_headers["Header 3"] = None
            if "Header 2" in md:
                prev_headers["Header 2"] = md["Header 2"]
                prev_headers["Header 3"] = None
            if "Header 3" in md:
                prev_headers["Header 3"] = md["Header 3"]

            # Determine section hierarchy
            section_header = (
                md.get("Header 3") or md.get("Header 2") or md.get("Header 1")
            )
            if "Header 3" in md:
                section_level = 3
                parent_section = prev_headers["Header 2"]
            elif "Header 2" in md:
                section_level = 2
                parent_section = prev_headers["Header 1"]
            elif "Header 1" in md:
                section_level = 1
                parent_section = None
            else:
                section_level = None
                parent_section = None

            base_md = {
                "section_header": section_header,
                "parent_section": parent_section,
                "section_level": section_level,
            }

            # Check if content fits in one chunk
            content_tokens = count_tokens(content)
            if content_tokens <= self.config.max_chunk_tokens:
                final.append(
                    {
                        "orig_index": idx,
                        "content": content,
                        "metadata": base_md,
                        "estimated_tokens": content_tokens,
                        "char_start": char_position,
                        "char_end": char_position + len(content),
                    }
                )
                idx += 1
            else:
                # Split large sections by sentences
                subs = split_by_sentences(content, self.config.max_chunk_tokens)
                sub_pos = char_position
                for sub in subs:
                    final.append(
                        {
                            "orig_index": idx,
                            "content": sub,
                            "metadata": base_md,
                            "estimated_tokens": count_tokens(sub),
                            "char_start": sub_pos,
                            "char_end": sub_pos + len(sub),
                        }
                    )
                    sub_pos += len(sub)
                    idx += 1

            char_position += len(content)

        return final

    def _join_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge chunks that are too short or header-only with their neighbors.
        """
        merged: List[Dict[str, Any]] = []
        i = 0

        while i < len(chunks):
            cur = chunks[i]
            content = (cur.get("content") or "").strip()
            md = cur.get("metadata") or {}
            est_tokens = cur.get("estimated_tokens", count_tokens(content))

            header_text = md.get("section_header") or ""
            header_only = (
                bool(header_text)
                and len(content) < max(50, len(header_text) * 2)
                and header_text.strip().lower() in content.strip().lower()
            )
            too_short = est_tokens < self.config.min_tokens_for_chunk

            if (header_only or too_short) and (i + 1) < len(chunks):
                # Merge with next chunk
                nxt = chunks[i + 1]
                merged_text = (
                    content + "\n\n" + (nxt.get("content") or "").strip()
                ).strip()
                nxt["content"] = merged_text
                nxt["estimated_tokens"] = count_tokens(merged_text)
                i += 1
            else:
                # Clean up temporary fields before adding
                cur.pop("char_start", None)
                cur.pop("char_end", None)
                merged.append(cur)
                i += 1

        return merged

    # ---------------------------
    # Saving / serialization
    # ---------------------------
    def _sanitize_filename(self, name: str, fallback: str = "document") -> str:
        """Sanitize a string to be safe for filenames."""
        if not name:
            name = fallback
        # Remove path unsafe characters
        name = re.sub(r"[^\w\s-]", "", name).strip()
        # Replace spaces and repeated dashes with single underscore
        name = re.sub(r"[-\s]+", "_", name)
        return name[:200]  # limit length

    def _save_chunks(
        self,
        chunks: List[Dict[str, Any]],
        save_dir: str,
        formats: List[str],
        source_path: Optional[str] = None,
    ) -> None:
        """
        Save chunks to disk in requested formats.

        formats: list containing any of {'jsonl', 'json', 'text', 'indiv'}
        - 'jsonl': one JSON object per line
        - 'json': array of JSON objects
        - 'text': single markdown file with all chunks concatenated (with separators)
        - 'indiv': individual markdown files per chunk (e.g., chunk_001.md)
        """
        allowed = {"jsonl", "json", "text", "indiv"}
        requested = [f.lower() for f in formats if f and f.lower() in allowed]
        if not requested:
            logger.debug("No valid save formats requested; skipping saving.")
            return

        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Create a base filename from document title or source
        doc_title = ""
        if chunks and chunks[0].get("metadata"):
            doc_title = chunks[0]["metadata"].get("document_title") or ""
        if not doc_title and source_path:
            doc_title = Path(source_path).stem
        base_name = self._sanitize_filename(
            doc_title or f"extracted_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        # Prepare serializable representations (avoid non-serializable items)
        serializable_chunks = []
        for ch in chunks:
            # Shallow copy to avoid mutating original
            c = {
                "chunk_id": ch.get("chunk_id"),
                "orig_index": ch.get("orig_index"),
                "content": ch.get("content"),
                "estimated_tokens": ch.get("estimated_tokens"),
                "metadata": ch.get("metadata"),
                "prev_chunk_id": ch.get("prev_chunk_id"),
                "next_chunk_id": ch.get("next_chunk_id"),
            }
            serializable_chunks.append(c)

        if "jsonl" in requested:
            jsonl_file = out_path / f"{base_name}.jsonl"
            with jsonl_file.open("w", encoding="utf-8") as fh:
                for c in serializable_chunks:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            logger.debug("Saved JSONL to %s", jsonl_file)

        if "json" in requested:
            json_file = out_path / f"{base_name}.json"
            with json_file.open("w", encoding="utf-8") as fh:
                json.dump(serializable_chunks, fh, ensure_ascii=False, indent=2)
            logger.debug("Saved JSON to %s", json_file)

        if "text" in requested:
            text_file = out_path / f"{base_name}.md"
            separator = "\n\n---\n\n"
            with text_file.open("w", encoding="utf-8") as fh:
                for idx, c in enumerate(serializable_chunks, start=1):
                    header = f"<!-- chunk: {c.get('chunk_id') or idx} | estimated_tokens: {c.get('estimated_tokens')} -->\n"
                    fh.write(header)
                    fh.write((c.get("content") or "").strip() + "\n")
                    if idx != len(serializable_chunks):
                        fh.write(separator)
            logger.debug("Saved aggregated markdown to %s", text_file)

        if "indiv" in requested:
            indiv_dir = out_path / f"{base_name}_chunks"
            indiv_dir.mkdir(parents=True, exist_ok=True)
            pad = max(3, len(str(len(serializable_chunks))))
            for c in serializable_chunks:
                cid = c.get("chunk_id") or c.get("orig_index")
                fname = f"chunk_{str(cid).zfill(pad)}.md"
                fpath = indiv_dir / fname
                with fpath.open("w", encoding="utf-8") as fh:
                    # Include minimal metadata at top of file as YAML-like comment
                    meta = c.get("metadata") or {}
                    fh.write(
                        f"<!--\nchunk_id: {c.get('chunk_id')}\nestimated_tokens: {c.get('estimated_tokens')}\nmetadata: {json.dumps(meta, ensure_ascii=False)}\n-->\n\n"
                    )
                    fh.write((c.get("content") or "").strip() + "\n")
            logger.debug("Saved individual chunk files to %s", indiv_dir)


# Example usage
if __name__ == "__main__":
    import pprint
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize chunker with custom config
    config = ExtractionConfig(
        max_chunk_tokens=512,
        min_tokens_for_chunk=50,
        mode="single",
        ignore_code=False,
        table_strategy="lines",
    )

    chunker = PyMuPDFEnhancedChunker(config)

    # Load and chunk PDF
    # You can pass save_dir and save_formats to persist results locally
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "/home/copper/Documents/2510.04871v1.pdf"

    save_dir = "/home/copper/Documents/pdf_chunks_output"
    save_formats = ["jsonl", "json", "text", "indiv"]

    chunks = chunker.load_and_chunk(
        pdf_path, save_dir=save_dir, save_formats=save_formats
    )

    # Display results
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\n" + "=" * 50)

    # Show first chunk with full metadata
    if chunks:
        print("\nFirst chunk metadata:")
        pprint.pprint(
            {
                "chunk_id": chunks[0]["chunk_id"],
                "document_title": chunks[0]["metadata"]["document_title"],
                "document_author": chunks[0]["metadata"]["document_author"],
                "total_pages": chunks[0]["metadata"]["total_pages"],
                "section_header": chunks[0]["metadata"]["section_header"],
                "section_level": chunks[0]["metadata"]["section_level"],
                "parent_section": chunks[0]["metadata"]["parent_section"],
                "estimated_tokens": chunks[0]["estimated_tokens"],
            }
        )
        print("\n" + "=" * 50)
        print("\nChunk content preview:")
        print(chunks[0]["content"][:1000])
        print("\n" + "=" * 50)
        print("\nFull PDF metadata:")
        pprint.pprint(chunks[0]["metadata"].get("pdf_metadata", {}))
