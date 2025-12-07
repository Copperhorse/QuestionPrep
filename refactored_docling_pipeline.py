# refactored_docling_pipeline.py
# Clean, single-file refactor of your Docling extraction + chunking pipeline
# - keeps docling usage and picture-description annotations (no image saving)
# - header-first chunking, token-aware splitting, join-short-chunks
# - chunk metadata includes file_name and file_id, and prev_chunk_id/next_chunk_id
# - structured logging, exception chaining, validation
# - modular classes but single-file for now

from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    smolvlm_picture_description,
)

# Docling imports (must be installed in environment)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import CodeItem, PictureItem, TextItem

# LangChain splitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field, field_validator

# tiktoken optional (token counting)
try:
    import tiktoken  # type: ignore

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None

# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger("docling_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(ch)

# Reduce noisy libs
logging.getLogger("rapidocr").setLevel(logging.WARNING)
logging.getLogger("modelscope").setLevel(logging.WARNING)


# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class ExtractionConfig:
    extract_images: bool = False  # we won't save images, kept for compatibility
    extract_formulas: bool = True
    extract_code: bool = False
    extract_descriptions: bool = False  # picture description (smolvlm)
    output_dir: str = "extracted_content"
    images_scale: float = 1.0
    generate_page_images: bool = False
    generate_picture_images: bool = False
    pipeline_version: str = "v1"

    # chunking
    max_chunk_tokens: int = 1024
    min_tokens_for_chunk: int = 50


# ---------------------------
# Output models
# ---------------------------
class Formula(BaseModel):
    text: Optional[str]
    latex: Optional[str]


class CodeBlock(BaseModel):
    code: str
    language: str = "unknown"


class ImageAnnotation(BaseModel):
    self_ref: Optional[str] = None
    caption: Optional[str] = None
    descriptions: List[Dict[str, Any]] = Field(default_factory=list)
    primary_description: Optional[str] = None

    @field_validator("descriptions")
    def ensure_list(cls, v):
        return v or []


class ExtractionResult(BaseModel):
    file_name: str
    markdown: str = ""
    formulas: List[Formula] = Field(default_factory=list)
    code_blocks: List[CodeBlock] = Field(default_factory=list)
    image_annotations: List[ImageAnnotation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------
# Utilities
# ---------------------------
def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOKENIZER is None:
        # fallback: 1 token ≈ 4 characters (rough)
        return max(1, len(text) // 4)
    return len(_TOKENIZER.encode(text))


def split_by_sentences(text: str, max_tokens: int) -> List[str]:
    """
    Sentence-aware splitting with hard fallback.
    Avoids splitting inside code fences by a simple heuristic (triple-backticks).
    """
    if not text:
        return []
    # protect code fences: replace newlines inside ``` ``` with placeholder
    code_fences = []

    def _protect_code(match):
        code = match.group(0)
        token = f"@@CODE{len(code_fences)}@@"
        code_fences.append(code)
        return token

    protected = re.sub(r"```.*?```", _protect_code, text, flags=re.DOTALL)
    # split by sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", protected)
    chunks, current = [], []
    for p in parts:
        candidate = " ".join(current + [p]).strip()
        if not candidate:
            continue
        if count_tokens(candidate) > max_tokens and current:
            chunks.append(" ".join(current).strip())
            current = [p]
        else:
            current.append(p)
    if current:
        chunks.append(" ".join(current).strip())

    # restore code fences and hard-split too-large chunks
    out: List[str] = []
    for c in chunks:
        for i, code in enumerate(code_fences):
            token = f"@@CODE{i}@@"
            c = c.replace(token, code)
        if count_tokens(c) > max_tokens:
            max_chars = max_tokens * 4
            for i in range(0, len(c), max_chars):
                out.append(c[i : i + max_chars])
        else:
            out.append(c)
    return out


def _validate_file_path(file_path: str) -> Path:
    p = Path(file_path)
    if not p.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")
    if not p.is_file():
        logger.error("Path is not a file: %s", file_path)
        raise ValueError(f"Path is not a file: {file_path}")
    return p


def _get_extension(file_path: str) -> str:
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext not in ("pdf", "txt"):
        logger.error("Unsupported file extension: %s", ext)
        raise ValueError(f"Unsupported file extension: {ext}")
    return ext


def _validate_picture_description_config():
    if smolvlm_picture_description is None:
        logger.error("smolvlm_picture_description is not configured")
        raise RuntimeError("smolvlm_picture_description is not configured")
    if not hasattr(smolvlm_picture_description, "prompt"):
        logger.warning(
            "smolvlm_picture_description does not expose a 'prompt' attribute"
        )


# ---------------------------
# Extractor (stateless)
# ---------------------------
class EnhancedDoclingExtractor:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Extractor initialized with config: %s", self.config)

    def extract(self, file_path: str) -> ExtractionResult:
        try:
            p = _validate_file_path(file_path)
            ext = _get_extension(file_path)
            file_name = p.stem
            if ext == "txt":
                markdown = self._extract_text(p)
                return ExtractionResult(
                    file_name=file_name, markdown=markdown, metadata={"source": str(p)}
                )
            return self._extract_pdf(p)
        except Exception as e:
            logger.exception("Extraction failed for file: %s", file_path)
            raise RuntimeError(f"Extraction failed for file {file_path}") from e

    def _extract_text(self, path: Path) -> str:
        try:
            logger.debug("Reading TXT file: %s", path)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.exception("Failed to read TXT file: %s", path)
            raise IOError(f"Failed to read TXT file: {path}") from e

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        try:
            logger.info("Starting PDF extraction: %s", path)
            if self.config.extract_descriptions:
                _validate_picture_description_config()

            pipeline_options = PdfPipelineOptions()
            if self.config.extract_formulas:
                pipeline_options.do_formula_enrichment = True
            if self.config.extract_code:
                pipeline_options.do_code_enrichment = True
            if self.config.extract_images:
                pipeline_options.images_scale = self.config.images_scale
                pipeline_options.generate_page_images = self.config.generate_page_images
                pipeline_options.generate_picture_images = (
                    self.config.generate_picture_images
                )
            if self.config.extract_descriptions:
                pipeline_options.do_picture_description = True
                pipeline_options.picture_description_options = (
                    smolvlm_picture_description
                )
                # try to set custom prompt if mutable
                try:
                    pipeline_options.picture_description_options.prompt = (
                        "Provide a clear and concise technical description of the picture. "
                        "If it is a chart or diagram, describe its type and what it conveys. "
                        "Do not speculate beyond visible information."
                    )
                except Exception:
                    logger.debug(
                        "Could not set prompt on picture_description_options (read-only)"
                    )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.debug("Running converter.convert() for file: %s", path)
            conversion_result = converter.convert(str(path))
            doc = conversion_result.document

            markdown_text = (
                doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
            )
            # Clean up common OCR/PDF artifacts before chunking
            markdown_text = self._normalize_markdown(markdown_text)
            structured = self._extract_structured_content(doc)

            formulas = [Formula(**f) for f in structured.get("formulas", [])]
            code_blocks = [CodeBlock(**c) for c in structured.get("code_blocks", [])]
            image_annotations = [
                ImageAnnotation(**ia) for ia in structured.get("image_annotations", [])
            ]

            metadata = {
                "file_name": path.name,
                "pipeline_version": self.config.pipeline_version,
            }
            result = ExtractionResult(
                file_name=path.stem,
                markdown=markdown_text,
                formulas=formulas,
                code_blocks=code_blocks,
                image_annotations=image_annotations,
                metadata=metadata,
            )
            logger.info(
                "PDF extraction completed: %s (formulas=%d, code_blocks=%d, image_annotations=%d)",
                path,
                len(formulas),
                len(code_blocks),
                len(image_annotations),
            )
            return result
        except Exception as e:
            logger.exception("PDF extraction failed: %s", path)
            raise RuntimeError(f"PDF extraction failed for {path}") from e

    def _extract_structured_content(self, doc) -> Dict[str, List[Dict[str, Any]]]:
        formulas, code_blocks, image_annotations = [], [], []
        try:
            for element, _level in doc.iterate_items():
                # formulas
                if (
                    isinstance(element, TextItem)
                    and getattr(element, "label", None) == "FORMULA"
                ):
                    formulas.append(
                        {
                            "text": getattr(element, "text", None),
                            "latex": getattr(element, "latex", None),
                        }
                    )
                # code
                elif isinstance(element, CodeItem):
                    code_text = getattr(element, "text", "")
                    code_blocks.append(
                        {
                            "code": code_text.strip("\n"),
                            "language": getattr(element, "code_language", "unknown"),
                        }
                    )
                # picture annotations (no image saving)
                elif isinstance(element, PictureItem):
                    caption = None
                    try:
                        if hasattr(element, "caption_text"):
                            cap = element.caption_text(doc=doc)
                            caption = cap if isinstance(cap, str) else None
                    except Exception:
                        logger.debug("Failed to get caption for a picture")

                    descriptions: List[Dict[str, Any]] = []
                    meta = getattr(element, "meta", None)
                    if meta is not None:
                        pd_list = getattr(meta, "picture_descriptions", None)
                        if pd_list:
                            for pd in pd_list:
                                try:
                                    descriptions.append(
                                        {
                                            "text": getattr(pd, "text", None),
                                            "provenance": getattr(
                                                pd, "provenance", None
                                            ),
                                        }
                                    )
                                except Exception:
                                    logger.debug(
                                        "Skipping malformed picture_description entry"
                                    )
                    else:
                        ann_list = getattr(element, "annotations", None)
                        if ann_list:
                            for ann in ann_list:
                                try:
                                    text_val = getattr(ann, "text", None)
                                    provenance_val = getattr(ann, "provenance", None)
                                    if text_val is not None:
                                        descriptions.append(
                                            {
                                                "text": text_val,
                                                "provenance": provenance_val,
                                            }
                                        )
                                except Exception:
                                    logger.debug("Skipping malformed annotation entry")

                    primary_description = (
                        descriptions[0]["text"] if descriptions else None
                    )
                    image_annotations.append(
                        {
                            "self_ref": getattr(element, "self_ref", None),
                            "caption": caption,
                            "descriptions": descriptions,
                            "primary_description": primary_description,
                        }
                    )

            return {
                "formulas": formulas,
                "code_blocks": code_blocks,
                "image_annotations": image_annotations,
            }
        except Exception as e:
            logger.exception("Failed to extract structured content")
            raise RuntimeError("Failed to extract structured content") from e

    def _normalize_markdown(self, md: str) -> str:
        """Remove HTML comments, page numbers and normalize whitespace to reduce noisy chunks."""
        if not md:
            return md
        # remove HTML comments like <!-- image -->
        md = re.sub(r"<!--.*?-->", " ", md, flags=re.DOTALL)
        # remove 'Page 1' style footers/headers (lines that are just numbers or 'Page X')
        md = re.sub(r"(?m)^\s*(page\s*\d+|\d{1,4})\s*$", "", md, flags=re.IGNORECASE)
        # collapse repeated dots or hyphens often found in TOC or separators
        md = re.sub(r"[.]{3,}", " ... ", md)
        md = re.sub(r"[-]{3,}", "\n", md)
        # collapse multiple blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()


# ---------------------------
# Chunker (header-first, token-split, join-short)
# ---------------------------
class EnhancedMarkdownChunker:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

    def split_with_annotations(
        self, markdown_text: str, image_annotations: List[ImageAnnotation]
    ) -> List[Dict[str, Any]]:
        try:
            # 1. Header split
            header_docs = self.splitter.split_text(markdown_text)

            # 2. Token-aware normalization (preserve header context)
            token_chunks = self._token_split_and_normalize(header_docs)

            # 3. Associate image annotations (IMPROVED - no duplication)
            annotated = self._match_annotations_to_chunks(
                token_chunks, image_annotations, markdown_text
            )

            # 4. Join short chunks (header-only or too short)
            merged = self._join_short_chunks(annotated)

            # 5. Assign sequential chunk_id and prev/next links
            for i, c in enumerate(merged, start=1):
                c["chunk_id"] = str(i)
                c["prev_chunk_id"] = str(i - 1) if i > 1 else None
                c["next_chunk_id"] = str(i + 1) if i < len(merged) else None

            logger.debug("Chunker produced %d chunks", len(merged))
            return merged
        except Exception as e:
            logger.exception("Chunking with annotations failed")
            raise RuntimeError("Chunking with annotations failed") from e

    def _token_split_and_normalize(self, docs) -> List[Dict[str, Any]]:
        """
        For each header section (doc):
         - determine section_header, section_level, parent_section using previous headers
         - split into sub-chunks if tokens exceed max_chunk_tokens
         - track character positions for image matching
        """
        final: List[Dict[str, Any]] = []
        prev_headers = {"Header 1": None, "Header 2": None, "Header 3": None}
        idx = 1
        char_position = 0  # Track position in original markdown

        for doc in docs:
            content = getattr(doc, "page_content", "") or ""
            md = getattr(doc, "metadata", {}) or {}

            # Update prev_headers
            if "Header 1" in md:
                prev_headers["Header 1"] = md["Header 1"]
                prev_headers["Header 2"] = None
                prev_headers["Header 3"] = None
            if "Header 2" in md:
                prev_headers["Header 2"] = md["Header 2"]
                prev_headers["Header 3"] = None
            if "Header 3" in md:
                prev_headers["Header 3"] = md["Header 3"]

            section_header = (
                md.get("Header 3") or md.get("Header 2") or md.get("Header 1")
            )
            if "Header 3" in md:
                section_level = 3
            elif "Header 2" in md:
                section_level = 2
            elif "Header 1" in md:
                section_level = 1
            else:
                section_level = None

            if section_level == 3:
                parent_section = prev_headers["Header 2"]
            elif section_level == 2:
                parent_section = prev_headers["Header 1"]
            else:
                parent_section = None

            base_md = {
                "section_header": section_header,
                "parent_section": parent_section,
                "section_level": section_level,
            }

            if count_tokens(content) <= self.config.max_chunk_tokens:
                final.append(
                    {
                        "orig_index": idx,
                        "content": content,
                        "metadata": base_md,
                        "estimated_tokens": count_tokens(content),
                        "char_start": char_position,
                        "char_end": char_position + len(content),
                    }
                )
                idx += 1
            else:
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

    def _match_annotations_to_chunks(
        self,
        chunks: List[Dict[str, Any]],
        annotations: List[ImageAnnotation],
        full_markdown: str,
    ) -> List[Dict[str, Any]]:
        """
        Improved image annotation matching:
        1. Find each image's position in the full markdown
        2. Assign to the chunk that contains that position
        3. No duplication across chunks
        """
        # Initialize empty annotation lists
        for chunk in chunks:
            chunk["image_annotations"] = []

        if not annotations:
            return chunks

        # Track which annotations have been assigned
        assigned_annotations = set()

        for ann in annotations:
            # Strategy 1: Try to find position using self_ref (e.g., "#/pictures/0")
            position = -1

            if ann.self_ref:
                # Look for image marker with this ref
                # Docling often puts <!-- image --> or similar markers
                ref_pattern = re.escape(ann.self_ref)
                match = re.search(ref_pattern, full_markdown)
                if match:
                    position = match.start()

            # Strategy 2: Use caption text to find position
            if position == -1 and ann.caption:
                caption_escaped = re.escape(
                    ann.caption[:100]
                )  # First 100 chars to avoid huge regexes
                match = re.search(caption_escaped, full_markdown, re.IGNORECASE)
                if match:
                    position = match.start()

            # Strategy 3: Look for image markers (<!-- image -->, ![...])
            if position == -1:
                # Find all image markers
                image_markers = list(
                    re.finditer(
                        r"<!--\s*image\s*-->|!\[.*?\]\(.*?\)",
                        full_markdown,
                        re.IGNORECASE,
                    )
                )

                # Try to match by index (assuming annotations are in order)
                ann_index = annotations.index(ann)
                if ann_index < len(image_markers):
                    position = image_markers[ann_index].start()

            # Assign to the appropriate chunk based on position
            if position >= 0:
                for chunk in chunks:
                    char_start = chunk.get("char_start", 0)
                    char_end = chunk.get("char_end", float("inf"))

                    if char_start <= position < char_end:
                        chunk["image_annotations"].append(ann.model_dump())
                        assigned_annotations.add(id(ann))
                        break
            else:
                # Fallback: content-based matching (but only if not already assigned)
                if id(ann) not in assigned_annotations:
                    self._fallback_content_matching(chunks, ann, assigned_annotations)

        # Clean up temporary fields
        for chunk in chunks:
            chunk.pop("char_start", None)
            chunk.pop("char_end", None)

        return chunks

    def _fallback_content_matching(
        self,
        chunks: List[Dict[str, Any]],
        ann: ImageAnnotation,
        assigned_annotations: set,
    ):
        """
        Fallback matching when position-based matching fails.
        Only assigns to the FIRST matching chunk to avoid duplication.
        """
        for chunk in chunks:
            content = chunk.get("content", "") or ""
            content_lower = content.lower()

            # Check for image markers
            has_marker = "<!-- image -->" in content_lower or "![" in content_lower

            # Check for caption match
            caption_match = False
            if ann.caption:
                # Use a substring of caption (first 50 chars) to avoid false negatives
                caption_substr = ann.caption[:50].lower()
                caption_match = caption_substr in content_lower

            # Check for self_ref match
            ref_match = False
            if ann.self_ref:
                ref_match = ann.self_ref.lower() in content_lower

            # If any match found, assign and stop
            if has_marker or caption_match or ref_match:
                chunk["image_annotations"].append(ann.model_dump())
                assigned_annotations.add(id(ann))
                break  # IMPORTANT: Only assign to first matching chunk

    def _join_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Join chunks that are:
         - "header-only" (content shorter than the header string)
         - too short token-wise (< min_tokens_for_chunk)
        When merging, combine annotations without duplication.
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
                # Merge into next chunk
                nxt = chunks[i + 1]
                merged_text = (
                    content + "\n\n" + (nxt.get("content") or "").strip()
                ).strip()
                nxt["content"] = merged_text
                nxt["estimated_tokens"] = count_tokens(merged_text)

                # Merge annotations WITHOUT duplication
                nxt_ann = nxt.get("image_annotations", [])
                cur_ann = cur.get("image_annotations", [])

                # Deduplicate based on self_ref
                combined_ann = cur_ann.copy()
                existing_refs = {
                    a.get("self_ref") for a in cur_ann if a.get("self_ref")
                }

                for ann in nxt_ann:
                    ann_ref = ann.get("self_ref")
                    if not ann_ref or ann_ref not in existing_refs:
                        combined_ann.append(ann)
                        if ann_ref:
                            existing_refs.add(ann_ref)

                nxt["image_annotations"] = combined_ann
                i += 1
            else:
                merged.append(cur)
                i += 1

        return merged


# ---------------------------
# Rule-based evaluator (always used; LLM evaluator can be added externally)
# ---------------------------
class RuleBasedEvaluator:
    def __init__(self):
        self.metadata_excludes = [
            "table of contents",
            "references",
            "bibliography",
            "acknowledgment",
            "acknowledgement",
            "appendix",
            "author",
        ]
        self.content_prefix_excludes = ["the image is", "<!-- image -->"]
        self.content_anywhere_excludes = [
            "copyright",
            "doi:",
            "contact@",
            "http://",
            "https://",
            "www.",
        ]

    def _alpha_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        alpha = sum(1 for c in text if c.isalpha())
        return alpha / max(1, len(text))

    def _is_useful_table(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Detect if this is a data table (useful) vs formatting table (not useful)"""
        lines = text.split("\n")

        # Count table rows (lines with |)
        table_rows = sum(1 for line in lines if "|" in line)

        # If it's mostly table AND has a meaningful header, it's useful
        header = metadata.get("section_header", "").lower()

        # Table of contents indicators
        if any(x in header for x in ["table of contents", "contents"]):
            return False

        # If it has substantial table content (>5 rows) and meaningful header, keep it
        if table_rows > 5 and header and header not in ["", "##", "###"]:
            return True

        return False

    def evaluate(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        text = (content or "").strip()
        text_lower = text.lower()
        header_text = ""
        for k in ("section_header", "Header 3", "Header 2", "Header 1"):
            if metadata.get(k):
                header_text = (metadata.get(k) or "").lower()
                break

        # quick header-based exclusions
        for ex in self.metadata_excludes:
            if ex in header_text:
                return {
                    "should_use": False,
                    "reason": f"Header contains '{ex}'",
                    "content_type": "metadata",
                }

        if len(text) < 30 or text_lower in ["", "<!-- image -->"]:
            return {
                "should_use": False,
                "reason": "Empty or image-only",
                "content_type": "metadata",
            }

        for ex in self.content_prefix_excludes:
            if text_lower.startswith(ex):
                return {
                    "should_use": False,
                    "reason": f"Starts with '{ex}'",
                    "content_type": "other",
                }

        for ex in self.content_anywhere_excludes:
            if ex in text_lower[:400]:
                return {
                    "should_use": False,
                    "reason": f"Contains '{ex}'",
                    "content_type": "metadata",
                }

        # Detect TOC-like structure: many lines with dots and trailing page numbers
        lines = text.splitlines()
        toc_like = (
            sum(1 for l in lines if re.search(r"\.{3,}\s*\d{1,4}$", l.strip())) > 3
            or "table of contents" in text_lower[:400]
        )
        if toc_like:
            return {
                "should_use": False,
                "reason": "Detected Table of Contents",
                "content_type": "toc",
            }

        alpha_ratio = self._alpha_ratio(text)
        if alpha_ratio < 0.30:
            if self._is_useful_table(text, metadata):
                return {
                    "should_use": True,
                    "reason": "Useful data table",
                    "content_type": "table",
                }
            return {
                "should_use": False,
                "reason": "Low alphabetic content",
                "content_type": "metadata",
            }

        return {"should_use": True, "reason": "OK", "content_type": "main_content"}


# ---------------------------
# Orchestrator
# ---------------------------
class EnhancedOrchestrator:
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        evaluator: Optional[RuleBasedEvaluator] = None,
    ):
        self.config = config or ExtractionConfig()
        self.extractor = EnhancedDoclingExtractor(config=self.config)
        self.chunker = EnhancedMarkdownChunker(config=self.config)
        self.evaluator = evaluator or RuleBasedEvaluator()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_file(
        self, file_path: str, save_local: bool = True, upload_to_supabase: bool = False
    ) -> Dict[str, Any]:
        """
        Main pipeline:
        - extract (docling)
        - chunk (header-first, token-split, join-short)
        - annotate chunks with file metadata and chunk prev/next
        - evaluate each chunk (rule-based)
        - optionally save locally or upload (uploader is intentionally external)
        """
        try:
            extraction: ExtractionResult = self.extractor.extract(file_path)
            file_id = str(uuid.uuid4())
            file_name = extraction.file_name + Path(file_path).suffix

            # chunking
            raw_chunks = self.chunker.split_with_annotations(
                extraction.markdown, extraction.image_annotations
            )

            # add file-level metadata and chunk-level metadata and run evaluator
            processed_chunks = []
            for c in raw_chunks:
                # add file metadata
                c.setdefault("metadata", {})
                c["metadata"]["file_id"] = file_id
                c["metadata"]["file_name"] = file_name

                # run evaluator (rule-based always)
                eval_res = self.evaluator.evaluate(c.get("content", ""), c["metadata"])
                c["should_use"] = eval_res["should_use"]
                c["evaluation_reason"] = eval_res["reason"]
                c["content_type"] = eval_res.get("content_type", "unknown")
                processed_chunks.append(c)

            # re-assign chunk ids and prev/next (in case evaluator changed ordering not expected)
            for i, c in enumerate(processed_chunks, start=1):
                c["chunk_id"] = f"{file_id}-{i}"
                c["prev_chunk_id"] = f"{file_id}-{i - 1}" if i > 1 else None
                c["next_chunk_id"] = (
                    f"{file_id}-{i + 1}" if i < len(processed_chunks) else None
                )
            # optional save locally
            if save_local:
                out_dir = self.output_dir / extraction.file_name
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "markdown.md").write_text(
                    extraction.markdown or "", encoding="utf-8"
                )
                with open(out_dir / "chunks.json", "w", encoding="utf-8") as fh:
                    json.dump(processed_chunks, fh, ensure_ascii=False, indent=2)
                # save structured extras
                if extraction.formulas:
                    with open(out_dir / "formulas.json", "w", encoding="utf-8") as fh:
                        json.dump(
                            [f.model_dump() for f in extraction.formulas],
                            fh,
                            ensure_ascii=False,
                            indent=2,
                        )
                if extraction.code_blocks:
                    with open(
                        out_dir / "code_blocks.json", "w", encoding="utf-8"
                    ) as fh:
                        json.dump(
                            [c.model_dump() for c in extraction.code_blocks],
                            fh,
                            ensure_ascii=False,
                            indent=2,
                        )
                if extraction.image_annotations:
                    with open(
                        out_dir / "image_annotations.json", "w", encoding="utf-8"
                    ) as fh:
                        json.dump(
                            [ia.model_dump() for ia in extraction.image_annotations],
                            fh,
                            ensure_ascii=False,
                            indent=2,
                        )
                logger.info("Saved outputs to %s", out_dir)

            # NOTE: upload_to_supabase is left as a hook; actual uploader should be external and called here
            if upload_to_supabase:
                logger.info(
                    "upload_to_supabase requested but uploader not implemented in this file"
                )

            return {
                "file_id": file_id,
                "file_name": file_name,
                "chunks": processed_chunks,
                "formulas": [f.model_dump() for f in extraction.formulas],
                "code_blocks": [c.model_dump() for c in extraction.code_blocks],
                "image_annotations": [
                    ia.model_dump() for ia in extraction.image_annotations
                ],
                "metadata": extraction.metadata,
            }
        except Exception as e:
            logger.exception("Processing failed for %s", file_path)
            raise RuntimeError(f"Processing failed for {file_path}") from e


# # ---------------------------
# # CLI Entrypoint
# # ---------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Docling extraction + chunking pipeline (refactored)")
#     parser.add_argument("--file", "-f", required=True, help="Path to PDF or TXT file")
#     parser.add_argument("--out", "-o", default="extracted_content", help="Output folder")
#     parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per chunk")
#     parser.add_argument("--min-tokens", type=int, default=50, help="Min tokens to consider a chunk too short")
#     parser.add_argument("--no-save", action="store_true", help="Do not save local outputs")
#     args = parser.parse_args()

#     cfg = ExtractionConfig(output_dir=args.out, max_chunk_tokens=args.max_tokens, min_tokens_for_chunk=args.min_tokens)
#     orch = EnhancedOrchestrator(config=cfg)
#     result = orch.process_file(args.file, save_local=not args.no_save)
#     logger.info("Processing complete: file_id=%s chunks=%d", result["file_id"], len(result["chunks"]))


# if __name__ == "__main__":
#     main()
