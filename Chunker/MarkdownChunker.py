import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================
# TOKEN COUNTING
# ============================================================
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, otherwise approximate."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))


# ============================================================
# CONTENT TYPES
# ============================================================
class ContentType(Enum):
    CODE = "code"
    MATH = "math"
    TABLE = "table"
    PROSE = "prose"


@dataclass
class ContentBlock:
    content: str
    type: ContentType
    start_idx: int
    end_idx: int


# ============================================================
# BLOCK DETECTION
# ============================================================
def detect_content_blocks(text: str) -> List[ContentBlock]:
    """
    Detect code, math, table, and prose blocks inside markdown text.
    """
    blocks = []
    patterns = []

    # Code blocks: ```...```
    code_pat = r"```[\s\S]*?```"
    patterns.append((code_pat, ContentType.CODE))

    # Inline math $$...$$
    math_pat = r"\$\$[\s\S]*?\$\$"
    patterns.append((math_pat, ContentType.MATH))

    # Display math \[ ... \]
    math_bracket_pat = r"\\\[[\s\S]*?\\\]"
    patterns.append((math_bracket_pat, ContentType.MATH))

    # LaTeX environments
    env_pat = r"\\begin\{(equation|align|gather|multline|eqnarray).*?\}\\end\{\1\}"
    patterns.append((env_pat, ContentType.MATH))

    # Markdown tables
    table_pat = (
        r"(?:^\|.*\|\s*\n"  # header
        r"^\|[-: ]+\|\s*\n"  # separator
        r"(?:^\|.*\|\s*\n)+)"  # body rows
    )
    patterns.append((table_pat, ContentType.TABLE))

    # Collect all matches
    matches = []
    for pat, typ in patterns:
        for m in re.finditer(pat, text, re.MULTILINE):
            matches.append((m.start(), m.end(), m.group(0), typ))

    matches.sort(key=lambda x: x[0])
    last_end = 0

    for start, end, content, typ in matches:
        if start > last_end:
            prose = text[last_end:start].strip()
            if prose:
                blocks.append(ContentBlock(prose, ContentType.PROSE, last_end, start))
        blocks.append(ContentBlock(content, typ, start, end))
        last_end = end

    if last_end < len(text):
        prose = text[last_end:].strip()
        if prose:
            blocks.append(ContentBlock(prose, ContentType.PROSE, last_end, len(text)))

    if not blocks:
        blocks.append(ContentBlock(text, ContentType.PROSE, 0, len(text)))

    return blocks


# ============================================================
# SPECIALIZED BLOCK SPLITTING
# ============================================================
def split_math_block(math: str, max_tokens: int) -> List[str]:
    if count_tokens(math) <= max_tokens:
        return [math]
    # Fallback: split by lines
    lines = math.split("\n")
    chunks, current = [], ""
    for line in lines:
        test = current + line + "\n"
        if count_tokens(test) > max_tokens and current:
            chunks.append(current.rstrip("\n"))
            current = line + "\n"
        else:
            current = test
    if current:
        chunks.append(current.rstrip("\n"))
    return chunks


def split_table_block(table: str, max_tokens: int) -> List[str]:
    if count_tokens(table) <= max_tokens:
        return [table]
    lines = table.strip().split("\n")
    if len(lines) < 3:
        return [table]
    header = lines[0]
    separator = lines[1]
    body = lines[2:]
    chunks = []
    current = [header, separator]
    for row in body:
        test = "\n".join(current + [row])
        if count_tokens(test) > max_tokens and len(current) > 2:
            chunks.append("\n".join(current))
            current = [header, separator, row]
        else:
            current.append(row)
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_code_block(code: str, max_tokens: int) -> List[str]:
    if count_tokens(code) <= max_tokens:
        return [code]
    chunks, current = [], ""
    for line in code.split("\n"):
        test = current + line + "\n"
        if count_tokens(test) > max_tokens and current:
            chunks.append(current.rstrip("\n"))
            current = line + "\n"
        else:
            current = test
    if current:
        chunks.append(current.rstrip("\n"))
    return chunks


# ============================================================
# CONFIG
# ============================================================
@dataclass
class ChunkConfig:
    max_chunk_tokens: int = 300
    min_chunk_tokens: int = 50
    preserve_code_blocks: bool = True
    merge_short_chunks: bool = True
    use_recursive_splitter: bool = True


# ============================================================
# MAIN CHUNKER
# ============================================================
class MarkdownChunker:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        if LANGCHAIN_AVAILABLE:
            self.header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ],
                strip_headers=False,
            )
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_tokens * 4,
                chunk_overlap=50,
                length_function=count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self.header_splitter = None
            self.recursive_splitter = None

    # ------------------------------------------------------------
    # NUMERIC HEADING DETECTION
    # ------------------------------------------------------------
    def _detect_numeric_heading(self, line: str) -> Optional[tuple]:
        match = re.match(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$", line.strip())
        if not match:
            return None
        numeric_part = match.group(1).rstrip(".")
        title = match.group(2).strip()
        levels = list(map(int, numeric_part.split(".")))
        return (levels, title)

    def _split_by_numeric_headers(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split("\n")
        sections = []
        current = {"header": None, "content": [], "section_level": None}
        level_headers = {}

        for line in lines:
            detected = self._detect_numeric_heading(line)
            if detected:
                if current["content"]:
                    sections.append(
                        {
                            "header": current["header"],
                            "section_level": current["section_level"],
                            "parent": current.get("parent"),
                            "top_header": level_headers.get(1),
                            "content": "\n".join(current["content"]),
                        }
                    )
                levels, title = detected
                level = len(levels)
                full_header = f"{'.'.join(map(str, levels))}. {title}"
                level_headers[level] = full_header
                for k in range(level + 1, 10):
                    level_headers.pop(k, None)
                parent = level_headers.get(level - 1)
                current = {
                    "header": full_header,
                    "section_level": level,
                    "parent": parent,
                    "content": [line],
                }
            else:
                current["content"].append(line)

        if current["content"]:
            sections.append(
                {
                    "header": current["header"],
                    "section_level": current["section_level"],
                    "parent": current.get("parent"),
                    "top_header": level_headers.get(1),
                    "content": "\n".join(current["content"]),
                }
            )
        return sections

    # ------------------------------------------------------------
    # MARKDOWN HEADER SPLITTING
    # ------------------------------------------------------------
    def _split_by_markdown_headers(self, text: str) -> List[Dict[str, Any]]:
        if not LANGCHAIN_AVAILABLE or not self.header_splitter:
            return []

        docs = self.header_splitter.split_text(text)
        sections = []

        # Track current active headers at each level
        current_headers = {
            1: None,
            2: None,
            3: None,
            4: None,
        }

        for doc in docs:
            content = doc.page_content.strip()
            metadata = doc.metadata

            # Update current headers based on what's present in this chunk
            header_text = None
            level = None

            if "Header 4" in metadata:
                header_text = metadata["Header 4"]
                level = 4
                current_headers[4] = header_text
                current_headers[3] = metadata.get(
                    "Header 3"
                )  # in case it was carried over
                current_headers[2] = metadata.get("Header 2")
                current_headers[1] = metadata.get("Header 1")
            elif "Header 3" in metadata:
                header_text = metadata["Header 3"]
                level = 3
                current_headers[3] = header_text
                current_headers[4] = None  # reset lower level
                current_headers[2] = metadata.get("Header 2")
                current_headers[1] = metadata.get("Header 1")
            elif "Header 2" in metadata:
                header_text = metadata["Header 2"]
                level = 2
                current_headers[2] = header_text
                current_headers[3] = None
                current_headers[4] = None
                current_headers[1] = metadata.get("Header 1")
            elif "Header 1" in metadata:
                header_text = metadata["Header 1"]
                level = 1
                current_headers[1] = header_text
                current_headers[2] = None
                current_headers[3] = None
                current_headers[4] = None

            # Determine parent: one level up
            parent = None
            if level and level > 1:
                parent = current_headers.get(level - 1)

            sections.append(
                {
                    "header": header_text,
                    "section_level": level,
                    "parent": parent,
                    "top_header": current_headers[1],
                    "content": content,
                }
            )

        return sections

    # ------------------------------------------------------------
    # HEADER STYLE DETECTION (IMPROVED)
    # ------------------------------------------------------------
    def _detect_header_style(self, text: str) -> str:
        md_pattern = r"^#{1,4}\s+.+"  # # to ####
        md_matches = len(re.findall(md_pattern, text, re.MULTILINE))

        # Only count numeric headings that do NOT start with #
        num_pattern = r"^(?!#{1,4}\s)\d+(?:\.\d+)*\.?\s+.+"
        num_matches = len(re.findall(num_pattern, text, re.MULTILINE))

        if md_matches > 0:
            return "markdown"  # Prefer markdown if any real # headers exist
        return "numeric" if num_matches > 0 else "markdown"

    # ------------------------------------------------------------
    # CONTENT-AWARE PROCESSING
    # ------------------------------------------------------------
    def _process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for section in sections:
            content = section["content"]
            blocks = detect_content_blocks(content)

            for block in blocks:
                if block.type in [
                    ContentType.CODE,
                    ContentType.MATH,
                    ContentType.TABLE,
                ]:
                    chunk_texts = self._split_special_block(block)
                else:
                    chunk_texts = self._split_prose_block(block)

                for text in chunk_texts:
                    all_chunks.append(
                        {
                            "content": text.strip(),
                            "metadata": {
                                "section_header": section["header"],
                                "section_level": section["section_level"],
                                "parent_section": section.get("parent"),
                                "top_header": section.get("top_header"),
                                "content_type": block.type.value,
                            },
                            "estimated_tokens": count_tokens(text),
                        }
                    )
        return all_chunks

    def _split_special_block(self, block: ContentBlock) -> List[str]:
        max_t = self.config.max_chunk_tokens
        if block.type == ContentType.MATH:
            return split_math_block(block.content, max_t)
        elif block.type == ContentType.TABLE:
            return split_table_block(block.content, max_t)
        elif block.type == ContentType.CODE:
            return split_code_block(block.content, max_t)
        return [block.content]

    def _split_prose_block(self, block: ContentBlock) -> List[str]:
        """Split prose using recursive splitter if available and needed."""
        text = block.content.strip()
        if not text:
            return []

        if (
            self.config.use_recursive_splitter
            and LANGCHAIN_AVAILABLE
            and self.recursive_splitter
            and count_tokens(text) > self.config.max_chunk_tokens
        ):
            chunks = self.recursive_splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]

        return [text]

    # ------------------------------------------------------------
    #   REMOVE CORRUPTED LATEX (REFINED LINE-BY-LINE)
    # -----------------------------------------------------------
    def _remove_corrupted_latex(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refined line-by-line cleaner to remove OCR artifacts, 1-character lines,
        and 'textual salad' while preserving valid prose, headers, and lists.
        """
        # Skip cleaning for code or math blocks to avoid breaking syntax
        if chunk["metadata"].get("content_type") in {"code", "math"}:
            return chunk

        text = chunk["content"]
        lines = text.splitlines()
        cleaned_lines = []

        # Patterns for known corruption tags (like 1\_b or <latexi)
        corruption_markers = [
            r"1\\_b",
            r"<latexi",
            r"&lt;",
            r"&gt;",
            r'64="',
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            # 1. Remove lines containing known noise markers
            if any(
                re.search(pat, stripped, re.IGNORECASE) for pat in corruption_markers
            ):
                continue

            # 2. Aggressive filter for short fragments (Salad/OCR noise)
            if len(stripped) < 15:
                # Discard lines with only 1 or 2 characters (e.g., "I", "7", "uD")
                if len(stripped) <= 2:
                    continue

                # Discard short lines that are NOT headers, lists, or full sentences.
                # Valid short lines usually start with markdown (#, -) or end with (. , :)
                is_markdown_element = stripped.startswith(
                    ("#", "-", "*", ">", "1.", "2.")
                )
                has_sentence_punctuation = stripped.endswith(
                    (".", ":", "!", "?", ")", "]")
                )

                if not is_markdown_element and not has_sentence_punctuation:
                    continue

                # High symbol/non-alphanumeric check (e.g. "+z", "v+")
                non_alnum_ratio = sum(not c.isalnum() for c in stripped) / len(stripped)
                if non_alnum_ratio > 0.4:
                    continue

            cleaned_lines.append(line)

        # Reconstruct text and normalize whitespace
        cleaned_text = "\n".join(cleaned_lines).strip()
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        # Fallback if the entire chunk was garbage
        if not cleaned_text or len(cleaned_text) < 5:
            cleaned_text = "[REMOVED: Corrupted LaTeX content]"

        chunk["content"] = cleaned_text
        chunk["estimated_tokens"] = count_tokens(cleaned_text)

        return chunk

    # ------------------------------------------------------------
    # CONSERVATIVE ROLL-UP / MERGE LOGIC (WITH FRONT-MATTER SUPPORT)
    # ------------------------------------------------------------
    def _is_front_matter(self, chunk: Dict[str, Any]) -> bool:
        """
        Identify academic front-matter blocks that are allowed to roll up
        even if section headers differ.
        """
        header = (chunk["metadata"].get("section_header") or "").strip().upper()
        return header in {
            "",
            "ARTICLE HISTORY",
            "ABSTRACT",
            "KEYWORDS",
            "AUTHOR INFORMATION",
        }

    def _merge_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Conservatively roll up adjacent small chunks.

        Guarantees:
        - Only adjacent chunks are merged
        - No reordering
        - No cross-content-type merging
        - Section boundaries are respected,
            except for explicit front-matter roll-up
        - Hard token limits are enforced
        """

        if not chunks or not self.config.merge_short_chunks:
            return chunks

        merged: List[Dict[str, Any]] = []
        buffer: Optional[Dict[str, Any]] = None

        def compatible(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            # Strict rule: same section + same content type
            if (
                a["metadata"]["section_header"] == b["metadata"]["section_header"]
                and a["metadata"]["content_type"] == b["metadata"]["content_type"]
                and (a["estimated_tokens"] + b["estimated_tokens"])
                <= self.config.max_chunk_tokens
            ):
                return True

            # Front-matter exception (prose only)
            if (
                self._is_front_matter(a)
                and self._is_front_matter(b)
                and a["metadata"]["content_type"] == "prose"
                and b["metadata"]["content_type"] == "prose"
                and (a["estimated_tokens"] + b["estimated_tokens"])
                <= self.config.max_chunk_tokens
            ):
                return True

            return False

        for chunk in chunks:
            if buffer is None:
                buffer = chunk
                continue

            # Roll-up if safe
            if buffer["estimated_tokens"] < self.config.min_chunk_tokens and compatible(
                buffer, chunk
            ):
                buffer["content"] = buffer["content"] + "\n\n" + chunk["content"]
                buffer["estimated_tokens"] = count_tokens(buffer["content"])
            else:
                merged.append(buffer)
                buffer = chunk

        # Flush final buffer
        if buffer is not None:
            merged.append(buffer)

        return merged

    # ------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------
    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        header_style = self._detect_header_style(markdown_text)

        if header_style == "markdown":
            sections = self._split_by_markdown_headers(markdown_text)
            if not sections:  # Fallback
                sections = self._split_by_numeric_headers(markdown_text)
        else:
            sections = self._split_by_numeric_headers(markdown_text)

        chunks = self._process_sections(sections)

        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)

        # 4. PLACE THE CALL HERE (Before indexing)
        # This cleans every chunk line-by-line
        chunks = [self._remove_corrupted_latex(c) for c in chunks]

        for idx, chunk in enumerate(chunks):
            chunk["chunk_index"] = idx

        return chunks


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================
def chunk_markdown(
    markdown_text: str,
    max_tokens: int = 500,
    min_tokens: int = 50,
    use_recursive_splitter: bool = True,
    merge_short: bool = True,
) -> List[Dict[str, Any]]:
    config = ChunkConfig(
        max_chunk_tokens=max_tokens,
        min_chunk_tokens=min_tokens,
        use_recursive_splitter=use_recursive_splitter,
        merge_short_chunks=merge_short,
    )
    chunker = MarkdownChunker(config)
    return chunker.process(markdown_text)
