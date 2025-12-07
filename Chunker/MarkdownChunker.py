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

    # Code blocks
    code_pat = r"```[\s\S]*?```"
    patterns.append((code_pat, ContentType.CODE))

    # Math $$...$$
    math_pat = r"\$\$[\s\S]*?\$\$"
    patterns.append((math_pat, ContentType.MATH))

    # Math \[ ... \]
    math_bracket_pat = r"\\\[[\s\S]*?\\\]"
    patterns.append((math_bracket_pat, ContentType.MATH))

    # Math environments
    env_pat = r"\\begin\{(equation|align|gather|multline)\}[\s\S]*?\\end\{\1\}"
    patterns.append((env_pat, ContentType.MATH))

    # Tables
    table_pat = (
        r"(?:^\|.*\|\s*\n"  # header
        r"^\|[-: ]+\|\s*\n"  # separator
        r"(?:^\|.*\|\s*\n)+)"  # body rows
    )
    patterns.append((table_pat, ContentType.TABLE))

    # Collect matches
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
    """Keep entire math block intact whenever possible."""
    if count_tokens(math) <= max_tokens:
        return [math]

    lines = math.split("\n")
    chunks, current = [], ""

    for line in lines:
        test = current + line + "\n"
        if count_tokens(test) > max_tokens and current:
            chunks.append(current)
            current = line + "\n"
        else:
            current = test

    if current:
        chunks.append(current)

    return chunks


def split_table_block(table: str, max_tokens: int) -> List[str]:
    """Keep table intact or split by row groups if extremely large."""
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
    """Split code block by lines, preserving syntax structure."""
    if count_tokens(code) <= max_tokens:
        return [code]

    chunks, current = [], ""
    for line in code.split("\n"):
        test = current + line + "\n"
        if count_tokens(test) > max_tokens and current:
            chunks.append(current)
            current = line + "\n"
        else:
            current = test

    if current:
        chunks.append(current)
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
    use_recursive_splitter: bool = True  # NEW: Enable recursive splitting


# ============================================================
# MAIN CHUNKER WITH BOTH APPROACHES
# ============================================================


class HybridMarkdownChunker:
    """
    Supports both:
    1. Numeric headers (2.1.3 Title) - original approach
    2. Standard markdown headers (# ## ###) - via MarkdownHeaderTextSplitter
    3. Recursive splitting for prose - intelligently splits long text
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # Initialize LangChain splitters if available
        if LANGCHAIN_AVAILABLE:
            self.header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ],
                strip_headers=False,
            )

            # Recursive splitter for prose content
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_tokens * 4,  # Approximate chars
                chunk_overlap=50,
                length_function=count_tokens,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self.header_splitter = None
            self.recursive_splitter = None

    # ------------------------------------------------------------
    # NUMERIC HEADING DETECTION (Original approach)
    # ------------------------------------------------------------
    def _detect_numeric_heading(self, line: str) -> Optional[tuple]:
        """
        Extract numeric prefix from heading.
        Example: "2.1. Definition" -> ([2, 1], "Definition")
        """
        match = re.match(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$", line)
        if not match:
            return None
        numeric_part = match.group(1).rstrip(".")
        title = match.group(2).strip()
        levels = list(map(int, numeric_part.split(".")))
        return (levels, title)

    def _split_by_numeric_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split by numeric headers like 2.1, 3.4.2, etc."""
        lines = text.split("\n")
        sections = []
        current = {"header": None, "content": [], "section_level": None}
        level_headers = {}

        for line in lines:
            detected = self._detect_numeric_heading(line)

            if detected:
                # Save previous section
                if current["content"]:
                    sections.append(
                        {
                            "header": current["header"],
                            "section_level": current["section_level"],
                            "parent": current.get("parent"),
                            "top_header": level_headers.get(1),  # Top-level header
                            "content": "\n".join(current["content"]),
                        }
                    )

                levels, title = detected
                level = len(levels)
                full_header = f"{'.'.join(map(str, levels))}. {title}"

                # Update hierarchy
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

        # Final section
        if current["content"]:
            sections.append(
                {
                    "header": current["header"],
                    "section_level": current["section_level"],
                    "parent": current.get("parent"),
                    "top_header": level_headers.get(1),  # Top-level header
                    "content": "\n".join(current["content"]),
                }
            )

        return sections

    # ------------------------------------------------------------
    # MARKDOWN HEADER SPLITTING (LangChain approach)
    # ------------------------------------------------------------
    def _split_by_markdown_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split by standard markdown headers using LangChain."""
        if not LANGCHAIN_AVAILABLE or not self.header_splitter:
            return []

        docs = self.header_splitter.split_text(text)
        sections = []
        prev_headers = {"Header 1": None, "Header 2": None, "Header 3": None}

        for doc in docs:
            content = getattr(doc, "page_content", "") or ""
            md = getattr(doc, "metadata", {}) or {}

            # Update header hierarchy
            if "Header 1" in md:
                prev_headers["Header 1"] = md["Header 1"]
                prev_headers["Header 2"] = None
                prev_headers["Header 3"] = None
            if "Header 2" in md:
                prev_headers["Header 2"] = md["Header 2"]
                prev_headers["Header 3"] = None
            if "Header 3" in md:
                prev_headers["Header 3"] = md["Header 3"]

            # Determine section info
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

            sections.append(
                {
                    "header": section_header,
                    "section_level": section_level,
                    "parent": parent_section,
                    "top_header": prev_headers["Header 1"],  # Top-level header
                    "content": content,
                }
            )

        return sections

    # ------------------------------------------------------------
    # AUTO-DETECT HEADER STYLE
    # ------------------------------------------------------------
    def _detect_header_style(self, text: str) -> str:
        """Detect whether document uses numeric or markdown headers."""
        # Check for markdown headers
        md_pattern = r"^#{1,3}\s+.+"
        md_matches = len(re.findall(md_pattern, text, re.MULTILINE))

        # Check for numeric headers
        num_pattern = r"^\d+(?:\.\d+)*\.?\s+.+"
        num_matches = len(re.findall(num_pattern, text, re.MULTILINE))

        if md_matches > num_matches:
            return "markdown"
        elif num_matches > 0:
            return "numeric"
        else:
            return "markdown"  # Default to markdown

    # ------------------------------------------------------------
    # PROCESS SECTIONS WITH CONTENT-AWARE SPLITTING
    # ------------------------------------------------------------
    def _process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process sections with content-aware splitting."""
        all_chunks = []

        for section in sections:
            content = section["content"]
            blocks = detect_content_blocks(content)

            for block in blocks:
                # Special handling for each block type
                if block.type in [
                    ContentType.CODE,
                    ContentType.MATH,
                    ContentType.TABLE,
                ]:
                    # Keep these intact using specialized splitters
                    chunk_texts = self._split_special_block(block)
                else:
                    # Use recursive splitter for prose if available
                    chunk_texts = self._split_prose_block(block)

                for text in chunk_texts:
                    all_chunks.append(
                        {
                            "content": text,
                            "metadata": {
                                "section_header": section["header"],
                                "section_level": section["section_level"],
                                "parent_section": section.get("parent"),
                                "top_header": section.get("top_header"),  # ← RESTORED
                                "content_type": block.type.value,
                            },
                            "estimated_tokens": count_tokens(text),
                        }
                    )

        return all_chunks

    def _split_special_block(self, block: ContentBlock) -> List[str]:
        """Split code, math, or table blocks."""
        max_t = self.config.max_chunk_tokens

        if block.type == ContentType.MATH:
            return split_math_block(block.content, max_t)
        elif block.type == ContentType.TABLE:
            return split_table_block(block.content, max_t)
        elif block.type == ContentType.CODE:
            return split_code_block(block.content, max_t)

        return [block.content]

    def _split_prose_block(self, block: ContentBlock) -> List[str]:
        """Split prose using recursive splitter if available, else fallback."""
        if (
            self.config.use_recursive_splitter
            and LANGCHAIN_AVAILABLE
            and self.recursive_splitter
        ):
            # Use recursive splitter for better results
            if count_tokens(block.content) > self.config.max_chunk_tokens:
                docs = self.recursive_splitter.split_text(block.content)
                return [doc for doc in docs if doc.strip()]
            return [block.content]
        else:
            # Fallback to paragraph splitting
            return self._split_by_paragraphs(block.content)

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Fallback paragraph-based splitting."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        max_t = self.config.max_chunk_tokens

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            if count_tokens(p) <= max_t:
                chunks.append(p)
            else:
                # Split by sentences
                pattern = r"(?<=[.!?])\s+(?=[A-Z])"
                sentences = re.split(pattern, p)
                current = ""

                for s in sentences:
                    test = (current + " " + s).strip()
                    if count_tokens(test) > max_t and current:
                        chunks.append(current)
                        current = s
                    else:
                        current = test

                if current:
                    chunks.append(current)

        return chunks if chunks else [text]

    # ------------------------------------------------------------
    # MERGE SHORT CHUNKS
    # ------------------------------------------------------------
    def _merge_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge chunks that are too short."""
        if len(chunks) <= 1 or not self.config.merge_short_chunks:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            curr = chunks[i]

            if curr["estimated_tokens"] < self.config.min_chunk_tokens and i + 1 < len(
                chunks
            ):
                nxt = chunks[i + 1]

                same_section = (
                    curr["metadata"]["section_header"]
                    == nxt["metadata"]["section_header"]
                )
                merged_tokens = curr["estimated_tokens"] + nxt["estimated_tokens"]
                fits = merged_tokens <= int(self.config.max_chunk_tokens * 1.2)

                if same_section and fits:
                    combined = curr["content"] + "\n\n" + nxt["content"]
                    nxt["content"] = combined
                    nxt["estimated_tokens"] = count_tokens(combined)
                    i += 1
                    continue

            merged.append(curr)
            i += 1

        return merged

    # ------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------
    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        """
        Main processing function that:
        1. Auto-detects header style
        2. Splits by headers
        3. Processes content blocks with appropriate splitters
        4. Merges short chunks
        5. Assigns indices
        """
        # Auto-detect and split by headers
        header_style = self._detect_header_style(markdown_text)

        if header_style == "numeric":
            sections = self._split_by_numeric_headers(markdown_text)
        else:
            sections = self._split_by_markdown_headers(markdown_text)
            if not sections:  # Fallback if LangChain not available
                sections = self._split_by_numeric_headers(markdown_text)

        # Process sections with content-aware splitting
        chunks = self._process_sections(sections)

        # Merge short chunks if enabled
        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)

        # Assign chunk indices
        for idx, chunk in enumerate(chunks):
            chunk["chunk_index"] = idx

        return chunks


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def chunk_markdown(
    markdown_text: str,
    max_tokens: int = 300,
    min_tokens: int = 50,
    use_recursive_splitter: bool = True,
):
    """
    Convenience function to chunk markdown text.

    Args:
        markdown_text: The markdown text to chunk
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk (for merging)
        use_recursive_splitter: Use recursive splitter for prose (recommended)

    Returns:
        List of chunk dictionaries with content, metadata, and token counts
    """
    config = ChunkConfig(
        max_chunk_tokens=max_tokens,
        min_chunk_tokens=min_tokens,
        use_recursive_splitter=use_recursive_splitter,
    )
    return HybridMarkdownChunker(config).process(markdown_text)


# ============================================================
# EXAMPLE USAGE
# ============================================================


if __name__ == "__main__":
    # Example with both header styles
    sample_md = """
# Introduction

This is the introduction section with some text.

## Background

Some background information here.

### Related Work

Details about related work.

2. AUDIO DEEPFAKE VERIFICATION

This section uses numeric headers.

2.1. Definition

The definition subsection.

2.1.1. Technical Details

```python
def example():
    return "code block"
```

More technical details here.

## Conclusion

Final thoughts and conclusions.
"""

    chunks = chunk_markdown(sample_md, max_tokens=200)

    print(f"Generated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']}:")
        print(f"  Header: {chunk['metadata']['section_header']}")
        print(f"  Level: {chunk['metadata']['section_level']}")
        print(f"  Type: {chunk['metadata']['content_type']}")
        print(f"  Tokens: {chunk['estimated_tokens']}")
        print(f"  Content preview: {chunk['content'][:100]}...")
        print()
