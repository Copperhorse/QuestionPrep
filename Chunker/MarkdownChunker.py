import re
import uuid
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
    min_chunk_tokens: int = 50  # Threshold for aggressive merging
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
    # HELPER: HEADER NORMALIZATION
    # ------------------------------------------------------------
    def _normalize_headers(self, text: str) -> str:
        """
        Pre-processing: Converts "7. Linear Layers" to "## 7. Linear Layers"
        so that the splitter detects them as sections properly.
        """
        lines = text.split("\n")
        new_lines = []
        # Pattern: Start of line, Number + Dot + Space, Uppercase Letter, Length < 100
        # Excludes lines starting with - or * to avoid list items
        header_candidate = re.compile(r"^(?!\s*[-*])(\d+\.)\s+([A-Z].{0,80})$")

        for line in lines:
            if header_candidate.match(line):
                new_lines.append(f"## {line}")
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    # ------------------------------------------------------------
    # NUMERIC HEADING DETECTION (Fallback)
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
        current_headers = {1: None, 2: None, 3: None, 4: None}

        for doc in docs:
            content = doc.page_content.strip()
            metadata = doc.metadata
            header_text = None
            level = None

            for i in range(4, 0, -1):
                key = f"Header {i}"
                if key in metadata:
                    header_text = metadata[key]
                    level = i
                    current_headers[i] = header_text
                    for k in range(i + 1, 5):
                        current_headers[k] = None
                    # Fill parents
                    for k in range(i - 1, 0, -1):
                        if metadata.get(f"Header {k}"):
                            current_headers[k] = metadata[f"Header {k}"]
                    break

            parent = current_headers.get(level - 1) if level and level > 1 else None

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
    # HEADER STYLE DETECTION
    # ------------------------------------------------------------
    def _detect_header_style(self, text: str) -> str:
        md_matches = len(re.findall(r"^#{1,4}\s+.+", text, re.MULTILINE))
        return "markdown" if md_matches > 0 else "numeric"

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
                    meta = {
                        "section_header": section["header"],
                        "section_level": section["section_level"],
                        "parent_section": section.get("parent"),
                        "top_header": section.get("top_header"),
                        "content_type": block.type.value,
                    }

                    all_chunks.append(
                        {
                            "content": text.strip(),
                            "metadata": meta,
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
    # REMOVE CORRUPTED LATEX (CLEANING)
    # ------------------------------------------------------------
    def _remove_corrupted_latex(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        if chunk["metadata"].get("content_type") in {"code", "math"}:
            return chunk

        text = chunk["content"]
        lines = text.splitlines()
        cleaned_lines = []
        corruption_markers = [r"1\\_b", r"<latexi", r"&lt;", r"&gt;", r'64="']

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if any(
                re.search(pat, stripped, re.IGNORECASE) for pat in corruption_markers
            ):
                continue
            if len(stripped) < 15:
                # Keep small valid markdown or punctuation
                if len(stripped) <= 2:
                    continue
                is_md = stripped.startswith(("#", "-", "*", ">", "1.", "2."))
                has_punct = stripped.endswith((".", ":", "!", "?", ")", "]"))
                if not is_md and not has_punct:
                    non_alnum = sum(not c.isalnum() for c in stripped) / len(stripped)
                    if non_alnum > 0.4:
                        continue
            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        if not cleaned_text or len(cleaned_text) < 5:
            cleaned_text = "[REMOVED: Corrupted LaTeX content]"

        chunk["content"] = cleaned_text
        chunk["estimated_tokens"] = count_tokens(cleaned_text)
        return chunk

    # ------------------------------------------------------------
    # REPAIR STEP: CONSOLIDATE ORPHANED HEADERS
    # ------------------------------------------------------------
    def _consolidate_orphaned_headers(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return chunks
        consolidated = []
        buffer_text = ""
        for i, chunk in enumerate(chunks):
            text = chunk["content"]
            tokens = chunk["estimated_tokens"]
            c_type = chunk["metadata"]["content_type"]
            # Orphaned header logic: small prose chunk starting with #
            is_orphaned_header = (
                c_type == "prose"
                and text.strip().startswith("#")
                and tokens < 15
                and i < len(chunks) - 1
            )
            if is_orphaned_header:
                buffer_text += text + "\n\n"
            else:
                if buffer_text:
                    chunk["content"] = buffer_text + chunk["content"]
                    chunk["estimated_tokens"] = count_tokens(chunk["content"])
                    buffer_text = ""
                consolidated.append(chunk)
        if buffer_text:
            consolidated.append(
                {
                    "content": buffer_text.strip(),
                    "metadata": chunks[-1]["metadata"].copy(),
                    "estimated_tokens": count_tokens(buffer_text),
                }
            )
        return consolidated

    # ------------------------------------------------------------
    # UNIFIED MERGE LOGIC (Updated to keep code+prose together)
    # ------------------------------------------------------------
    def _merge_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks or not self.config.merge_short_chunks:
            return chunks

        merged = []
        buffer = None
        # Threshold to allow merging different types (e.g. Code + Prose) if both are small
        SMALL_CHUNK_THRESHOLD = 150

        def is_compatible(a, b):
            # 1. Size Check
            if (
                a["estimated_tokens"] + b["estimated_tokens"]
            ) > self.config.max_chunk_tokens:
                return False

            # 2. Section Check
            if a["metadata"]["section_header"] != b["metadata"]["section_header"]:
                return False

            # 3. Type Check (Relaxed)
            # If combined size is small, allow merging Prose + Code (keeps explanation with code)
            if (a["estimated_tokens"] + b["estimated_tokens"]) < SMALL_CHUNK_THRESHOLD:
                return True

            # Otherwise, strict type matching
            return a["metadata"]["content_type"] == b["metadata"]["content_type"]

        for chunk in chunks:
            if buffer is None:
                buffer = chunk
                continue
            if is_compatible(buffer, chunk):
                buffer["content"] = buffer["content"] + "\n\n" + chunk["content"]
                buffer["estimated_tokens"] = count_tokens(buffer["content"])
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer is not None:
            merged.append(buffer)

        return merged

    # ------------------------------------------------------------
    # LINKAGE HELPER
    # ------------------------------------------------------------
    def _add_linkage_and_ids(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for idx, chunk in enumerate(chunks):
            chunk["chunk_id"] = str(uuid.uuid4())
            chunk["chunk_index"] = idx
        for i in range(len(chunks)):
            chunks[i]["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            chunks[i]["next_chunk_id"] = (
                chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None
            )
        return chunks

    # ------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------
    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        # 1. Normalize Headers (Fix "7. Title" -> "## 7. Title")
        clean_text = self._normalize_headers(markdown_text)

        # 2. Split
        if self._detect_header_style(clean_text) == "markdown":
            sections = self._split_by_markdown_headers(clean_text)
            if not sections:
                sections = self._split_by_numeric_headers(clean_text)
        else:
            sections = self._split_by_numeric_headers(clean_text)

        # 3. Process
        chunks = self._process_sections(sections)
        chunks = self._consolidate_orphaned_headers(chunks)

        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)

        chunks = [self._remove_corrupted_latex(c) for c in chunks]
        chunks = self._add_linkage_and_ids(chunks)

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
