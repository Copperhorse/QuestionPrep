import re
import textwrap
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ============================================================

# 1. DEPENDENCY CHECKS

# ============================================================

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True

except ImportError:
    TIKTOKEN_AVAILABLE = False


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True

except ImportError:
    LANGCHAIN_AVAILABLE = False


def count_tokens(text: str) -> int:
    if not text:
        return 0

    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")

            return len(enc.encode(text))

        except Exception:
            pass

    return max(1, int(len(text) / 4))


# ============================================================

# 2. ENUMS & CONFIG

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


@dataclass
class ChunkConfig:
    max_chunk_tokens: int = 500

    min_chunk_tokens: int = 50

    merge_short_chunks: bool = True

    use_recursive_splitter: bool = True


# ============================================================

# 3. TEXT SPLITTERS (Native & LangChain Wrapper)

# ============================================================


class NativeTextSplitter:
    """Recursively splits text by separators without external libs."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []

        separator = separators[-1]

        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep

                break

            if sep in text:
                separator = sep

                new_separators = separators[i + 1 :]

                break

        splits = text.split(separator) if separator else list(text)

        current_chunk = []

        current_len = 0

        for s in splits:
            s_len = count_tokens(s)

            if current_len + s_len > self.chunk_size:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))

                if s_len > self.chunk_size and new_separators:
                    final_chunks.extend(self._split(s, new_separators))

                    current_chunk = []

                    current_len = 0

                else:
                    current_chunk = [s]

                    current_len = s_len

            else:
                current_chunk.append(s)

                current_len += s_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return [c.strip() for c in final_chunks if c.strip()]


# ============================================================

# 4. MAIN CHUNKER CLASS

# ============================================================


class MarkdownChunker:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # Use LangChain if available, else Native

        if LANGCHAIN_AVAILABLE and self.config.use_recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_tokens * 4,
                chunk_overlap=50,
                length_function=count_tokens,
            )

        else:
            self.splitter = NativeTextSplitter(chunk_size=self.config.max_chunk_tokens)

    def _normalize_headers(self, text: str) -> str:
        """

        Smart Normalization:

        - Detects Roman Numerals (I., II.) -> Converts to Level 1 Header (#)

        - Detects Letters (A., B.) -> Converts to Level 2 Header (##)

        - Detects Numbers (1., 2.) -> Converts to Level 3 Header (###)

        This restores the hierarchy lost during PDF extraction.

        """

        lines = text.split("\n")

        new_lines = []

        # Patterns

        roman_pat = re.compile(
            r"^\s*(?!\s*[-*])(?:#{1,6}\s+)?([IVX]+\.)\s+(.*)$"
        )  # I. Intro

        alpha_pat = re.compile(
            r"^\s*(?!\s*[-*])(?:#{1,6}\s+)?([A-Z]\.)\s+(.*)$"
        )  # A. Method

        num_pat = re.compile(r"^\s*(?!\s*[-*])(?:#{1,6}\s+)?(\d+\.)\s+(.*)$")  # 1. Data

        md_pat = re.compile(r"^\s*(#{1,6})\s+(.*)")

        for line in lines:
            line = line.strip()

            if not line:
                new_lines.append("")

                continue

            # 1. Check Roman (Level 1)

            m_roman = roman_pat.match(line)

            if m_roman:
                new_lines.append(f"# {m_roman.group(1)} {m_roman.group(2)}")

                continue

            # 2. Check Alpha (Level 2)

            m_alpha = alpha_pat.match(line)

            if m_alpha:
                new_lines.append(f"## {m_alpha.group(1)} {m_alpha.group(2)}")

                continue

            # 3. Check Numeric (Level 3)

            # Only if line is short (heuristic to avoid list items becoming headers)

            m_num = num_pat.match(line)

            if m_num and len(line) < 100:
                new_lines.append(f"### {m_num.group(1)} {m_num.group(2)}")

                continue

            # 4. Standard Markdown (Normalize indentation)

            m_md = md_pat.match(line)

            if m_md:
                new_lines.append(f"{m_md.group(1)} {m_md.group(2)}")

                continue

            new_lines.append(line)

        return "\n".join(new_lines)

    def _split_hierarchical_sections(self, text: str) -> List[Dict[str, Any]]:
        """

        Stateful parser that tracks Header Hierarchy (Stack).

        Output: List of sections, each with 'parent' and 'top_header' metadata.

        """

        lines = text.split("\n")

        sections = []

        current_content = []

        # The Stack: {1: "I. Intro", 2: "A. Naive RAG"}

        header_stack = {}

        current_level = 0

        for line in lines:
            match = re.match(r"^(#{1,6})\s+(.*)", line)

            if match:
                # 1. Save previous section

                if current_content:
                    # Resolve Parents

                    header_text = header_stack.get(current_level, "Root")

                    # Parent is the nearest defined level above current

                    parent_text = next(
                        (
                            header_stack[k]
                            for k in range(current_level - 1, 0, -1)
                            if k in header_stack
                        ),
                        None,
                    )

                    top_text = header_stack.get(
                        1, header_stack.get(current_level, "Root")
                    )

                    sections.append(
                        {
                            "header": header_text,
                            "section_level": current_level,
                            "parent": parent_text,
                            "top_header": top_text,
                            "content": "\n".join(current_content).strip(),
                        }
                    )

                # 2. Update Stack

                hashes, title = match.groups()

                new_level = len(hashes)

                current_level = new_level

                header_stack[new_level] = title.strip()

                # Wipe deeper levels (we moved up or stayed same)

                keys_to_wipe = [k for k in header_stack if k > new_level]

                for k in keys_to_wipe:
                    del header_stack[k]

                current_content = []

            else:
                current_content.append(line)

        # Flush last section

        if current_content:
            header_text = header_stack.get(current_level, "Root")

            parent_text = next(
                (
                    header_stack[k]
                    for k in range(current_level - 1, 0, -1)
                    if k in header_stack
                ),
                None,
            )

            top_text = header_stack.get(1, header_text)

            sections.append(
                {
                    "header": header_text,
                    "section_level": current_level,
                    "parent": parent_text,
                    "top_header": top_text,
                    "content": "\n".join(current_content).strip(),
                }
            )

        return sections

    def _process_blocks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_chunks = []

        patterns = [
            (r"```[\s\S]*?```", ContentType.CODE),
            (r"\$\$[\s\S]*?\$\$", ContentType.MATH),
            (r"(?:^\|\s*.*\|\s*\n^[-: |]+\s*\n(?:^\|.*\|\s*\n)+)", ContentType.TABLE),
        ]

        for section in sections:
            text = section["content"]

            if not text:
                continue

            # 1. Identify Blocks

            special_spans = []

            for pat, typ in patterns:
                for m in re.finditer(pat, text, re.MULTILINE):
                    special_spans.append((m.start(), m.end(), m.group(0), typ))

            special_spans.sort(key=lambda x: x[0])

            # 2. Slice & Dice

            last_end = 0

            blocks = []

            for start, end, content, typ in special_spans:
                if start > last_end:
                    blocks.append(
                        ContentBlock(text[last_end:start].strip(), ContentType.PROSE)
                    )

                blocks.append(ContentBlock(content, typ))

                last_end = end

            if last_end < len(text):
                blocks.append(ContentBlock(text[last_end:].strip(), ContentType.PROSE))

            # 3. Create Chunks

            for block in blocks:
                if not block.content:
                    continue

                # Split Logic

                if block.type == ContentType.PROSE:
                    if hasattr(
                        self.splitter, "split_text"
                    ):  # Duck typing for LangChain vs Native
                        sub_texts = self.splitter.split_text(block.content)

                    else:
                        sub_texts = [block.content]

                elif (
                    block.type == ContentType.TABLE
                    and count_tokens(block.content) > self.config.max_chunk_tokens
                ):
                    # Fallback for huge tables

                    sub_texts = (
                        self.splitter.split_text(block.content)
                        if hasattr(self.splitter, "split_text")
                        else [block.content]
                    )

                else:
                    sub_texts = [block.content]

                for t in sub_texts:
                    final_chunks.append(
                        {
                            "content": t,
                            "metadata": {
                                "section_header": section["header"],
                                "parent_section": section[
                                    "parent"
                                ],  # Metadata Restored!
                                "top_header": section[
                                    "top_header"
                                ],  # Metadata Restored!
                                "content_type": block.type.value,
                            },
                            "estimated_tokens": count_tokens(t),
                        }
                    )

        return final_chunks

    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        merged = []

        buffer = None

        for chunk in chunks:
            if not buffer:
                buffer = chunk

                continue

            # Boundary Checks

            diff_header = (
                buffer["metadata"]["section_header"]
                != chunk["metadata"]["section_header"]
            )

            diff_type = (
                buffer["metadata"]["content_type"] != chunk["metadata"]["content_type"]
            )

            too_big = (
                buffer["estimated_tokens"] + chunk["estimated_tokens"]
            ) > self.config.max_chunk_tokens

            if diff_header or diff_type or too_big:
                merged.append(buffer)

                buffer = chunk

            else:
                buffer["content"] += "\n\n" + chunk["content"]

                buffer["estimated_tokens"] += chunk["estimated_tokens"]

        if buffer:
            merged.append(buffer)

        return merged

    def _add_ids(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i, c in enumerate(chunks):
            c["chunk_id"] = str(uuid.uuid4())

            c["chunk_index"] = i

            c["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None

            c["next_chunk_id"] = None

        for i in range(len(chunks) - 1):
            chunks[i]["next_chunk_id"] = chunks[i + 1]["chunk_id"]

        return chunks

    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        # 1. Normalize (Smart Hierarchy Inference)

        norm_text = self._normalize_headers(markdown_text)

        # 2. Split Sections (Stateful Stack Parser)

        sections = self._split_hierarchical_sections(norm_text)

        # 3. Block detection & Splitting

        chunks = self._process_blocks(sections)

        # 4. Merge

        if self.config.merge_short_chunks:
            chunks = self._merge_chunks(chunks)

        # 5. IDs

        return self._add_ids(chunks)


def chunk_markdown(text: str, **kwargs) -> List[Dict[str, Any]]:
    config = ChunkConfig(**kwargs)

    return MarkdownChunker(config).process(text)


if __name__ == "__main__":
    # Test Data: Imitating your PDF extraction

    sample = textwrap.dedent("""

        ## I. INTRODUCTION

        Large models are great.


        ## A. Naive RAG

        This is a subsection of Introduction.


        ## B. Advanced RAG

        This is another subsection.


        ## II. OVERVIEW

        This is a new top level chapter.

    """)

    print("=" * 60)

    chunks = chunk_markdown(sample)

    for i, c in enumerate(chunks):
        m = c["metadata"]

        print(f"[{i}] {m['content_type']}")

        print(f"    Header: {m['section_header']}")

        print(f"    Parent: {m['parent_section']}")  # CHECK THIS!

        print(f"    Content: {c['content'][:40]}...")

        print("-" * 60)
