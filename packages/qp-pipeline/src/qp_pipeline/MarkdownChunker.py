import re
import textwrap
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
# 3. TEXT SPLITTERS
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
# 4. ROBUST CHUNKER CLASS
# ============================================================


class MarkdownChunker:
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

        # Patterns for identifying special blocks
        self.block_patterns = [
            (re.compile(r"```[\s\S]*?```"), ContentType.CODE),
            (re.compile(r"\$\$[\s\S]*?\$\$"), ContentType.MATH),
            # Table pattern: Pipes on one line, followed by separator line
            (
                re.compile(
                    r"(?:^\|\s*.*\|\s*\n^[-: |]+\s*\n(?:^\|.*\|\s*\n)+)", re.MULTILINE
                ),
                ContentType.TABLE,
            ),
        ]

        if LANGCHAIN_AVAILABLE and self.config.use_recursive_splitter:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.max_chunk_tokens * 4,
                chunk_overlap=50,
                length_function=count_tokens,
            )
        else:
            self.splitter = NativeTextSplitter(chunk_size=self.config.max_chunk_tokens)

    # -----------------------------------------------------------------
    # A. MASKING UTILITIES (Protection)
    # -----------------------------------------------------------------

    def _mask_content(
        self, text: str
    ) -> Tuple[str, Dict[str, Tuple[str, ContentType]]]:
        """
        Replaces Code, Math, and Tables with safe placeholders (masks).
        Returns: (masked_text, dictionary_of_replacements)
        """
        mask_store = {}
        masked_text = text

        for pat, c_type in self.block_patterns:

            def replacer(match):
                content = match.group(0)
                mask_id = f"__MASK_{uuid.uuid4().hex[:8]}__"
                mask_store[mask_id] = (content, c_type)
                return f"\n{mask_id}\n"

            masked_text = pat.sub(replacer, masked_text)

        return masked_text, mask_store

    # -----------------------------------------------------------------
    # B. NORMALIZATION (Strict Mode)
    # -----------------------------------------------------------------

    def _normalize_headers(self, text: str) -> str:
        """
        Strict Normalization:
        Only touches lines that ALREADY start with #.
        Does NOT promote list items (1., I., A.) to headers.
        """
        lines = text.split("\n")
        new_lines = []

        # Only match lines that explicitly start with # (1 to 6 times)
        # This regex ensures we only capture valid markdown headers
        md_pat = re.compile(r"^\s*(#{1,6})\s+(.*)")

        for line in lines:
            line_str = line.strip()

            # Skip empty or mask lines
            if not line_str or (
                line_str.startswith("__MASK_") and line_str.endswith("__")
            ):
                new_lines.append(line)
                continue

            # Standardize spacing for existing headers
            # e.g., "##Title" becomes "## Title"
            m_md = md_pat.match(line_str)
            if m_md:
                new_lines.append(f"{m_md.group(1)} {m_md.group(2)}")
                continue

            # If it's not a header, leave it exactly as is (Prose/List items)
            new_lines.append(line)

        return "\n".join(new_lines)

    # -----------------------------------------------------------------
    # C. HIERARCHY SPLITTING
    # -----------------------------------------------------------------

    def _split_hierarchical_sections(self, text: str) -> List[Dict[str, Any]]:
        """Splits text by headers, maintaining parent/child context."""
        lines = text.split("\n")
        sections = []
        current_content = []
        header_stack = {}  # {level: title}
        current_level = 0

        for line in lines:
            # STRICT CHECK: Must start with # to be a splitter
            match = re.match(r"^(#{1,6})\s+(.*)", line)

            is_mask = "__MASK_" in line and line.strip().endswith("__")

            if match and not is_mask:
                # -- SAVE PREVIOUS SECTION --
                if current_content:
                    header_text = header_stack.get(current_level, "Root")
                    # Find nearest parent
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

                # -- START NEW SECTION --
                hashes, title = match.groups()
                new_level = len(hashes)
                current_level = new_level
                header_stack[new_level] = title.strip()

                # Wipe deeper levels
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

    # -----------------------------------------------------------------
    # D. BLOCK PROCESSING
    # -----------------------------------------------------------------

    def _process_section_blocks(
        self, section: Dict[str, Any], mask_store: Dict[str, Tuple[str, ContentType]]
    ) -> List[Dict[str, Any]]:
        masked_text = section["content"]
        if not masked_text:
            return []

        # Split prose by masks
        mask_pat = re.compile(r"(__MASK_[0-9a-f]+__)")
        parts = mask_pat.split(masked_text)
        final_chunks = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Case A: Mask (Restore content)
            if part in mask_store:
                original_content, c_type = mask_store[part]

                # Handle large tables
                if (
                    c_type == ContentType.TABLE
                    and count_tokens(original_content) > self.config.max_chunk_tokens
                ):
                    sub_texts = (
                        self.splitter.split_text(original_content)
                        if hasattr(self.splitter, "split_text")
                        else [original_content]
                    )
                else:
                    sub_texts = [original_content]

                for t in sub_texts:
                    final_chunks.append(
                        {
                            "content": t,
                            "metadata": {
                                "section_header": section["header"],
                                "parent_section": section["parent"],
                                "top_header": section["top_header"],
                                "content_type": c_type.value,
                            },
                            "estimated_tokens": count_tokens(t),
                        }
                    )

            # Case B: Prose (Split text)
            else:
                if hasattr(self.splitter, "split_text"):
                    sub_texts = self.splitter.split_text(part)
                else:
                    sub_texts = [part]

                for t in sub_texts:
                    final_chunks.append(
                        {
                            "content": t,
                            "metadata": {
                                "section_header": section["header"],
                                "parent_section": section["parent"],
                                "top_header": section["top_header"],
                                "content_type": ContentType.PROSE.value,
                            },
                            "estimated_tokens": count_tokens(t),
                        }
                    )

        return final_chunks

    # -----------------------------------------------------------------
    # E. MAIN PROCESS
    # -----------------------------------------------------------------

    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        # 1. Masking (Protection)
        masked_text, mask_store = self._mask_content(markdown_text)

        # 2. Normalize (Strict)
        norm_text = self._normalize_headers(masked_text)

        # 3. Split Sections
        sections = self._split_hierarchical_sections(norm_text)

        # 4. Block Processing & Unmasking
        all_chunks = []
        for sec in sections:
            all_chunks.extend(self._process_section_blocks(sec, mask_store))

        # 5. Merge
        if self.config.merge_short_chunks:
            all_chunks = self._merge_chunks(all_chunks)

        # 6. IDs
        return self._add_ids(all_chunks)

    def _merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        merged = []
        buffer = None

        for chunk in chunks:
            if not buffer:
                buffer = chunk
                continue

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


def chunk_markdown(text: str, **kwargs) -> List[Dict[str, Any]]:
    config = ChunkConfig(**kwargs)
    return MarkdownChunker(config).process(text)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING STRICT CHUNKING (README SAMPLE)")
    print("=" * 60)

    # Use Raw String (r"") to handle Windows paths in the text
    readme_sample = textwrap.dedent(r"""
        # 🚀 Setup Instructions

        ## ⚡ Fast Setup

        ### Step 1: Backend Setup
        ```powershell
        cd "f:\FYP WEB\backend"
        npm install
        ```
        ✅ Backend running...

        ### Step 2: Frontend Setup
        ```powershell
        cd "f:\FYP WEB\my-fyp-app"
        ```

        ## 🔑 Get OpenAI API Key

        1. Go to https://platform.openai.com
        2. Sign up / Login
        3. Go to API Keys section

        ## 📝 Test the Application

        1. Open browser: **http://localhost:5173**
        2. Click "Sign up"
    """)

    chunks = chunk_markdown(readme_sample)

    for i, c in enumerate(chunks):
        m = c["metadata"]
        content_preview = c["content"].replace("\n", "\\n")[:50]

        print(f"[{i}] {m['content_type'].upper()}")
        print(f"    Header: '{m['section_header']}'")
        print(f"    Content: {content_preview}...")
        print("-" * 40)

    # VERIFICATION
    # We check if "Get OpenAI API Key" exists and contains the numbered list as CONTENT, not headers
    key_chunk = next(
        (c for c in chunks if "Get OpenAI API Key" in c["metadata"]["section_header"]),
        None,
    )

    if key_chunk and "1. Go to" in key_chunk["content"]:
        print(
            "\n✅ SUCCESS: Numbered lists were PRESERVED as content (not converted to headers)."
        )
    else:
        print("\n❌ FAILURE: Numbered lists are missing or converted incorrectly.")
