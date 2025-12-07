import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, using word-based approximation")


# -------------------------------------------------------------------
# Token counter
# -------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, otherwise approximate."""
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: approximate tokens as 1.3x words for technical text
    return max(1, int(len(text.split()) * 1.3))


# -------------------------------------------------------------------
# Content type detection
# -------------------------------------------------------------------
class ContentType(Enum):
    CODE = "code"
    PROSE = "prose"
    MIXED = "mixed"


@dataclass
class ContentBlock:
    """Represents a block of content with its type and boundaries."""

    content: str
    type: ContentType
    start_idx: int
    end_idx: int


def detect_content_blocks(text: str) -> List[ContentBlock]:
    """
    Detect code blocks and prose sections in markdown text.
    Returns list of ContentBlock objects preserving order.
    """
    blocks = []
    code_pattern = r"```[\s\S]*?```"

    last_end = 0
    for match in re.finditer(code_pattern, text):
        start, end = match.span()

        # Add prose before this code block
        if start > last_end:
            prose = text[last_end:start].strip()
            if prose:
                blocks.append(
                    ContentBlock(
                        content=prose,
                        type=ContentType.PROSE,
                        start_idx=last_end,
                        end_idx=start,
                    )
                )

        # Add code block
        blocks.append(
            ContentBlock(
                content=match.group(0),
                type=ContentType.CODE,
                start_idx=start,
                end_idx=end,
            )
        )
        last_end = end

    # Add remaining prose after last code block
    if last_end < len(text):
        prose = text[last_end:].strip()
        if prose:
            blocks.append(
                ContentBlock(
                    content=prose,
                    type=ContentType.PROSE,
                    start_idx=last_end,
                    end_idx=len(text),
                )
            )

    # If no code blocks found, treat entire text as prose
    if not blocks:
        blocks.append(
            ContentBlock(
                content=text, type=ContentType.PROSE, start_idx=0, end_idx=len(text)
            )
        )

    return blocks


# -------------------------------------------------------------------
# Smart splitting functions
# -------------------------------------------------------------------
def split_code_block(code: str, max_tokens: int) -> List[str]:
    """
    Split code block intelligently, trying to preserve complete functions.
    Only splits if absolutely necessary.
    """
    # If it fits, return as-is
    if count_tokens(code) <= max_tokens:
        return [code]

    chunks = []

    # Try to extract code fence markers
    fence_match = re.match(r"^(```\w*\n)([\s\S]*?)(```\s*)$", code)
    if fence_match:
        opening, code_content, closing = fence_match.groups()
    else:
        opening, code_content, closing = "", code, ""

    # Split by function definitions (def, class, etc.)
    function_pattern = r"(^(?:def|class|async def)\s+\w+.*?:\n(?:(?!^(?:def|class|async def)\s+\w+).*\n)*)"
    functions = re.findall(function_pattern, code_content, re.MULTILINE)

    if functions:
        # Split by functions
        current_chunk = opening

        for func in functions:
            test_chunk = current_chunk + func
            if (
                count_tokens(test_chunk + closing) > max_tokens
                and current_chunk != opening
            ):
                # Save current chunk and start new one
                chunks.append(current_chunk + closing)
                current_chunk = opening + func
            else:
                current_chunk = test_chunk

        if current_chunk != opening:
            chunks.append(current_chunk + closing)
    else:
        # Fallback: split by lines, trying to keep logical groups
        lines = code_content.split("\n")
        current_chunk = opening

        for line in lines:
            test_chunk = current_chunk + line + "\n"
            if (
                count_tokens(test_chunk + closing) > max_tokens
                and current_chunk != opening
            ):
                chunks.append(current_chunk + closing)
                current_chunk = opening + line + "\n"
            else:
                current_chunk = test_chunk

        if current_chunk != opening:
            chunks.append(current_chunk + closing)

    return chunks if chunks else [code]


def split_prose_by_sentences(text: str, max_tokens: int) -> List[str]:
    """Split prose by sentences, respecting token limits."""
    # Improved sentence splitting that handles abbreviations better
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    current = ""

    for sentence in sentences:
        test_chunk = (current + " " + sentence).strip()

        if count_tokens(test_chunk) > max_tokens:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = test_chunk

    if current:
        chunks.append(current.strip())

    return chunks if chunks else [text]


def split_by_paragraphs(text: str, max_tokens: int) -> List[str]:
    """Split text by paragraphs first, then by sentences if needed."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if count_tokens(para) <= max_tokens:
            chunks.append(para)
        else:
            # Paragraph too large, split by sentences
            chunks.extend(split_prose_by_sentences(para, max_tokens))

    return chunks


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
@dataclass
class ChunkConfig:
    max_chunk_tokens: int = 300
    min_chunk_tokens: int = 50
    preserve_code_blocks: bool = True
    merge_short_chunks: bool = True


# -------------------------------------------------------------------
# Main Chunker
# -------------------------------------------------------------------
class MarkdownChunker:
    """
    Content-aware chunker that:
    - Preserves code blocks
    - Splits intelligently by content type
    - Maintains header context
    - Merges short chunks smartly
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def process(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Main entry point for chunking."""
        sections = self._split_by_headers(markdown_text)
        chunks = self._process_sections(sections)

        if self.config.merge_short_chunks:
            chunks = self._merge_short_chunks(chunks)

        chunks = self._assign_chunk_indices(chunks)
        return chunks

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """Split text by markdown headers while preserving hierarchy."""
        sections = []

        # Pattern to match headers and capture their level and text
        header_pattern = r"^(#{1,6})\s+(.+?)$"

        lines = text.split("\n")
        current_section = {"content": [], "header": None, "level": 0, "parent": None}

        header_stack = [None, None, None, None, None, None]  # Track hierarchy

        for line in lines:
            match = re.match(header_pattern, line)

            if match:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append(
                        {
                            "content": "\n".join(current_section["content"]),
                            "header": current_section["header"],
                            "level": current_section["level"],
                            "parent": current_section["parent"],
                        }
                    )

                # Start new section
                level = len(match.group(1))
                header_text = match.group(2).strip()

                # Update header stack
                header_stack[level - 1] = header_text
                for i in range(level, 6):
                    header_stack[i] = None

                # Find parent (first non-None header at lower level)
                parent = None
                for i in range(level - 2, -1, -1):
                    if header_stack[i]:
                        parent = header_stack[i]
                        break

                current_section = {
                    "content": [line],
                    "header": header_text,
                    "level": level,
                    "parent": parent,
                }
            else:
                current_section["content"].append(line)

        # Don't forget the last section
        if current_section["content"]:
            sections.append(
                {
                    "content": "\n".join(current_section["content"]),
                    "header": current_section["header"],
                    "level": current_section["level"],
                    "parent": current_section["parent"],
                }
            )

        return sections

    def _process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process each section, splitting by content type and token limits."""
        all_chunks = []

        for section in sections:
            content = section["content"]

            # Detect content blocks (code vs prose)
            blocks = detect_content_blocks(content)

            for block in blocks:
                chunks = self._split_block(block)

                for chunk_text in chunks:
                    all_chunks.append(
                        {
                            "content": chunk_text,
                            "metadata": {
                                "section_header": section["header"],
                                "section_level": section["level"],
                                "parent_section": section["parent"],
                                "content_type": block.type.value,
                            },
                            "estimated_tokens": count_tokens(chunk_text),
                        }
                    )

        return all_chunks

    def _split_block(self, block: ContentBlock) -> List[str]:
        """Split a content block based on its type."""
        max_tokens = self.config.max_chunk_tokens

        if block.type == ContentType.CODE:
            if self.config.preserve_code_blocks:
                return split_code_block(block.content, max_tokens)
            else:
                # Treat as prose if not preserving
                return split_by_paragraphs(block.content, max_tokens)
        else:  # PROSE or MIXED
            return split_by_paragraphs(block.content, max_tokens)

    def _merge_short_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge short chunks intelligently."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # Check if current chunk is too short
            if current[
                "estimated_tokens"
            ] < self.config.min_chunk_tokens and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]

                # Check if they're from the same section
                same_section = (
                    current["metadata"]["section_header"]
                    == next_chunk["metadata"]["section_header"]
                )

                # Check if merged size would be reasonable
                merged_tokens = (
                    current["estimated_tokens"] + next_chunk["estimated_tokens"]
                )
                fits = (
                    merged_tokens <= self.config.max_chunk_tokens * 1.2
                )  # Allow 20% overflow

                if same_section and fits:
                    # Merge current into next
                    merged_content = current["content"] + "\n\n" + next_chunk["content"]
                    next_chunk["content"] = merged_content
                    next_chunk["estimated_tokens"] = count_tokens(merged_content)
                    i += 1  # Skip current, process next in next iteration
                    continue

            merged.append(current)
            i += 1

        return merged

    def _assign_chunk_indices(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assign sequential indices to chunks."""
        for idx, chunk in enumerate(chunks):
            chunk["chunk_index"] = idx
        return chunks


# -------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------
def chunk_markdown(
    markdown_text: str, max_tokens: int = 300, min_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk markdown text.

    Args:
        markdown_text: The markdown text to chunk
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk (for merging)

    Returns:
        List of chunk dictionaries with content, metadata, and token counts
    """
    config = ChunkConfig(max_chunk_tokens=max_tokens, min_chunk_tokens=min_tokens)
    chunker = MarkdownChunker(config)
    return chunker.process(markdown_text)


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_md = """
# Recursive Reasoning with Tiny Networks

## Algorithms with different number of latent features

```python
def latent_recursion(x, z, n=6):
    for i in range(n+1):  # latent recursion
        z = net(x, z)
    return z

def deep_recursion(x, z, n=6, T=3):
    # recursing T-1 times to improve z (no gradients needed)
    with torch.no_grad():
        for j in range(T-1):
            z = latent_recursion(x, z, n)
    # recursing once to improve z
    z = latent_recursion(x, z, n)
    return z.detach(), output_head(y), Q_head(y)
```

This approach allows for flexible handling of latent features.

## Example on Sudoku-Extreme

The following figure shows results on a challenging Sudoku puzzle.
"""

    chunks = chunk_markdown(sample_md, max_tokens=200)

    print(f"Generated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']}:")
        print(f"  Tokens: {chunk['estimated_tokens']}")
        print(f"  Header: {chunk['metadata']['section_header']}")
        print(f"  Type: {chunk['metadata']['content_type']}")
        print(f"  Content preview: {chunk['content'][:100]}...")
        print()
