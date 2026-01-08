# import json
# import re
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional


# @dataclass
# class ToCEntry:
#     """Represents a single entry in the table of contents."""

#     title: str
#     page: Optional[int]
#     level: int
#     numeric_id: Optional[str]  # e.g., "1.2.3"
#     parent_id: Optional[str]
#     children: List["ToCEntry"]
#     full_path: str  # e.g., "CHAPTER 1 > INTRODUCTION > Background"

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary, recursively handling children."""
#         result = {
#             "title": self.title,
#             "page": self.page,
#             "level": self.level,
#             "numeric_id": self.numeric_id,
#             "parent_id": self.parent_id,
#             "full_path": self.full_path,
#             "children": [child.to_dict() for child in self.children],
#         }
#         return result


# class ToCParser:
#     """Parse table of contents from various markdown formats."""

#     def __init__(self):
#         self.entries: List[ToCEntry] = []
#         self.flat_entries: List[ToCEntry] = []

#     def parse(self, toc_text: str) -> Dict[str, Any]:
#         """
#         Main parsing method that handles both table and list formats.
#         Args:
#             toc_text: Raw table of contents text
#         Returns:
#             Structured dictionary with hierarchical and flat representations
#         """
#         # Clean the text
#         toc_text = self._preprocess_text(toc_text)
#         # Detect format and parse accordingly
#         if self._is_table_format(toc_text):
#             self.entries = self._parse_table_format(toc_text)
#         else:
#             self.entries = self._parse_list_format(toc_text)
#         # Build hierarchy
#         self.entries = self._build_hierarchy(self.entries)
#         # Generate flat list with full paths
#         self.flat_entries = self._flatten_with_paths(self.entries)
#         return {
#             "hierarchical": [entry.to_dict() for entry in self.entries],
#             "flat": [entry.to_dict() for entry in self.flat_entries],
#             "metadata": self._generate_metadata(),
#         }

#     def _preprocess_text(self, text: str) -> str:
#         """Clean and normalize the text, preserving newlines."""
#         # Remove HTML entities
#         text = text.replace("&amp;", "&")
#         text = text.replace("&lt;", "<")
#         text = text.replace("&gt;", ">")
#         # Normalize horizontal whitespace only
#         text = re.sub(r"[ \t]+", " ", text)
#         # Normalize dot leaders
#         text = re.sub(r"\s*\.\s*\.", "..", text)
#         return text.strip()

#     def _is_table_format(self, text: str) -> bool:
#         """Detect if ToC is in table format (uses | delimiters)."""
#         return "|" in text and text.count("|") > 3

#     def _parse_table_format(self, text: str) -> List[ToCEntry]:
#         """Parse table-style ToC."""
#         entries = []
#         lines = text.split("\n")
#         for line in lines:
#             line = line.strip()
#             # Skip table separators and headers
#             if (
#                 not line
#                 or line.startswith("|---")
#                 or "TABLE OF CONTENTS" in line.upper()
#             ):
#                 continue
#             # Remove leading/trailing pipes
#             line = line.strip("|").strip()
#             if not line:
#                 continue
#             # Parse the entry
#             entry = self._parse_line(line)
#             if entry:
#                 entries.append(entry)
#         return entries

#     def _parse_list_format(self, text: str) -> List[ToCEntry]:
#         """Parse list-style ToC (markdown bullets/dashes)."""
#         entries = []
#         lines = text.split("\n")
#         for line in lines:
#             line = line.strip()
#             # Skip headers
#             if not line or "TABLE OF CONTENTS" in line.upper() or line.startswith("#"):
#                 continue
#             # Iteratively remove list markers and count level
#             level = 0
#             clean_line = line
#             while re.match(r"^[-*+]\s*", clean_line):
#                 clean_line = re.sub(r"^[-*+]\s*", "", clean_line)
#                 level += 1
#             # Adjust level to start from 0
#             adjusted_level = level - 1 if level > 0 else 0
#             # Parse the entry
#             entry = self._parse_line(clean_line, force_level=adjusted_level)
#             if entry:
#                 entries.append(entry)
#         return entries

#     def _parse_line(
#         self, line: str, force_level: Optional[int] = None
#     ) -> Optional[ToCEntry]:
#         """Parse a single ToC line into a ToCEntry."""
#         # Pattern 1: Numeric format (1.2.3. Title ... Page)
#         numeric_pattern = r"^(\d+(?:\.\d+)*\.?)\s+(.+?)(?:\s*\.{2,}\s*(\d+))?$"
#         match = re.match(numeric_pattern, line)
#         if match:
#             numeric_id = match.group(1).rstrip(".")
#             title = match.group(2).strip()
#             page = int(match.group(3)) if match.group(3) else None
#             level = len(numeric_id.split("."))
#             # Extract parent ID
#             parts = numeric_id.split(".")
#             parent_id = ".".join(parts[:-1]) if len(parts) > 1 else None
#             return ToCEntry(
#                 title=title,
#                 page=page,
#                 level=level,
#                 numeric_id=numeric_id,
#                 parent_id=parent_id,
#                 children=[],
#                 full_path=title,
#             )
#         # Pattern 2: Chapter format (CHAPTER 1, ABSTRACT, etc.)
#         chapter_pattern = r"^(CHAPTER\s+(\d+)|ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|APPENDIX)(?:\s*\.{2,}\s*(\d+))?$"
#         match = re.match(chapter_pattern, line, re.IGNORECASE)
#         if match:
#             title_group = match.group(1)
#             chapter_num = match.group(2)
#             page_group = match.group(3)
#             title = title_group.strip()
#             page = int(page_group) if page_group else None
#             numeric_id = chapter_num if chapter_num else None
#             return ToCEntry(
#                 title=title,
#                 page=page,
#                 level=0,
#                 numeric_id=numeric_id,
#                 parent_id=None,
#                 children=[],
#                 full_path=title,
#             )
#         # Pattern 3: Simple title format (for list-style ToCs)
#         simple_pattern = r"^(.+?)(?:\s*\.{2,}\s*(\d+))?$"
#         match = re.match(simple_pattern, line)
#         if match and force_level is not None:
#             title = match.group(1).strip()
#             page = int(match.group(2)) if match.group(2) else None
#             return ToCEntry(
#                 title=title,
#                 page=page,
#                 level=force_level,
#                 numeric_id=None,
#                 parent_id=None,
#                 children=[],
#                 full_path=title,
#             )
#         return None

#     def _build_hierarchy(self, entries: List[ToCEntry]) -> List[ToCEntry]:
#         """Build parent-child relationships between entries."""
#         has_numeric_ids = any(e.numeric_id is not None for e in entries)
#         if has_numeric_ids:
#             # Use ID mapping for numeric-based hierarchy
#             root_entries = []
#             entry_map = {e.numeric_id: e for e in entries if e.numeric_id}
#             for entry in entries:
#                 if entry.parent_id and entry.parent_id in entry_map:
#                     entry_map[entry.parent_id].children.append(entry)
#                 else:
#                     root_entries.append(entry)
#             return root_entries
#         else:
#             # Use stack for level-based hierarchy
#             root_entries = []
#             stack: List[ToCEntry] = []
#             for entry in entries:
#                 while stack and stack[-1].level >= entry.level:
#                     stack.pop()
#                 if stack:
#                     stack[-1].children.append(entry)
#                 else:
#                     root_entries.append(entry)
#                 stack.append(entry)
#             return root_entries

#     def _flatten_with_paths(
#         self, entries: List[ToCEntry], parent_path: str = ""
#     ) -> List[ToCEntry]:
#         """Flatten hierarchy and build full paths."""
#         result = []
#         for entry in entries:
#             # Build full path
#             if parent_path:
#                 entry.full_path = f"{parent_path} > {entry.title}"
#             else:
#                 entry.full_path = entry.title
#             result.append(entry)
#             # Recursively process children
#             if entry.children:
#                 result.extend(self._flatten_with_paths(entry.children, entry.full_path))
#         return result

#     def _generate_metadata(self) -> Dict[str, Any]:
#         """Generate metadata about the ToC structure."""
#         return {
#             "total_entries": len(self.flat_entries),
#             "max_depth": max((e.level for e in self.flat_entries), default=0),
#             "chapters": sum(1 for e in self.flat_entries if e.level == 0),
#             "has_page_numbers": any(e.page is not None for e in self.flat_entries),
#             "has_numeric_ids": any(e.numeric_id is not None for e in self.flat_entries),
#         }

#     def export_for_chunker(self) -> Dict[str, Any]:
#         """
#         Export in a format optimized for the MarkdownChunker.
#         Returns a mapping structure that can be used to enrich chunk metadata.
#         """
#         section_map = {}
#         for entry in self.flat_entries:
#             # Create lookup keys for the chunker
#             keys = [entry.title.strip()]
#             if entry.numeric_id:
#                 keys.append(f"{entry.numeric_id}. {entry.title}")
#                 keys.append(f"{entry.numeric_id} {entry.title}")
#             # Store metadata for each key
#             for key in keys:
#                 section_map[key] = {
#                     "title": entry.title,
#                     "level": entry.level,
#                     "numeric_id": entry.numeric_id,
#                     "page": entry.page,
#                     "full_path": entry.full_path,
#                     "parent_id": entry.parent_id,
#                 }
#         return {
#             "section_map": section_map,
#             "hierarchy": [e.to_dict() for e in self.entries],
#             "metadata": self._generate_metadata(),
#         }


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Table format
#     toc_table = """
#     ## TABLE OF CONTENTS
#     | CHAPTER 1................................................................................................................................13 |
#     | INTRODUCTION....................................................................................................................13 |
#     | 1.1. Background..........................................................................................................13 |
#     | 1.2. Heading Level 2..................................................................................................13 |
#     | 1.2.1. Heading Level 3...........................................................................................13 |
#     | 1.3. Introduction.........................................................................................................14 |
#     """
#     normal_table = """
#     | Aspect                                                       | Option 1: Multiple / Concatenated Vectors                    | Option 2: Pure Knowledge Graph                               | Option 3: Hybrid (Knowledge Graph + Graph Neural Network) – Recommended |
#     | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
#     | Core idea                                                    | Embed each field separately (or concatenate) → store one or more dense vectors per company → query with cosine similarity | Explicit nodes & typed relationships (Company → has_industry → Tech, Company → offers → “ChatGPT”) → query with graph traversal (Cypher/SPARQL) | Build a KG first, then run a Graph Neural Network (e.g., GraphSAGE, GATv2, or FastRP + GNN) to generate rich node embeddings that respect both text semantics and graph structure |
#     | Semantic / fuzzy matching                                    | Excellent (handles synonyms, messy text)                     | Poor unless you add vector search on node properties         | Excellent (inherits text embeddings + learns structural patterns) |
#     | Explicit relational reasoning                                | No (only cosine similarity)                                  | Excellent (multi-hop paths, explainable)                     | Excellent (graph traversal still possible + embeddings capture learned multi-hop patterns) |
#     | Explainability                                               | Low (black-box similarity score)                             | Very high (you can return the exact path)                    | High (path explainability + similarity score)                |
#     | Build time & effort                                          | Very low (just embed fields)                                 | High (schema design + manual or LLM-assisted relation extraction) | Medium-high (KG first, then one-time GNN training)           |
#     | Maintenance when data changes                                | Extremely easy (re-embed only changed fields)                | Medium–hard (may need to add/remove edges, sometimes cascade) | Easy for new nodes (inductive GNNs like GraphSAGE embed unseen nodes instantly) |
#     | Inference latency                                            | Lowest (single vector lookup + ANN)                          | Medium–high (graph traversal)                                | Low (vector lookup) + optional traversal only when explanation is needed |
#     | Handles completely new companies / jobs                      | Yes (just embed the new record)                              | Yes, but you must insert node + edges correctly              | Best (inductive GNN instantly produces a high-quality embedding for the new node using its features and neighbors) |
#     | Typical accuracy in 2025 job-rec / company-similarity benchmarks | Baseline (simple sentence-transformer averages)              | Strong on exact relational queries, weaker on fuzzy          | State-of-the-art (+10–25 % over pure vector baselines and +8–18 % over transductive graph methods like Node2Vec on inductive tasks) |
#     """
#     # Example 2: List format
#     toc_list = """
#     ## Table of Contents
#     - Abstract
#     - Domain & Tools
#     - -Domain
#     - -Tools & Frameworks
#     - System Architecture
#     """
#     parser = ToCParser()
#     # Parse table format
#     result_table = parser.parse(toc_table)
#     print("Table Format Result:")
#     print(json.dumps(result_table, indent=2))
#     # Parse list format
#     result_list = parser.parse(toc_list)
#     print("\nList Format Result:")
#     print(json.dumps(result_list, indent=2))
#     # Parse normal table
#     print("\n normal table results")
#     result_normal = parser.parse(toc_text=normal_table)
#     print(json.dumps(obj=result_normal, indent=2))
#     # Export for chunker
#     chunker_map = parser.export_for_chunker()
#     print("\nChunker Integration Map:")
#     print(json.dumps(chunker_map, indent=2))
