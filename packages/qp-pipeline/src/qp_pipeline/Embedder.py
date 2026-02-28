"""
Embedder.py - Vector Indexing Pass for Enriched QA Pairs

Runs as an independent second stage after Enricher.py has completed. Reads
accepted QA pairs from the database, generates embeddings, and upserts them
into the Chroma vector store.

Why separate from Enricher:
- An embedding failure (Chroma down, OOM, etc.) does not affect enrichment.
- Either stage can be re-run independently without re-doing LLM work.
- Keeps EnrichmentManager focused on generation and validation only.

What is embedded:
- The question text is the primary document pushed to Chroma — this is what
  retrieval queries will match against at inference time.
- The answer, source_quote, difficulty, question_type, and tags are stored
  as Chroma metadata for filtering and display.

Deduplication strategy:
- Before indexing, the question_id is checked against the existing Chroma
  collection. If it already exists, it is skipped. This makes every run
  idempotent — safe to re-run after partial failures or new enrichment.

Public entry points on VectorIndexer:
- index_chunk(chunk_id)  — index all QA pairs for one chunk.
- index_file(file_id)    — index all QA pairs for one file.
- run()                  — index all enriched files in the DB.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Set

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# ---------------- ROBUST PROJECT ROOT DETECTION ----------------
_env_root = os.environ.get("RAG_PROJECT_ROOT")
if _env_root:
    project_root = Path(_env_root).resolve()
else:
    project_root = Path(__file__).resolve().parents[4]

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DB_PATH = str(project_root / "data" / "rag_staging.db")
CHROMA_DIR = str(project_root / "data" / "chroma_store")

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

try:
    from qp_core.DBManager import DBManager
    from qp_core.VectorStore import QAVectorStore
except ImportError:
    print("Import errors")

# ---------------- RICH LOGGING SETUP ----------------
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("Embedder")


# ---------------- VECTOR INDEXER ----------------
class VectorIndexer:
    """
    Reads enriched QA pairs from the DB and pushes them to the Chroma vector
    store. Every operation is idempotent — question_ids that already exist in
    Chroma are skipped automatically.
    """

    def __init__(self):
        logger.info("🔷 Starting Vector Indexer...")
        self.db = DBManager(DB_PATH)
        self.vector_store = QAVectorStore(chroma_path=CHROMA_DIR)

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================
    def _get_indexed_ids(self) -> Set[str]:
        """
        Query Chroma for all document IDs currently in the collection.

        This is used to skip already-indexed questions on every run, making
        the indexer fully idempotent without needing an extra DB column.

        Returns:
            Set of question_id strings already present in Chroma.
        """
        try:
            result = self.vector_store.collection.get(include=[])
            return set(result.get("ids", []))
        except Exception as e:
            logger.warning(f"Could not fetch existing Chroma IDs — will index all: {e}")
            return set()

    def _index_questions(self, questions: list, indexed_ids: Set[str]) -> tuple:
        """
        Push a list of QA dicts to Chroma, skipping any whose question_id
        already exists in the indexed_ids set.

        Args:
            questions:   list of QA dicts from DBManager.get_questions_for_*
            indexed_ids: set of question_ids already in Chroma

        Returns:
            (added: int, skipped: int)
        """
        added = 0
        skipped = 0

        for qa in questions:
            qid = qa.get("question_id", "")
            if not qid:
                logger.warning("QA row has no question_id — skipping")
                skipped += 1
                continue

            if qid in indexed_ids:
                logger.debug(f"Already indexed: {qid[:16]}")
                skipped += 1
                continue

            try:
                self.vector_store.add_qa_pair(
                    chunk_id=qa["chunk_id"],
                    question_text=qa["question_text"],
                    answer_text=qa["answer_text"],
                    source_quote=qa.get("source_quote", ""),
                    difficulty=qa.get("difficulty", "Medium"),
                    question_type=qa.get("question_type", "Fact"),
                    tags=qa.get("tags", []),
                    # Pass the question_id so Chroma uses it as the document ID,
                    # enabling the idempotency check above. QAVectorStore must
                    # accept and forward this as the `ids` parameter to collection.add.
                    question_id=qid,
                )
                indexed_ids.add(qid)  # update local set so this run doesn't re-add
                added += 1
            except Exception as e:
                logger.error(f"Failed to index {qid[:16]}: {e}")
                skipped += 1

        return added, skipped

    # =========================================================
    # PUBLIC: index_chunk
    # =========================================================
    def index_chunk(self, chunk_id: str) -> None:
        """
        Index all accepted QA pairs belonging to a single chunk.

        Safe to call multiple times — already-indexed questions are skipped.

        Args:
            chunk_id: UUID of the chunk to index
        """
        console.print(
            Panel(
                f"[bold]Chunk:[/] {chunk_id[:8]}",
                title="[bold blue]🔷 Indexing Chunk[/]",
                expand=False,
            )
        )

        questions = self.db.get_questions_for_chunk(chunk_id)
        indexed_ids = self._get_indexed_ids()

        if not questions:
            logger.warning(f"No QA pairs found for chunk {chunk_id[:8]}")
            return

        added, skipped = self._index_questions(questions, indexed_ids)
        self.vector_store.persist()

        logger.info(
            f"[green]✅ Chunk {chunk_id[:8]}[/] — "
            f"added=[green]{added}[/]  skipped=[dim]{skipped}[/]"
        )

    # =========================================================
    # PUBLIC: index_file
    # =========================================================
    def index_file(self, file_id: str) -> None:
        """
        Index all accepted QA pairs for every chunk in a file.

        Fetches existing Chroma IDs once per file to avoid repeated round-trips.
        Safe to call multiple times — already-indexed questions are skipped.

        Args:
            file_id: UUID of the file to index
        """
        console.print(
            Panel(
                f"[bold]File:[/] {file_id[:8]}",
                title="[bold blue]📂 Indexing File[/]",
                expand=False,
            )
        )

        questions = self.db.get_questions_for_file(file_id)
        indexed_ids = self._get_indexed_ids()

        if not questions:
            logger.warning(f"No QA pairs found for file {file_id[:8]}")
            return

        logger.info(f"Found {len(questions)} QA pair(s) to consider")
        added, skipped = self._index_questions(questions, indexed_ids)
        self.vector_store.persist()

        logger.info(
            f"[green]✅ File {file_id[:8]} indexed[/] — "
            f"added=[green]{added}[/]  skipped=[dim]{skipped}[/]"
        )

    # =========================================================
    # PUBLIC: run
    # =========================================================
    def run(self) -> None:
        """
        Index all enriched files in the DB.

        Fetches the full set of existing Chroma IDs once at the start of the
        run rather than once per file — this avoids redundant Chroma round-trips
        when processing many files in sequence.

        Already-indexed questions are skipped throughout. Adding new files to
        the DB and re-running this is the normal incremental workflow.
        """
        console.print(
            Panel(
                "[bold green]Vector Indexing Pass[/]\n"
                f"DB:     [cyan]{DB_PATH}[/]\n"
                f"Chroma: [cyan]{CHROMA_DIR}[/]",
                title="[bold]🚀 Starting Embedder[/]",
                expand=False,
            )
        )

        file_ids = self.db.get_all_enriched_file_ids()
        indexed_ids = self._get_indexed_ids()

        if not file_ids:
            console.print(
                Panel(
                    "[yellow]No enriched files found in DB.[/]",
                    title="ℹ️  Nothing to index",
                    expand=False,
                )
            )
            return

        logger.info(
            f"Found [bold]{len(file_ids)}[/] enriched file(s) | "
            f"[dim]{len(indexed_ids)} already indexed in Chroma[/]"
        )

        total_added = 0
        total_skipped = 0

        for file_id in file_ids:
            questions = self.db.get_questions_for_file(file_id)
            added, skipped = self._index_questions(questions, indexed_ids)
            total_added += added
            total_skipped += skipped

            # Summary row per file
            summary_table = Table.grid(padding=(0, 2))
            summary_table.add_row(
                f"[green]Added: {added}[/]",
                f"[dim]Skipped: {skipped}[/]",
                f"[dim]Total QA: {len(questions)}[/]",
            )
            console.print(
                Panel(
                    summary_table,
                    title=f"[blue]File {file_id[:8]}[/]",
                    expand=False,
                )
            )

        # Persist once at the end of the full run rather than per-file
        self.vector_store.persist()

        console.print(
            Panel(
                f"[green]Total added:[/]   [bold]{total_added}[/]\n"
                f"[dim]Total skipped:[/] [bold]{total_skipped}[/]",
                title="[bold green]✅ Indexing Complete[/]",
                expand=False,
            )
        )


# Entry point for manual runs.
if __name__ == "__main__":
    console.print(
        Panel(
            "[bold cyan]Embedder.py[/] — Standalone Vector Indexing Pass\n"
            "Reads from SQLite → pushes to Chroma",
            title="[bold]🔷 Embedder[/]",
            expand=False,
        )
    )
    VectorIndexer().run()
