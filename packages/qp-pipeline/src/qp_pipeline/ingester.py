"""
ingester.py - Fixed with Batch Processing and Aggressive Chunking
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Path resolving for internal packages
current_file = Path(__file__).resolve()
project_root = current_file.parents[4]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qp_core.DBManager import DBManager
from qp_core.IDGenerator import IDGenerator
from qp_core.SimHashHandler import SimHashHandler

from qp_pipeline.ChunkEvaluator import ChunkEvaluator
from qp_pipeline.docling_ocr import PDFDocumentConverter
from qp_pipeline.MarkdownChunker import ChunkConfig, MarkdownChunker

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingester")


def get_default_db_path():
    return project_root / "data" / "rag_staging.db"


def ingest(
    file_path,
    db_manager,
    converter,
    chunker,
    evaluator,
    id_generator,
    simhash_handler,
    auto_confirm=False,
):
    """Refactored ingest logic to be callable by both single and batch modes."""
    file_path = Path(file_path)

    # Step 1: Convert
    try:
        metadata, markdown = converter.process_document(str(file_path))
        if not markdown:
            logger.error(f"Empty markdown for {file_path}")
            return False
    except Exception as e:
        logger.error(f"Conversion failed for {file_path}: {e}")
        return False

    # Step 2: Duplicate Detection
    duplicate_check = simhash_handler.check_duplicate(markdown)
    if duplicate_check["is_duplicate"]:
        logger.warning(f"Duplicate detected: {file_path}")
        if not auto_confirm:
            if input("Continue anyway? (y/n): ").lower() != "y":
                return False

    # Step 3 & 4: ID and Chunking
    file_id = id_generator.generate_file_id()
    chunks = chunker.process(markdown)

    # Step 5: Evaluate
    eval_stats = evaluator.evaluate_chunks(chunks)
    evaluated_chunks = eval_stats["evaluated_chunks"]

    # Step 6 & 7: Database Save
    try:
        db_manager.save_file_metadata(
            file_id=file_id,
            file_path=str(file_path),
            simhash=duplicate_check["simhash"],
            metadata=metadata,
            content_length=len(markdown),
        )
        db_manager.save_chunks(file_id=file_id, chunks=evaluated_chunks)
        simhash_handler.add_to_index(file_id, duplicate_check["simhash"])
        print("Chunks saved")
        return True
    except Exception as e:
        logger.error(f"DB Error for {file_path}: {e}")
        return False


def batch_process(db_path=None):
    """Processes all PDFs in a specified directory."""
    if db_path is None:
        db_path = get_default_db_path()

    dir_path = input("\nEnter directory path to scan for PDFs: ")
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        print("❌ Invalid directory.")
        return

    files = list(p.glob("*.pdf"))
    print(f"\n📂 Found {len(files)} PDF files in {dir_path}")

    # Initialize shared components
    converter = PDFDocumentConverter()
    # Apply aggressive merging config
    chunk_config = ChunkConfig(max_chunk_tokens=1000, merge_short_chunks=True)
    chunker = MarkdownChunker(chunk_config)
    evaluator = ChunkEvaluator()
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)
    db_manager = DBManager(db_path=str(db_path))

    # Load existing hashes
    existing = (
        db_manager.get_all_simhashes()
        if hasattr(db_manager, "get_all_simhashes")
        else {}
    )
    simhash_handler.load_index_from_data(existing)

    success_count = 0
    for i, file_path in enumerate(files):
        print(f"\n[{i + 1}/{len(files)}] Processing: {file_path.name}")
        # In batch mode, we auto_confirm=True to avoid blocking prompts
        if ingest(
            file_path,
            db_manager,
            converter,
            chunker,
            evaluator,
            id_generator,
            simhash_handler,
            auto_confirm=True,
        ):
            success_count += 1
            print(f"✅ Successfully processed.")
        else:
            print(f"⚠️ Skipped or failed.")

    print(
        f"\n{'=' * 30}\nBATCH COMPLETE\nSuccess: {success_count}/{len(files)}\n{'=' * 30}"
    )


if __name__ == "__main__":
    target_db_path = get_default_db_path()
    mode = input("\nSelect mode:\n 1. Single file\n 2. Batch processing\nChoice: ")

    if mode == "1":
        # Single file flow
        converter = PDFDocumentConverter()
        chunker = MarkdownChunker(ChunkConfig(max_chunk_tokens=1000))
        evaluator = ChunkEvaluator()
        id_generator = IDGenerator()
        simhash_handler = SimHashHandler()
        db_manager = DBManager(db_path=str(target_db_path))

        path = input("File path: ")
        ingest(
            path,
            db_manager,
            converter,
            chunker,
            evaluator,
            id_generator,
            simhash_handler,
        )
    elif mode == "2":
        batch_process(db_path=target_db_path)
