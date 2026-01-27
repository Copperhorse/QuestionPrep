"""
Orchestrator.py
Complete document processing pipeline with SQLite Database storage
"""

import sqlite3  # Added for SimHash loading

# from Utils.CSVManager import CSVManager # ❌ REMOVED
from Utils.DBManager import (
    DBManager,  # ✅ ADDED (Assumes DBManager.py is in the same folder or Utils)
)

from Chunker.MarkdownChunker import ChunkConfig, MarkdownChunker
from Evaluator.ChunkEvaluator import ChunkEvaluator
from Extractor.docling_ocr import PDFDocumentConverter
from Utils.IDGenerator import IDGenerator
from Utils.SimHashHandler import SimHashHandler


def load_simhashes_from_db(db_path):
    """Helper to load existing simhashes from SQLite."""
    simhashes = {}
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        # Check if table exists first
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
        )
        if cur.fetchone():
            cur.execute("SELECT simhash, file_id FROM files WHERE simhash IS NOT NULL")
            for row in cur.fetchall():
                simhashes[row[0]] = row[1]
        con.close()
    except Exception as e:
        print(f" Warning: Could not load existing simhashes: {e}")
    return simhashes


def main():
    """Main orchestration function."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    print("\n Initializing components...")
    converter = PDFDocumentConverter()
    chunk_config = ChunkConfig(max_chunk_tokens=500)
    chunker = MarkdownChunker(chunk_config)
    evaluator = ChunkEvaluator()
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)

    # ✅ DB INIT
    DB_PATH = "rag_staging.db"
    db_manager = DBManager(db_path=DB_PATH)

    FILE_PATH = input("Enter the file path: ")

    # ------------------------------------------------------------------
    # Load existing simhashes
    # ------------------------------------------------------------------
    print(" Loading existing simhashes...")
    existing_simhashes = load_simhashes_from_db(DB_PATH)
    if existing_simhashes:
        simhash_handler.load_index_from_data(existing_simhashes)
        print(f" Loaded {len(existing_simhashes)} existing files into index")

    # ------------------------------------------------------------------
    # Step 1: Convert document
    # ------------------------------------------------------------------
    print("\n Step 1: Converting PDF to markdown...")
    metadata, markdown = converter.process_document(FILE_PATH)
    print(f" Converted successfully ({len(markdown)} characters)")

    # ------------------------------------------------------------------
    # Step 2: Duplicate detection
    # ------------------------------------------------------------------
    print("\n Step 2: Checking for duplicates...")
    duplicate_check = simhash_handler.check_duplicate(markdown)
    if duplicate_check["is_duplicate"]:
        print(" WARNING: This document appears to be a duplicate!")
        print(f" Similar to file IDs: {duplicate_check['duplicate_file_ids']}")
        if input("\n Continue processing anyway? (y/n): ").lower() != "y":
            print("\n Processing cancelled.")
            return
    else:
        print(f" No duplicates found (SimHash: {duplicate_check['simhash']})")

    # ------------------------------------------------------------------
    # Step 3: Generate file ID
    # ------------------------------------------------------------------
    print("\n Step 3: Generating file ID...")
    file_id = id_generator.generate_file_id()
    print(f" File ID: {file_id}")

    # ------------------------------------------------------------------
    # Step 4: Chunking
    # ------------------------------------------------------------------
    print("\n Step 4: Chunking document...")
    # Cleaned markdown is just the markdown in this pipeline
    chunks = chunker.process(markdown)
    print(f" Generated {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Step 5: Evaluate chunks
    # ------------------------------------------------------------------
    print("\n Step 5: Evaluating chunk quality...")
    eval_stats = evaluator.evaluate_chunks(chunks)

    evaluated_chunks = eval_stats["evaluated_chunks"]
    accepted_chunks = [c for c in evaluated_chunks if c["evaluation"]["should_use"]]

    print(
        f" Accepted: {len(accepted_chunks)}/{len(evaluated_chunks)} "
        f"({eval_stats['acceptance_rate']:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Step 6: Save file metadata (DB)
    # ------------------------------------------------------------------
    print("\n Step 6: Saving file metadata to DB...")
    extended_metadata = metadata.copy()

    db_manager.save_file_metadata(
        file_id=file_id,
        file_path=FILE_PATH,
        simhash=duplicate_check["simhash"],
        metadata=extended_metadata,
        content_length=len(markdown),
    )

    # ------------------------------------------------------------------
    # Step 7: Save chunks (DB)
    # ------------------------------------------------------------------
    print("\n Step 7: Saving chunks to DB...")
    # Note: DBManager extracts chunk_id directly from the chunk dict,
    # so we don't need to pass a separate list of IDs.
    db_manager.save_chunks(
        file_id=file_id,
        chunks=evaluated_chunks,
    )

    # ------------------------------------------------------------------
    # Step 8: Update SimHash index
    # ------------------------------------------------------------------
    simhash_handler.add_to_index(file_id, duplicate_check["simhash"])

    print("\nProcessing complete!")
    print(f"Data saved to database: {DB_PATH}")


def batch_process():
    """Process multiple files in batch."""
    print(" BATCH PROCESSING MODE")
    print("=" * 80)

    file_paths = []
    while True:
        path = input("Enter file path (or 'done' to finish): ")
        if path.lower() == "done":
            break
        file_paths.append(path)

    if not file_paths:
        print("No files provided.")
        return

    print(f"\nInitializing components for {len(file_paths)} files...")
    converter = PDFDocumentConverter()
    chunk_config = ChunkConfig()
    chunker = MarkdownChunker(chunk_config)
    evaluator = ChunkEvaluator()
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)

    # ✅ DB INIT
    DB_PATH = "rag_staging.db"
    db_manager = DBManager(db_path=DB_PATH)

    # Load simhashes
    existing_simhashes = load_simhashes_from_db(DB_PATH)
    if existing_simhashes:
        simhash_handler.load_index_from_data(existing_simhashes)

    results = []
    successful = skipped = failed = 0

    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing file {idx}/{len(file_paths)}: {file_path}")
        print("=" * 80)

        try:
            metadata, markdown = converter.process_document(file_path)

            duplicate_check = simhash_handler.check_duplicate(markdown)
            if duplicate_check["is_duplicate"]:
                print("Duplicate detected! Skipping...")
                skipped += 1
                results.append(
                    {"file": file_path, "status": "skipped", "reason": "duplicate"}
                )
                continue

            file_id = id_generator.generate_file_id()

            # Chunk
            chunks = chunker.process(markdown)

            # Evaluate
            eval_stats = evaluator.evaluate_chunks(chunks)
            evaluated_chunks = eval_stats["evaluated_chunks"]

            # Save Metadata
            extended_metadata = metadata.copy()
            db_manager.save_file_metadata(
                file_id=file_id,
                file_path=file_path,
                simhash=duplicate_check["simhash"],
                metadata=extended_metadata,
                content_length=len(markdown),
            )

            # Save Chunks
            db_manager.save_chunks(file_id, evaluated_chunks)

            # Update SimHash
            simhash_handler.add_to_index(file_id, duplicate_check["simhash"])

            print(f"Success! File ID: {file_id}, Chunks: {len(evaluated_chunks)}")
            successful += 1
            results.append(
                {
                    "file": file_path,
                    "status": "success",
                    "file_id": file_id,
                    "chunks": len(evaluated_chunks),
                }
            )

        except Exception as e:
            print(f"Error: {str(e)}")
            failed += 1
            results.append({"file": file_path, "status": "failed", "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print(" BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(file_paths)}")
    print(f"Successful: {successful}")
    print(f"Skipped (duplicates): {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING ORCHESTRATOR")
    print("=" * 80)

    mode = input(
        "\nSelect mode:\n 1. Single file\n 2. Batch processing\n\nChoice (1 or 2): "
    )

    if mode == "1":
        main()
    elif mode == "2":
        batch_process()
    else:
        print("Invalid choice. Exiting.")
