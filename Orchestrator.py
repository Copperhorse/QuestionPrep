"""
Orchestrator.py
Complete document processing pipeline with CSV export
"""

import re  # For ToC extraction regex

from Chunker.MarkdownChunker import ChunkConfig, MarkdownChunker
from Chunker.toc_parser import ToCParser  # Your original ToCParser
from Evaluator.ChunkEvaluator import ChunkEvaluator
from Extractor.docling_ocr import PDFDocumentConverter
from Utils.CSVManager import CSVManager
from Utils.IDGenerator import IDGenerator
from Utils.SimHashHandler import SimHashHandler


def main():
    """Main orchestration function."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    print("\n Initializing components...")
    converter = PDFDocumentConverter()
    chunk_config = ChunkConfig(max_chunk_tokens=500)
    chunker = MarkdownChunker(chunk_config)
    toc_parser = ToCParser()
    evaluator = ChunkEvaluator()
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)
    csv_manager = CSVManager(output_dir="output")

    FILE_PATH = input("Enter the file path: ")

    # ------------------------------------------------------------------
    # Load existing simhashes
    # ------------------------------------------------------------------
    print(" Loading existing simhashes...")
    existing_simhashes = csv_manager.load_existing_simhashes()
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
    # Step 2: Duplicate detection (before chunking)
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
    # Step 4: ToC extraction + parsing (document-level)
    # ------------------------------------------------------------------
    print("\n Step 4: Parsing Table of Contents (if present)...")

    toc_export = None
    cleaned_markdown = markdown

    toc_match = re.search(
        r"(##\s*Table of Contents|TABLE OF CONTENTS)[\s\S]*?"
        r"(?=\n##\s*(?!Table of Contents)|\Z)",
        markdown,
        flags=re.IGNORECASE,
    )

    if toc_match:
        toc_text = toc_match.group(0)
        toc_parser.parse(toc_text)
        toc_export = toc_parser.export_for_chunker()

        cleaned_markdown = markdown.replace(toc_text, "", 1).strip()

        print(
            f" ToC parsed: {toc_export['metadata']['total_entries']} entries | "
            f"max depth {toc_export['metadata']['max_depth']}"
        )
    else:
        print(" No ToC detected.")

    # ------------------------------------------------------------------
    # Step 5: Chunking (structure-agnostic)
    # ------------------------------------------------------------------
    print("\n Step 5: Chunking document...")
    chunks = chunker.process(cleaned_markdown)
    print(f" Generated {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Step 6: Enrich chunks with ToC metadata (if available)
    # ------------------------------------------------------------------
    # if toc_export:
    #     print(" Enriching chunks with ToC hierarchy...")
    #     section_map = toc_export.get("section_map", {})

    #     for chunk in chunks:
    #         header = (chunk["metadata"].get("section_header") or "").strip()
    #         if not header:
    #             continue

    #         matched = section_map.get(header)

    #         # Fallback: case-insensitive partial match
    #         if not matched:
    #             for key, value in section_map.items():
    #                 if header.lower() in key.lower():
    #                     matched = value
    #                     break

    #         if matched:
    #             chunk["metadata"].update(
    #                 {
    #                     "toc_full_path": matched.get("full_path"),
    #                     "toc_level": matched.get("level"),
    #                     "toc_numeric_id": matched.get("numeric_id"),
    #                     "toc_parent_id": matched.get("parent_id"),
    #                 }
    #             )

    # ------------------------------------------------------------------
    # Step 7: Evaluate chunks
    # ------------------------------------------------------------------
    print("\n Step 6: Evaluating chunk quality...")
    eval_stats = evaluator.evaluate_chunks(chunks)

    evaluated_chunks = eval_stats["evaluated_chunks"]
    accepted_chunks = [c for c in evaluated_chunks if c["evaluation"]["should_use"]]
    rejected_chunks = [c for c in evaluated_chunks if not c["evaluation"]["should_use"]]

    print(
        f" Accepted: {len(accepted_chunks)}/{len(evaluated_chunks)} "
        f"({eval_stats['acceptance_rate']:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Step 8: Generate chunk IDs
    # ------------------------------------------------------------------
    print("\n Step 7: Generating chunk IDs...")
    chunk_ids = id_generator.generate_chunk_ids(len(evaluated_chunks))

    # ------------------------------------------------------------------
    # Step 9: Save file metadata
    # ------------------------------------------------------------------
    print("\n Step 8: Saving file metadata...")
    extended_metadata = metadata.copy()
    if toc_export:
        extended_metadata["toc_metadata"] = toc_export["metadata"]

    csv_manager.save_file_metadata(
        file_id=file_id,
        file_path=FILE_PATH,
        simhash=duplicate_check["simhash"],
        metadata=extended_metadata,
        content_length=len(markdown),
    )

    # ------------------------------------------------------------------
    # Step 10: Save chunks
    # ------------------------------------------------------------------
    print("\n Step 9: Saving chunks...")
    csv_manager.save_chunks(
        file_id=file_id,
        chunk_ids=chunk_ids,
        chunks=evaluated_chunks,
    )

    # ------------------------------------------------------------------
    # Step 11: Update SimHash index
    # ------------------------------------------------------------------
    simhash_handler.add_to_index(file_id, duplicate_check["simhash"])

    print("\nProcessing complete!")
    print(f"Output saved to: {csv_manager.output_dir.absolute()}")


def batch_process():
    """Process multiple files in batch."""
    print(" BATCH PROCESSING MODE")
    print("=" * 80)
    # Get file paths
    file_paths = []
    while True:
        path = input("Enter file path (or 'done' to finish): ")
        if path.lower() == "done":
            break
        file_paths.append(path)
    if not file_paths:
        print("No files provided.")
        return
    # Initialize components
    print(f"\nInitializing components for {len(file_paths)} files...")
    converter = PDFDocumentConverter()
    chunk_config = ChunkConfig()
    chunker = MarkdownChunker(chunk_config)
    toc_parser = ToCParser()  # New: For ToC handling
    evaluator = ChunkEvaluator()
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)
    csv_manager = CSVManager(output_dir="output")
    # Load existing simhashes
    existing_simhashes = csv_manager.load_existing_simhashes()
    if existing_simhashes:
        simhash_handler.load_index_from_data(existing_simhashes)
    # Process each file
    results = []
    successful = 0
    skipped = 0
    failed = 0
    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing file {idx}/{len(file_paths)}: {file_path}")
        print("=" * 80)
        try:
            # Convert
            metadata, markdown = converter.process_document(file_path)
            # Check duplicates
            duplicate_check = simhash_handler.check_duplicate(markdown)
            if duplicate_check["is_duplicate"]:
                print("Duplicate detected! Skipping...")
                skipped += 1
                results.append(
                    {"file": file_path, "status": "skipped", "reason": "duplicate"}
                )
                continue
            # Generate IDs
            file_id = id_generator.generate_file_id()
            # Extract/parse/remove ToC
            toc_match = re.search(
                r"(##\s*Table of Contents|TABLE OF CONTENTS)[\s\S]*?(?=##\s*(?!Table of Contents)|$)",
                markdown,
                re.IGNORECASE | re.MULTILINE,
            )
            toc_export = None
            cleaned_markdown = markdown
            if toc_match:
                toc_text = toc_match.group(0)
                toc_result = toc_parser.parse(toc_text)
                toc_export = toc_parser.export_for_chunker()
                cleaned_markdown = markdown.replace(toc_text, "", 1).strip()
            # Chunk
            chunks = chunker.process(cleaned_markdown)
            # Enrich with ToC
            section_map = toc_export.get("section_map", {}) if toc_export else {}
            if section_map:
                for chunk in chunks:
                    header = chunk["metadata"].get("section_header") or ""
                    matched_meta = section_map.get(header)
                    if not matched_meta:
                        for key in section_map:
                            if header.lower() in key.lower():
                                matched_meta = section_map[key]
                                break
                    if matched_meta:
                        chunk["metadata"].update(
                            {
                                "toc_full_path": matched_meta.get("full_path"),
                                "toc_level": matched_meta.get("level"),
                                "toc_numeric_id": matched_meta.get("numeric_id"),
                                "toc_parent_id": matched_meta.get("parent_id"),
                            }
                        )
            # Evaluate
            eval_stats = evaluator.evaluate_chunks(chunks)
            evaluated_chunks = eval_stats["evaluated_chunks"]
            # Save
            extended_metadata = metadata.copy()
            if toc_export:
                extended_metadata["toc_metadata"] = toc_export["metadata"]
            csv_manager.save_file_metadata(
                file_id=file_id,
                file_path=file_path,
                simhash=duplicate_check["simhash"],
                metadata=extended_metadata,
                content_length=len(markdown),
            )
            chunk_ids = id_generator.generate_chunk_ids(len(evaluated_chunks))
            csv_manager.save_chunks(file_id, chunk_ids, evaluated_chunks)
            # Update index
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
    print("\n Detailed Results:")
    for result in results:
        print(f"\n{result['file']}:")
        print(f" Status: {result['status']}")
        if result["status"] == "success":
            print(f" File ID: {result['file_id']}")
            print(f" Chunks: {result['chunks']}")
        elif result["status"] == "failed":
            print(f" Error: {result['error']}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING ORCHESTRATOR")
    print("=" * 80)
    # Choose mode
    mode = input(
        "\nSelect mode:\n 1. Single file\n 2. Batch processing\n\nChoice (1 or 2): "
    )
    if mode == "1":
        main()
    elif mode == "2":
        batch_process()
    else:
        print("Invalid choice. Exiting.")
