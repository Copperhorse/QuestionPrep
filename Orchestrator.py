"""
Orchestrator.py
Complete document processing pipeline with CSV export
"""

from IPython.display import display_markdown

from Chunker.MarkdownChunker import ChunkConfig, MarkdownChunker
from Extractor.docling_ocr import PDFDocumentConverter
from Utils.CSVManager import CSVManager
from Utils.IDGenerator import IDGenerator
from Utils.SimHashHandler import SimHashHandler


def main():
    """Main orchestration function."""

    # Get file path from user
    FILE_PATH = input("Enter the file path: ")

    # Initialize all components
    print("\n Initializing components...")
    converter = PDFDocumentConverter()
    config = ChunkConfig()
    chunker = MarkdownChunker(config)
    id_generator = IDGenerator()
    simhash_handler = SimHashHandler(k=3)
    csv_manager = CSVManager(output_dir="output")

    # Load existing simhashes for duplicate detection
    print("Loading existing simhashes...")
    existing_simhashes = csv_manager.load_existing_simhashes()
    if existing_simhashes:
        simhash_handler.load_index_from_data(existing_simhashes)
        print("Loaded {len(existing_simhashes)} existing files into index")

    # Step 1: Convert PDF to markdown
    print("\nStep 1: Converting PDF to markdown...")
    metadata, markdown = converter.process_document(FILE_PATH)
    print(f"Converted successfully ({len(markdown)} characters)")

    # Step 2: Check for duplicates (BEFORE chunking)
    print("\nStep 2: Checking for duplicates...")
    duplicate_check = simhash_handler.check_duplicate(markdown)

    if duplicate_check["is_duplicate"]:
        print("WARNING: This document appears to be a duplicate!")
        print(f"   Similar to file IDs: {duplicate_check['duplicate_file_ids']}")

        proceed = input("\n   Continue processing anyway? (y/n): ")
        if proceed.lower() != "y":
            print("\n Processing cancelled.")
            return
    else:
        print(f"No duplicates found (SimHash: {duplicate_check['simhash']})")

    # Step 3: Generate file ID
    print("\n Step 3: Generating file ID...")
    file_id = id_generator.generate_file_id()
    print(f"   File ID: {file_id}")

    # Step 4: Chunk the markdown
    print("\n Step 4: Chunking markdown...")
    chunks = chunker.process(markdown)
    print(f"Generated {len(chunks)} chunks")

    # Step 5: Generate chunk IDs
    print("\nStep 5: Generating chunk IDs...")
    chunk_ids = id_generator.generate_chunk_ids(len(chunks))
    print(f"Generated {len(chunk_ids)} chunk IDs")

    # Step 6: Save file metadata to CSV
    print("\n Step 6: Saving file metadata...")
    file_saved = csv_manager.save_file_metadata(
        file_id=file_id,
        file_path=FILE_PATH,
        simhash=duplicate_check["simhash"],
        metadata=metadata,
        content_length=len(markdown),
    )

    if file_saved:
        print("File metadata saved to files.csv")
    else:
        print("Failed to save file metadata")
        return

    # Step 7: Save chunks to CSV
    print("\nStep 7: Saving chunks...")
    chunks_saved = csv_manager.save_chunks(
        file_id=file_id, chunk_ids=chunk_ids, chunks=chunks
    )

    if chunks_saved:
        print("Chunks saved to chunks.csv")
    else:
        print("Failed to save chunks")
        return

    # Step 8: Add simhash to index for future lookups
    print("\nStep 8: Updating simhash index...")
    simhash_handler.add_to_index(file_id, duplicate_check["simhash"])
    print("SimHash added to index")

    # Display results
    print("\n" + "=" * 80)
    print("📊 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"File ID: {file_id}")
    print(f"File Name: {FILE_PATH.split('/')[-1]}")
    print(f"SimHash: {duplicate_check['simhash']}")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Is Duplicate: {duplicate_check['is_duplicate']}")

    # Display chunk details
    print("\n" + "=" * 80)
    print("CHUNK DETAILS")
    print("=" * 80)
    print(f"Generated {len(chunks)} chunks:\n")

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk_ids[idx]
        print(f"Chunk {chunk['chunk_index']}:")
        print(f"  Chunk ID: {chunk_id}")
        print(f"  Tokens: {chunk['estimated_tokens']}")
        print(f"  Header: {chunk['metadata'].get('section_header', 'N/A')}")
        print(f"  Level: {chunk['metadata'].get('section_level', 'N/A')}")
        print(f"  Type: {chunk['metadata'].get('content_type', 'N/A')}")
        print(f"  Content preview: {chunk['content'][:100]}...")
        print()

    # Display file metadata
    print("=" * 80)
    print("📋 FILE METADATA")
    print("=" * 80)
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Display statistics
    print("\n" + "=" * 80)
    print("📈 STATISTICS")
    print("=" * 80)
    total_files = csv_manager.get_file_count()
    total_chunks = csv_manager.get_chunk_count()
    print(f"Total files in database: {total_files}")
    print(f"Total chunks in database: {total_chunks}")

    print("\nProcessing complete!")
    print(f"Output saved to: {csv_manager.output_dir.absolute()}")


def batch_process():
    """Process multiple files in batch."""

    print("📦 BATCH PROCESSING MODE")
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
    print(f"\n🔧 Initializing components for {len(file_paths)} files...")
    converter = PDFDocumentConverter()
    config = ChunkConfig()
    chunker = MarkdownChunker(config)
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

            # Chunk
            chunks = chunker.process(markdown)
            chunk_ids = id_generator.generate_chunk_ids(len(chunks))

            # Save
            csv_manager.save_file_metadata(
                file_id=file_id,
                file_path=file_path,
                simhash=duplicate_check["simhash"],
                metadata=metadata,
                content_length=len(markdown),
            )

            csv_manager.save_chunks(file_id, chunk_ids, chunks)

            # Update index
            simhash_handler.add_to_index(file_id, duplicate_check["simhash"])

            print(f"Success! File ID: {file_id}, Chunks: {len(chunks)}")
            successful += 1
            results.append(
                {
                    "file": file_path,
                    "status": "success",
                    "file_id": file_id,
                    "chunks": len(chunks),
                }
            )

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed += 1
            results.append({"file": file_path, "status": "failed", "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(file_paths)}")
    print(f"Successful: {successful}")
    print(f"Skipped (duplicates): {skipped}")
    print(f"Failed: {failed}")

    print("\nDetailed Results:")
    for result in results:
        print(f"\n{result['file']}:")
        print(f"  Status: {result['status']}")
        if result["status"] == "success":
            print(f"  File ID: {result['file_id']}")
            print(f"  Chunks: {result['chunks']}")
        elif result["status"] == "failed":
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING ORCHESTRATOR")
    print("=" * 80)

    # Choose mode
    mode = input(
        "\nSelect mode:\n  1. Single file\n  2. Batch processing\n\nChoice (1 or 2): "
    )

    if mode == "1":
        main()
    elif mode == "2":
        batch_process()
    else:
        print("Invalid choice. Exiting.")
