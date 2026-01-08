# QuestionPrep

## Project Information

**Academic Project Developers:**

- Muhammad Abdullah (COSC221102041)
- Muhammad Ahmad Siddiqui (COSC221102034)

**Supervisor:** Sir Adeel Abid  
**Class:** BS Data Science (2022)

---

## Project Status & Updates

### Completed Tasks

- **PDF OCR and Markdown Conversion:** Utilizes the Docling library for high-fidelity PDF-to-Markdown conversion. Includes formula enrichment to ensure mathematical equations are captured as LaTeX rather than flattened text.

- **Duplicate Detection (SimHash):** Integrated a 64-bit SimHash algorithm with a Hamming distance threshold (k=3) to identify and prevent the ingestion of near-identical documents.

- **Structural Markdown Chunking:** Developed a custom `MarkdownChunker` that respects document hierarchy and protects specialized content like tables, code blocks, and LaTeX math from being split across chunks.

- **Automated Quality Evaluation:** Built the `ChunkEvaluator` to score content quality. It calculates an Alpha Ratio to filter out OCR noise and uses exclusion lists to remove junk like Table of Contents or Bibliographies.

- **Persistent Data Management:** Implemented the `CSVManager` to handle structured storage. It maps file metadata to individual chunk IDs using 128-bit UUIDs for traceability.

### Pending Tasks

- **Vector Database Integration:** Exporting processed `chunks.csv` data into a vector store (e.g., ChromaDB or FAISS) for semantic retrieval.

- **AI Simulation Engine:** Developing the prompt engineering layer to generate interview questions based on the "Accepted" high-quality chunks.

- **Streamlit Dashboard UI:** Migrating the current CLI-based orchestration into a user-friendly web interface.

---

## 🛠️ File-by-File Documentation

### 1. Orchestrator.py (The Pipeline Brain)

The central coordinator for the entire system that manages the workflow by:

- **User Interface:** Providing a CLI for single file or batch processing modes.
- **Sequence Management:** Ordering the conversion, deduplication, chunking, and evaluation steps.
- **Data Integrity:** Checking for duplicates via SimHash before committing to the chunking process.

### 2. MarkdownChunker.py (Smart Context Splitter)

A specialized text-splitting engine that preserves the structural meaning of the document:

- **Block Detection:** Uses regex to identify and "protect" code blocks, math environments, and tables.
- **Hierarchy Awareness:** Can split based on Markdown headers (`#` to `####`) or numeric sectioning (e.g., 1.1.2).
- **Token Counting:** Integrates `tiktoken` (if available) to ensure chunks fit within LLM context windows.

### 3. ChunkEvaluator.py (The Quality Filter)

Acts as a gatekeeper to ensure only high-value information reaches the database:

- **Heuristic Scoring:** Scores chunks on a 0-100 scale based on length, completeness, and alphabetic density.
- **Content Filtering:** Automatically flags and rejects "metadata" chunks like copyright notices, DOIs, or image placeholders.
- **Transparency:** Marks chunks as `should_use` (True/False) while retaining the "reason" for rejection in the CSV logs.

### 4. SimHashHandler.py (Deduplication Engine)

Handles "fuzzy" duplicate detection:

- **SimHash Generation:** Creates a 64-bit fingerprint of the document using character n-gram features.
- **Indexing:** Maintains a `SimhashIndex` to allow for rapid comparison against previously processed files.
- **Hamming Distance:** Calculates the bit-difference between hashes to identify documents that are nearly identical.

### 5. docling_ocr.py (High-Fidelity Extraction)

Handles the raw conversion of PDF binaries:

- **OCR Engine:** Wraps the Docling converter for structural extraction.
- **Metadata Extraction:** Uses PyMuPDF (as `pymupdf`) to pull technical details like author, creator, and encryption status from the PDF header.

### 6. CSVManager.py & IDGenerator.py (Infrastructure)

- **CSVManager:** Handles the appending of data to `files.csv` and `chunks.csv`. Ensures headers are created if the files don't exist and manages the directory structure.
- **IDGenerator:** Provides unique UUIDs to link chunks back to their parent files, ensuring a relational structure within flat CSV files.

---
