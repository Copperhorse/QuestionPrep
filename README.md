# QuestionPrep

A comprehensive question preparation and interview coaching platform powered by AI and voice recognition. QuestionPrep leverages advanced natural language processing, vector embeddings, and speech-to-text technologies to help users prepare for interviews with real-time feedback and stress detection.

## Overview

QuestionPrep is an AI-driven interview preparation system that combines several key technologies:

- **Speech Recognition**: Converts user speech to text using advanced ASR models
- **Question Generation**: Generates interview questions with intelligent chunking and enrichment
- **Vector Search**: RAG (Retrieval-Augmented Generation) capabilities using ChromaDB
- **Stress Detection**: Real-time stress level monitoring during interview simulations
- **Web Interface**: Interactive web application for conducting mock interviews
- **Machine Learning Models**: Multiple ML models (Logistic Regression, Random Forest, XGBoost) for analysis

## Tech Stack

### Languages
- **Python** (21.8%) - Core backend logic and ML pipelines
- **Jupyter Notebook** (63.8%) - Data analysis and exploration
- **JavaScript** (6%) - Frontend interactivity and stress detection
- **HTML** (5.1%) - Web templates
- **CSS** (2.5%) - Styling
- **Shell** (0.8%) - Scripting and automation

### Key Technologies
- **FastAPI** - Web framework for the orchestrator service
- **Uvicorn** - ASGI server
- **Whisper/Qwen3-ASR** - Speech-to-text models
- **ChromaDB** - Vector database for semantic search
- **ONNX** - Machine learning model format
- **Docling** - Document processing and OCR
- **Pydantic** - Data validation

## Project Structure

```
QuestionPrep/
├── apps/                          # Web applications
│   └── orchestrator/              # Main FastAPI application
│       ├── main.py               # FastAPI server entry point
│       ├── rate_limiting.py       # Rate limiting logic
│       ├── static/                # Static assets
│       │   ├── css/               # Stylesheets
│       │   ├── js/                # JavaScript (app, interview, stress-detector)
│       │   └── models/            # ONNX ML models
│       └── templates/             # HTML templates (auth, interview, profile)
│
├── packages/                      # Reusable packages
│   ├── qp-core/                   # Core utilities
│   │   └── src/qp_core/
│   │       ├── DBManager.py       # Database management
│   │       ├── VectorStore.py     # Vector DB operations
│   │       ├── SimHashHandler.py  # Similarity hashing
│   │       ├── IDGenerator.py     # Unique ID generation
│   │       └── CSVManager.py      # CSV operations
│   │
│   ├── qp-pipeline/               # Data processing pipeline
│   │   └── src/qp_pipeline/
│   │       ├── ingester.py        # Data ingestion
│   │       ├── MarkdownChunker.py # Document chunking
│   │       ├── Embedder.py        # Text embeddings
│   │       ├── Enricher.py        # Data enrichment
│   │       ├── ChunkEvaluator.py  # Quality evaluation
│   │       ├── docling_ocr.py     # OCR processing
│   │       └── game_loop.py       # Interview simulation loop
│   │
│   └── qp-voice/                  # Voice processing
│       └── src/qp_voice/
│           ├── speech_to_text.py  # ASR functionality
│           └── text_to_speech.py  # TTS functionality
│
├── qp_notebooks/                  # Interactive notebooks and analysis
│   └── src/
│       ├── test.py                # Testing notebooks
│       └── __marimo__/            # Marimo notebook sessions
│
├── data/                          # Data storage
│   ├── chroma_store/              # Vector database
│   ├── rag_staging.db             # Staging database
│   └── server_log.txt             # Server logs
│
├── scripts/                       # Automation scripts
│   └── enrichment.sh              # Data enrichment automation
│
├── Qwen3-ASR-0.6B/                # ASR model files
│
├── pyproject.toml                 # Main project configuration
├── uv.lock                        # Dependency lock file
└── test.db                        # Test database

```

## Key Components

### 1. **Orchestrator Service** (`apps/orchestrator/`)
The main FastAPI application that serves the web interface and coordinates between different modules:
- Interview simulation and management
- Authentication (JWT tokens, bcrypt hashing)
- Rate limiting for API endpoints
- Real-time stress detection via WebSockets
- ML model inference on the frontend (ONNX models)

### 2. **QP Core** (`packages/qp-core/`)
Core utilities for system operations:
- **DBManager**: SQL database operations and migrations
- **VectorStore**: ChromaDB operations for semantic search
- **SimHashHandler**: Duplicate detection via similarity hashing
- **IDGenerator**: Unique identifier generation
- **CSVManager**: CSV file handling

### 3. **QP Pipeline** (`packages/qp-pipeline/`)
Data processing and question generation pipeline:
- **Ingester**: Load documents from various sources
- **MarkdownChunker**: Intelligent document chunking
- **Embedder**: Generate vector embeddings
- **Enricher**: Enhance questions with metadata
- **ChunkEvaluator**: Evaluate chunk quality
- **docling_ocr**: OCR and document understanding

### 4. **QP Voice** (`packages/qp-voice/`)
Voice processing capabilities:
- **Speech-to-Text**: Convert audio to text using Whisper/Qwen3-ASR
- **Text-to-Speech**: Convert text responses to audio

### 5. **Interactive Notebooks** (`qp_notebooks/`)
Marimo-powered interactive analysis and exploration notebooks for data analysis and model testing.

## Features

✅ **AI-Powered Interview Questions** - Automatically generates relevant interview questions from documents  
✅ **Real-time Speech Recognition** - Converts spoken answers to text in real-time  
✅ **Stress Detection** - Analyzes stress levels during mock interviews  
✅ **Vector Search (RAG)** - Semantic search for relevant questions and feedback  
✅ **Persistent Storage** - SQLite database for user profiles and session history  
✅ **Authentication** - Secure user authentication with JWT tokens  
✅ **Interactive Web UI** - User-friendly interface for mock interviews  
✅ **ML Model Integration** - Multiple classification models for analysis  

## Getting Started

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/Copperhorse/QuestionPrep.git
cd QuestionPrep

# Install dependencies using uv
uv sync

# Install workspace packages
uv run pip install -e packages/qp-core packages/qp-pipeline packages/qp-voice
```

### Running the Application

```bash
# Start the FastAPI server
uv run uvicorn apps.orchestrator.main:app --reload

# Access the application
# Open http://localhost:8000 in your browser
```

### Running Notebooks

```bash
# Install and run Marimo notebooks
uv run marimo run qp_notebooks/src/test.py
```

## Project Dependencies

### Core Dependencies
- `datasets` - Dataset handling
- `docling` - Document processing and OCR
- `faster-whisper` - Fast speech-to-text
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `python-jose` - JWT authentication
- `passlib[bcrypt]` - Password hashing
- `slowapi` - Rate limiting

### Workspace Packages
- `qp-core` - Core utilities
- `qp-pipeline` - Data pipeline
- `qp-voice` - Voice processing
- `qp-notebooks` - Interactive notebooks

## Configuration

All configuration is managed through `pyproject.toml`. The project uses UV's workspace feature to manage multiple packages:

```toml
[tool.uv.workspace]
members = ["packages/*", "apps/*", "qp_notebooks"]
```

## Database Schema

The system uses SQLite for persistent storage (`test.db`) with support for:
- User profiles and authentication
- Interview session history
- Question bank
- Session logs and analytics

## API Endpoints

The Orchestrator service provides REST API endpoints for:
- User authentication
- Mock interview management
- Question retrieval
- Session analytics
- Stress level monitoring

## Machine Learning Models

QuestionPrep includes pre-trained models for classification:
- **Logistic Regression** (logistic_regression.onnx)
- **Random Forest** (random_forest.onnx)
- **XGBoost** (xgboost.onnx)

These models run directly in the browser using ONNX.js.

## Performance Features

- **Rate Limiting**: Implemented via `slowapi` to prevent API abuse
- **Caching**: Vector embeddings cached in ChromaDB
- **Async Processing**: FastAPI async support for concurrent requests
- **WASM Models**: ML models compiled to WebAssembly for browser execution

## Development

### Running Tests
```bash
# Execute test notebooks and modules
uv run pytest qp_notebooks/src/test.py
```

### Project Structure
This project uses a monorepo structure with:
- `apps/` - Deployable applications
- `packages/` - Shared libraries
- `qp_notebooks/` - Data science and analysis notebooks

## Roadmap

- Enhanced stress detection algorithms
- Multi-language support
- Advanced analytics dashboard
- Mobile app integration
- Custom question banking

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source and available on GitHub.

## Support

For issues, questions, or suggestions, please visit the [GitHub Issues](https://github.com/Copperhorse/QuestionPrep/issues) page.

---

**Last Updated:** May 2026  
**Repository:** [Copperhorse/QuestionPrep](https://github.com/Copperhorse/QuestionPrep)
