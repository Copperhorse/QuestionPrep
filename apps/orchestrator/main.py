"""
main.py — QuestionPrep FastAPI Backend
"""

import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- PATH SETUP ----------------
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.append(str(project_root / "packages" / "qp-pipeline" / "src"))
sys.path.append(str(project_root / "packages" / "qp-core" / "src"))

# ---------------- INTERNAL IMPORTS ----------------
from qp_core.DBManager import DBManager
from qp_core.IDGenerator import IDGenerator
from qp_core.SimHashHandler import SimHashHandler
from qp_pipeline.ChunkEvaluator import ChunkEvaluator
from qp_pipeline.docling_ocr import PDFDocumentConverter
from qp_pipeline.Embedder import VectorIndexer
from qp_pipeline.Enricher import EnrichmentManager
from qp_pipeline.game_loop import InterviewSession
from qp_pipeline.ingester import ingest
from qp_pipeline.MarkdownChunker import ChunkConfig, MarkdownChunker

# ---------------- APP ----------------
app = FastAPI(title="QuestionPrep API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- SINGLETONS ----------------
# Only the DB and active sessions are held at the process level.
# EnrichmentManager and VectorIndexer load large models and are instantiated
# inside their route handlers so memory is only used when those routes are hit.
DB_PATH = str(project_root / "data" / "rag_staging.db")

db = DBManager(db_path=DB_PATH)

# In-memory session store. Sessions are per-user and created on demand.
# Replace with Redis for multi-process or persistent deployments.
active_sessions: Dict[str, InterviewSession] = {}


# ---------------- PYDANTIC MODELS ----------------
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str  # TODO: hash with passlib/bcrypt before storing


class LoginRequest(BaseModel):
    username: str
    password: str  # TODO: verify against hashed password


class EvaluateRequest(BaseModel):
    session_id: str
    user_answer: str


class GenerateRequest(BaseModel):
    file_id: str


class StartInterviewRequest(BaseModel):
    user_id: str


# ==========================================
# AUTH ROUTES
# ==========================================
@app.post("/api/auth/signup")
async def signup(user: SignupRequest):
    user_id = db.create_user(username=user.username, email=user.email)
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    return {"message": "User created successfully", "user_id": user_id}


@app.post("/api/auth/login")
async def login(user: LoginRequest):
    # TODO: replace mock token with real JWT (pyjwt) and verify hashed password
    db_user = db.get_user_by_username(user.username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"token": "mock-jwt-token-replace-me", "user": db_user}


@app.get("/api/auth/profile")
async def get_profile(user_id: str):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"profile": user}


# ==========================================
# PIPELINE ROUTES
# ==========================================
@app.post("/api/files/ingest")
async def ingest_file(user_id: str, file: UploadFile = File(...)):
    """
    Ingest a PDF for a specific user.

    1. Saves the upload to a temp path
    2. Runs conversion → chunking → evaluation → DB save
    3. Links the resulting file_id to the user in user_files

    Args (query param):
        user_id: UUID of the authenticated user uploading the file
    """
    # Verify user exists before doing any heavy work
    if not db.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    temp_path = project_root / "data" / file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    converter = PDFDocumentConverter()
    chunker = MarkdownChunker(ChunkConfig(max_chunk_tokens=1000))
    evaluator = ChunkEvaluator()
    id_gen = IDGenerator()
    simhash = SimHashHandler()

    # Load existing hashes so duplicate detection works correctly
    existing = db.get_all_simhashes()
    simhash.load_index_from_data(existing)

    success, file_id = ingest(
        temp_path,
        db,
        converter,
        chunker,
        evaluator,
        id_gen,
        simhash,
        auto_confirm=True,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to ingest file")

    # Link the file to the user so the game loop can scope questions correctly
    db.assign_file_to_user(user_id, file_id)

    return {
        "message": "File successfully ingested",
        "file_id": file_id,
        "user_id": user_id,
    }


@app.post("/api/questions/generate")
async def generate_questions(req: GenerateRequest):
    """
    Run the LLM enrichment pass then the vector indexing pass for a file.

    EnrichmentManager and VectorIndexer are instantiated here rather than at
    startup — they load large models (BGE embeddings, Chroma) that should only
    occupy memory when actually needed. This mirrors the ingester.py pattern.
    """
    try:
        enricher = EnrichmentManager()
        enricher.enrich_single_file(req.file_id)

        indexer = VectorIndexer()
        indexer.index_file(req.file_id)

        return {"message": f"Questions generated and indexed for file {req.file_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files")
async def list_user_files(user_id: str):
    """Return all files assigned to a user."""
    files = db.get_files_for_user(user_id)
    return {"files": files}


# ==========================================
# INTERVIEW ROUTES
# ==========================================
@app.post("/api/interview/start")
async def start_interview(req: StartInterviewRequest):
    """
    Create a new interview session for the given user.

    Each session owns its own LogicEngine which loads only that user's
    questions. The expensive grader model is shared across all sessions
    via the LogicEngine class-level singleton.
    """
    if not db.get_user_by_id(req.user_id):
        raise HTTPException(status_code=404, detail="User not found")

    session_id = str(uuid.uuid4())
    session = InterviewSession(
        session_id=session_id,
        user_id=req.user_id,
        db_path=DB_PATH,
    )
    first_question = session.start_interview()

    if not first_question:
        raise HTTPException(
            status_code=404,
            detail=(
                "No questions available for this user. "
                "Ensure at least one file has been ingested and enriched."
            ),
        )

    active_sessions[session_id] = session

    return {
        "session_id": session_id,
        "first_question": first_question,
    }


@app.post("/api/interview/evaluate")
async def evaluate_answer(req: EvaluateRequest):
    session = active_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = session.evaluate_turn(req.user_answer)
    return result


@app.get("/api/interview/{session_id}/summary")
async def get_summary(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session.ctx.history
    avg_score = sum(r.similarity for r in history) / len(history) if history else 0.0

    return {
        "questions_attempted": len(history),
        "average_similarity": round(avg_score, 3),
        "final_difficulty": session.ctx.difficulty_label,
        "detailed_history": [
            {
                "question_id": r.question_id,
                "similarity": r.similarity,
                "confidence": r.confidence,
                "feedback": r.feedback,
            }
            for r in history
        ],
    }


@app.delete("/api/interview/{session_id}")
async def end_session(session_id: str):
    """Explicitly remove a session from memory once the client is done."""
    if session_id in active_sessions:
        del active_sessions[session_id]
    return {"message": "Session ended"}


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
