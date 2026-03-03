"""
main.py — QuestionPrep FastAPI Backend (Orchestrator)
Handles API routes, AI background tasks, and HTML template rendering.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---------------- INTERNAL IMPORTS ----------------
# Native imports powered by uv workspaces
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

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

# ---------------- PATH SETUP ----------------
# BASE_DIR is apps/orchestrator/
BASE_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT is two levels up from orchestrator
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rag_staging.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- APP & TEMPLATES ----------------
app = FastAPI(title="QuestionPrep API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ---------------- STATE & SINGLETONS ----------------
db = DBManager(db_path=str(DB_PATH))
active_sessions: Dict[str, InterviewSession] = {}


# ---------------- PYDANTIC MODELS ----------------
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class EvaluateRequest(BaseModel):
    session_id: str
    user_answer: str


class GenerateRequest(BaseModel):
    file_id: str


class StartInterviewRequest(BaseModel):
    user_id: str


# ---------------- HELPERS / BACKGROUND TASKS ----------------
def run_ingestion_task(temp_path: Path, user_id: str):
    """Heavy CPU/IO task moved to a background thread."""
    try:
        converter = PDFDocumentConverter()
        chunker = MarkdownChunker(ChunkConfig(max_chunk_tokens=1000))
        evaluator = ChunkEvaluator()
        id_gen = IDGenerator()
        simhash = SimHashHandler()

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

        if success:
            db.assign_file_to_user(user_id, file_id)
            logger.info(f"Successfully ingested file {file_id} for user {user_id}")

        if temp_path.exists():
            temp_path.unlink()

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")


# ==========================================
# PAGE ROUTES (HTML)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})


@app.get("/profile", response_class=HTMLResponse)
async def get_profile_page(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request})


@app.get("/interview", response_class=HTMLResponse)
async def get_interview_page(request: Request):
    return templates.TemplateResponse("interview.html", {"request": request})


# ==========================================
# AUTH API ROUTES
# ==========================================
@app.post("/api/auth/signup")
async def signup(user: SignupRequest):
    user_id = db.create_user(username=user.username, email=user.email)
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    return {"message": "User created successfully", "user_id": user_id}


@app.post("/api/auth/login")
async def login(user: LoginRequest):
    db_user = db.get_user_by_username(user.username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    # TODO: Implement real JWT/Bcrypt logic
    return {"token": "mock-jwt-token", "user": db_user}


@app.get("/api/auth/profile")
async def get_profile(user_id: str):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"profile": user}


# ==========================================
# PIPELINE API ROUTES
# ==========================================
@app.post("/api/files/ingest")
async def ingest_file(
    user_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if not db.get_user_by_id(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    temp_path = DATA_DIR / f"{uuid.uuid4()}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    background_tasks.add_task(run_ingestion_task, temp_path, user_id)

    return {
        "status": "processing",
        "message": "File upload successful. Ingestion running in background.",
        "filename": file.filename,
    }


@app.post("/api/questions/generate")
async def generate_questions(req: GenerateRequest, background_tasks: BackgroundTasks):
    def run_enrichment():
        enricher = EnrichmentManager()
        enricher.enrich_single_file(req.file_id)
        indexer = VectorIndexer()
        indexer.index_file(req.file_id)

    background_tasks.add_task(run_enrichment)
    return {"message": f"Generation and indexing started for {req.file_id}"}


@app.get("/api/files")
async def list_user_files(user_id: str):
    return {"files": db.get_files_for_user(user_id)}


# ==========================================
# INTERVIEW API ROUTES
# ==========================================
@app.post("/api/interview/start")
async def start_interview(req: StartInterviewRequest):
    if not db.get_user_by_id(req.user_id):
        raise HTTPException(status_code=404, detail="User not found")

    session_id = str(uuid.uuid4())
    session = InterviewSession(
        session_id=session_id,
        user_id=req.user_id,
        db_path=str(DB_PATH),
    )

    first_question = session.start_interview()
    if not first_question:
        raise HTTPException(
            status_code=404, detail="No questions available. Enriched files required."
        )

    active_sessions[session_id] = session
    return {"session_id": session_id, "first_question": first_question}


@app.post("/api/interview/evaluate")
async def evaluate_answer(req: EvaluateRequest):
    session = active_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.evaluate_turn(req.user_answer)


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
    if session_id in active_sessions:
        del active_sessions[session_id]
    return {"message": "Session ended"}


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import uvicorn

    # Start from project root: uv run uvicorn apps.orchestrator.main:app --reload
    uvicorn.run("apps.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)
