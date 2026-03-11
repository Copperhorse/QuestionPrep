"""
main.py — QuestionPrep FastAPI Backend (Orchestrator)
Handles API routes, AI background tasks, and HTML template rendering.

The LLM server (llama-server) is managed here — no need to run enrichment.sh manually.
On the first "Generate Questions" click it starts the server, waits for it to load,
runs enrichment + indexing, then leaves the server running for subsequent requests.
On app shutdown the server is killed, mirroring enrichment.sh step 6.
"""

import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

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
from qp_voice.speech_to_text import SpeechToText, analyze_disfluencies

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rag_staging.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LLAMA SERVER CONFIG ----------------
# Mirrors enrichment.sh exactly — update these if paths change on the USB stick.
LLAMA_BIN = Path("/media/copper/USB_STICK/Git/llama.cpp/build/bin/llama-server")
LLAMA_MODEL = Path("/media/copper/USB_STICK/Models/LFM2.5-1.2B-Instruct-Q8_0.gguf")
LLAMA_HOST = "localhost"
LLAMA_PORT = 8080
LLAMA_HEALTH_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models"
LLAMA_LOG = DATA_DIR / "server_log.txt"

# enrichment.sh uses 60 × 3 s = 3 min. Match that.
HEALTH_RETRIES = 60
HEALTH_INTERVAL = 3  # seconds

# ---------------- APP & TEMPLATES ----------------
app = FastAPI(title="QuestionPrep API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# ---------------- GLOBAL STATE ----------------
db = DBManager(db_path=str(DB_PATH))
active_sessions: Dict[str, InterviewSession] = {}
_stt: Optional[SpeechToText] = None
_llm_process: Optional[subprocess.Popen] = None


def get_stt() -> SpeechToText:
    global _stt
    if _stt is None:
        logger.info("Loading Qwen3 ASR model...")
        _stt = SpeechToText()
        logger.info("Qwen3 ASR model loaded.")
    return _stt


# ==========================================
# LLAMA SERVER LIFECYCLE  (replaces enrichment.sh steps 1-4 and 6)
# ==========================================


def _llama_is_healthy() -> bool:
    """True if the server is up and the model has finished loading (HTTP 200)."""
    try:
        r = httpx.get(LLAMA_HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _kill_stale_server():
    """enrichment.sh step 1 — kill any leftover llama-server from a previous run."""
    result = subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    if result.returncode == 0:
        logger.info("Killed stale llama-server process.")
        time.sleep(2)


def _ensure_llama_running() -> bool:
    """
    Start llama-server and wait until the model is fully loaded,
    mirroring enrichment.sh steps 2-4.

    Returns True if the server is ready, False on any failure.
    Server is left running after this call (killed only on app shutdown).
    """
    global _llm_process

    # Already healthy — nothing to do.
    if _llama_is_healthy():
        logger.info("LLM server already running and healthy.")
        return True

    if not LLAMA_BIN.exists():
        logger.error(f"llama-server binary not found: {LLAMA_BIN}")
        return False
    if not LLAMA_MODEL.exists():
        logger.error(f"LLM model not found: {LLAMA_MODEL}")
        return False

    # Step 1 — clean up anything leftover.
    _kill_stale_server()

    # Step 2 — start the server, log stdout/stderr to data/server_log.txt.
    log_fh = open(LLAMA_LOG, "w")
    logger.info("Starting llama-server…")
    _llm_process = subprocess.Popen(
        [
            str(LLAMA_BIN),
            "-m",
            str(LLAMA_MODEL),
            "--host",
            LLAMA_HOST,
            "--port",
            str(LLAMA_PORT),
        ],
        stdout=log_fh,
        stderr=log_fh,
    )
    logger.info(
        f"llama-server started (PID {_llm_process.pid}). Waiting for model to load…"
    )

    # Step 3 — quick sanity check: did the process die immediately?
    time.sleep(10)
    if _llm_process.poll() is not None:
        logger.error("llama-server died immediately. Check data/server_log.txt")
        return False

    # Step 4 — health check loop (mirrors enrichment.sh for loop).
    for attempt in range(1, HEALTH_RETRIES + 1):
        time.sleep(HEALTH_INTERVAL)
        if _llama_is_healthy():
            elapsed = attempt * HEALTH_INTERVAL
            logger.info(f"LLM server ready after {elapsed}s.")
            return True
        if _llm_process.poll() is not None:
            logger.error(
                "llama-server process died while waiting. Check data/server_log.txt"
            )
            return False
        logger.debug(f"Health check {attempt}/{HEALTH_RETRIES}…")

    # Timeout — mirror enrichment.sh failure path.
    logger.error("Timeout waiting for LLM server to become healthy.")
    _llm_process.terminate()
    _llm_process = None
    return False


def _stop_llama_server():
    """enrichment.sh step 6 — kill the server on app shutdown."""
    global _llm_process
    if _llm_process and _llm_process.poll() is None:
        _llm_process.terminate()
        try:
            _llm_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _llm_process.kill()
        logger.info("LLM server stopped.")
    _llm_process = None


# ==========================================
# FASTAPI LIFESPAN
# ==========================================


@app.on_event("startup")
async def startup():
    logger.info("QuestionPrep API started.")


@app.on_event("shutdown")
async def shutdown():
    _stop_llama_server()
    logger.info("QuestionPrep API shut down.")


# ==========================================
# PYDANTIC MODELS
# ==========================================


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


# ==========================================
# BACKGROUND TASKS
# ==========================================


def run_ingestion_task(temp_path: Path, user_id: str):
    """CPU/IO-heavy ingestion — chunking, SimHash dedup, DB insert."""
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
            logger.info(f"Ingested file {file_id} for user {user_id}")

        if temp_path.exists():
            temp_path.unlink()

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")


def run_enrichment_task(file_id: str):
    """
    LLM enrichment + vector indexing for one file.

    Starts the llama-server if not already running (same as enrichment.sh),
    then calls EnrichmentManager and VectorIndexer directly — no subprocess needed
    for the Python side since we're already inside uv's environment.
    """
    try:
        # Start (or verify) the LLM server — mirrors enrichment.sh steps 1-4.
        if not _ensure_llama_running():
            logger.error(
                f"Enrichment aborted for {file_id[:8]} — LLM server could not start. "
                "Is the USB stick mounted? Check data/server_log.txt for details."
            )
            return

        logger.info(f"Running enrichment for file {file_id[:8]}…")
        EnrichmentManager().enrich_single_file(file_id)

        logger.info(f"Running vector indexing for file {file_id[:8]}…")
        VectorIndexer().index_file(file_id)

        logger.info(f"Enrichment + indexing complete for file {file_id[:8]}.")

    except Exception as e:
        logger.error(f"Enrichment task failed for {file_id}: {e}")
    finally:
        _stop_llama_server()


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
        "message": "File uploaded. Ingestion running in background.",
        "filename": file.filename,
    }


@app.post("/api/questions/generate")
async def generate_questions(req: GenerateRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_enrichment_task, req.file_id)
    return {"message": f"Enrichment and indexing started for file {req.file_id}"}


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
        session_id=session_id, user_id=req.user_id, db_path=str(DB_PATH)
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
    global _stt
    if session_id in active_sessions:
        del active_sessions[session_id]

    if not active_sessions and _stt is not None:
        del _stt
        _stt = None
        logger.info("Qwen3 ASR model unloaded — no active sessions.")

    return {"message": "Session ended"}


@app.post("/api/analyze-speech")
async def analyze_speech(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        transcript = get_stt().transcribe_opus_bytes(audio_bytes)
        analysis = analyze_disfluencies(transcript)
        return {
            "transcript": transcript,
            "stutter_flag": analysis["stutter_flag"],
            "disfluency_rate": analysis["disfluency_rate"],
            "details": analysis,
        }
    except Exception as e:
        logger.error(f"Speech analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)
