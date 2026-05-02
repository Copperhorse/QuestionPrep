"""
main.py — QuestionPrep FastAPI Backend (Orchestrator)
uv run uvicorn apps.orchestrator.main:app --reload

ds"""

import asyncio
import logging
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Set

from fastapi.responses import FileResponse

# Ensure .mjs files are served with correct MIME type
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")


import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# At top of main.py, with other fastapi imports:
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.responses import Response as PlainResponse
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
from qp_voice.speech_to_text import SpeechToText
from qp_voice.text_to_speech import TextToSpeech

# B25: Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
_DEFAULT_LLAMA_BIN = (
    "/home/copper/Desktop/Project/LLAMACPP/llama.cpp/build/bin/llama-server"
)
_DEFAULT_LLAMA_MODEL = (
    "/home/copper/Desktop/Project/Model/LFM2.5-1.2B-Instruct-Q8_0.gguf"
)

LLAMA_BIN = Path(os.environ.get("LLAMA_BIN", _DEFAULT_LLAMA_BIN))
LLAMA_MODEL = Path(os.environ.get("LLAMA_MODEL", _DEFAULT_LLAMA_MODEL))
LLAMA_HOST = os.environ.get("LLAMA_HOST", "localhost")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_HEALTH_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/models"
LLAMA_LOG = DATA_DIR / "server_log.txt"

HEALTH_RETRIES = 60
HEALTH_INTERVAL = 3  # seconds

# ---------------- SESSION CONFIG ----------------
SESSION_TTL_SECONDS = 1800  # 30 minutes

# ---------------- GLOBAL STATE ----------------
db = DBManager(db_path=str(DB_PATH))
active_sessions: Dict[str, InterviewSession] = {}
_session_last_active: Dict[str, float] = {}

_stt: Optional[SpeechToText] = None
_tts: Optional[TextToSpeech] = None
_llm_process: Optional[subprocess.Popen] = None

_tts_lock = threading.Lock()
_stt_lock = threading.Lock()

_sse_clients: Set[asyncio.Queue] = set()
_active_enrichments: int = 0
_enrichment_lock = threading.Lock()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


async def broadcast(message: str) -> None:
    dead = set()
    for q in list(_sse_clients):
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_clients.difference_update(dead)


def emit(message: str) -> None:
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(message), _main_loop)
    else:
        logger.info(f"[emit-fallback] {message}")


def _inc_enrichments() -> None:
    global _active_enrichments
    with _enrichment_lock:
        _active_enrichments += 1


def _dec_enrichments() -> None:
    global _active_enrichments
    with _enrichment_lock:
        _active_enrichments = max(0, _active_enrichments - 1)


def get_stt() -> SpeechToText:
    global _stt
    if _stt is None:
        with _stt_lock:
            if _stt is None:
                emit("[INFO] asr: ▶ Loading Qwen3-ASR-0.6B model into memory…")
                _stt = SpeechToText()
                emit("[INFO] asr: ✓ Qwen3 ASR model loaded and ready")
    return _stt


def get_tts() -> TextToSpeech:
    global _tts
    if _tts is None:
        with _tts_lock:
            if _tts is None:
                emit("[INFO] tts: ▶ Loading pocket-tts model…")
                _tts = TextToSpeech()
                emit("[INFO] tts: ✓ TextToSpeech ready")
    return _tts


# ==========================================
# B22: BACKGROUND SESSION PRUNING
# ==========================================


async def _session_pruner():
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        now = time.time()
        expired = [
            sid
            for sid, last in _session_last_active.items()
            if now - last > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            active_sessions.pop(sid, None)
            _session_last_active.pop(sid, None)
            logger.info(f"Pruned expired session {sid[:8]}")
            await broadcast(
                f"[WARNING] sessions: ■ Session {sid[:8]} pruned (TTL expired)"
            )
        if expired:
            logger.info(f"Pruned {len(expired)} expired session(s).")


# ==========================================
# LLAMA SERVER LIFECYCLE
# ==========================================


def _llama_is_healthy() -> bool:
    try:
        r = httpx.get(LLAMA_HEALTH_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _kill_stale_server():
    result = subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    if result.returncode == 0:
        logger.info("Killed stale llama-server process.")
        emit("[WARNING] llm: ■ Killed stale llama-server process")
        time.sleep(2)


def _ensure_llama_running() -> bool:
    global _llm_process

    if _llama_is_healthy():
        logger.info("LLM server already running and healthy.")
        emit("[INFO] llm: ✓ LLM server already running and healthy")
        return True

    if not LLAMA_BIN.exists():
        logger.error(f"llama-server binary not found: {LLAMA_BIN}")
        emit(f"[ERROR] llm: ✗ llama-server binary not found: {LLAMA_BIN}")
        return False
    if not LLAMA_MODEL.exists():
        logger.error(f"LLM model not found: {LLAMA_MODEL}")
        emit(f"[ERROR] llm: ✗ Model file not found: {LLAMA_MODEL}")
        return False

    _kill_stale_server()

    log_fh = open(LLAMA_LOG, "w")
    logger.info("Starting llama-server…")
    emit(f"[INFO] llm: ▶ Starting llama-server ({LLAMA_MODEL.name})…")
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
    log_fh.close()

    logger.info(
        f"llama-server started (PID {_llm_process.pid}). Waiting for model to load…"
    )
    emit(
        f"[INFO] llm: llama-server PID {_llm_process.pid} — waiting for model to load…"
    )

    time.sleep(10)
    if _llm_process.poll() is not None:
        logger.error("llama-server died immediately. Check data/server_log.txt")
        emit(
            "[ERROR] llm: ✗ llama-server exited immediately — check data/server_log.txt"
        )
        return False

    for attempt in range(1, HEALTH_RETRIES + 1):
        time.sleep(HEALTH_INTERVAL)
        if _llama_is_healthy():
            elapsed = attempt * HEALTH_INTERVAL
            logger.info(f"LLM server ready after {elapsed}s.")
            emit(f"[INFO] llm: ✓ LLM server ready after {elapsed}s")
            return True
        if _llm_process.poll() is not None:
            logger.error(
                "llama-server process died while waiting. Check data/server_log.txt"
            )
            emit(
                "[ERROR] llm: ✗ llama-server died during startup — check data/server_log.txt"
            )
            return False
        emit(f"[DEBUG] llm: health check {attempt}/{HEALTH_RETRIES}…")
        logger.debug(f"Health check {attempt}/{HEALTH_RETRIES}…")

    logger.error("Timeout waiting for LLM server to become healthy.")
    emit("[ERROR] llm: ✗ Timeout waiting for LLM server — aborting")
    _llm_process.terminate()
    _llm_process = None
    return False


def _stop_llama_server():
    global _llm_process
    if _llm_process and _llm_process.poll() is None:
        _llm_process.terminate()
        try:
            _llm_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _llm_process.kill()
        logger.info("LLM server stopped.")
        emit("[INFO] llm: ■ LLM server stopped")
    _llm_process = None


# ==========================================
# LIFESPAN
# ==========================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    asyncio.create_task(_session_pruner())
    await broadcast("[INFO] orchestrator: ✓ QuestionPrep API started")
    logger.info("QuestionPrep API started.")
    yield
    await broadcast("[INFO] orchestrator: ■ QuestionPrep API shutting down")
    _stop_llama_server()
    logger.info("QuestionPrep API shut down.")


# ==========================================
# APP & TEMPLATES
# ==========================================

app = FastAPI(title="QuestionPrep API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


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


class TTSRequest(BaseModel):
    text: str


class EmailUpdateRequest(BaseModel):
    new_email: str


# ==========================================
# BACKGROUND TASKS
# ==========================================


def run_ingestion_task(temp_path: Path, user_id: str):
    fname = temp_path.name.split("_", 1)[-1]
    emit(f"[INFO] ingest: ▶ Ingesting {fname} for user {user_id[:8]}…")
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

        if success and file_id:
            db.assign_file_to_user(user_id, file_id)
            logger.info(f"Ingested file {file_id} for user {user_id}")
            emit(f"[INFO] ingest: ✓ {fname} ingested — file_id {file_id[:8]}")
        else:
            emit(
                f"[WARNING] ingest: ■ {fname} skipped (duplicate or conversion failed)"
            )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        emit(f"[ERROR] ingest: ✗ Ingestion failed for {fname}: {e}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def run_enrichment_task(file_id: str):
    _inc_enrichments()
    emit(f"[INFO] enricher: ▶ Starting enrichment for file {file_id[:8]}…")
    try:
        if not _ensure_llama_running():
            logger.error(
                f"Enrichment aborted for {file_id[:8]} — LLM server could not start. "
                "Is the USB stick mounted? Check data/server_log.txt for details."
            )
            emit(
                f"[ERROR] enricher: ✗ Enrichment aborted for {file_id[:8]} — LLM server failed to start"
            )
            return

        logger.info(f"Running enrichment for file {file_id[:8]}…")
        emit(f"[INFO] enricher: Running LLM enrichment for {file_id[:8]}…")
        EnrichmentManager().enrich_single_file(file_id)

        logger.info(f"Running vector indexing for file {file_id[:8]}…")
        emit(f"[INFO] enricher: Running vector indexing for {file_id[:8]}…")
        VectorIndexer().index_file(file_id)

        logger.info(f"Enrichment + indexing complete for file {file_id[:8]}.")
        emit(f"[INFO] enricher: ✓ Enrichment + indexing complete for {file_id[:8]}")

    except Exception as e:
        logger.error(f"Enrichment task failed for {file_id}: {e}")
        emit(f"[ERROR] enricher: ✗ Enrichment failed for {file_id[:8]}: {e}")
    finally:
        _dec_enrichments()


# ==========================================
# PAGE ROUTES (HTML)
# ==========================================
@app.get("/companion", response_class=HTMLResponse)
async def get_companion_page(request: Request):
    return templates.TemplateResponse("companion.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/sw.js")
async def service_worker():
    return FileResponse(
        BASE_DIR / "sw.js",
        media_type="application/javascript",
    )


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
    user_id = db.create_user(
        username=user.username,
        email=user.email,
        password=user.password,
    )
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    return {"message": "User created successfully", "user_id": user_id}


@app.post("/api/auth/login")
async def login(user: LoginRequest):
    db_user = db.get_user_by_username(user.username)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not db.verify_password(user.username, user.password):
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"token": "mock-jwt-token", "user": db_user}


@app.get("/session", response_class=HTMLResponse)
async def get_session_page(request: Request):
    return templates.TemplateResponse("session.html", {"request": request})


@app.get("/api/auth/profile")
async def get_profile(user_id: str):
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"profile": user}


@app.delete("/api/users/{user_id}")
async def delete_user_account(user_id: str):
    try:
        db.delete_user(user_id)
        return {"status": "success", "message": "Account deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account.")


@app.put("/api/users/{user_id}/email")
async def update_user_email(user_id: str, payload: EmailUpdateRequest):
    try:
        db.update_user_email(user_id, payload.new_email)
        return {"status": "success", "message": "Email updated successfully."}
    except Exception as e:
        logger.error(f"Failed to update email for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update email.")


# ==========================================
# PIPELINE API ROUTES
# ==========================================


@app.post("/api/files/ingest")
@limiter.limit("5/minute")
async def ingest_file(
    request: Request,
    user_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
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
@limiter.limit("10/minute")
async def generate_questions(
    request: Request,
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    existing = db.get_questions_for_file(req.file_id)
    if existing:
        return {
            "message": (
                f"File {req.file_id[:8]} already has {len(existing)} question(s). "
                "Delete them first to re-enrich."
            ),
            "skipped": True,
        }
    background_tasks.add_task(run_enrichment_task, req.file_id)
    return {"message": f"Enrichment and indexing started for file {req.file_id}"}


@app.get("/api/files")
async def list_user_files(user_id: str):
    return {"files": db.get_files_for_user(user_id)}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str, user_id: str):
    user_files = db.get_files_for_user(user_id)
    if not any(f["file_id"] == file_id for f in user_files):
        raise HTTPException(status_code=403, detail="File not found or access denied")

    try:
        indexer = VectorIndexer()
        indexer.delete_embeddings_for_file(file_id)
    except Exception as e:
        logger.warning(f"Chroma cleanup failed for {file_id[:8]}: {e}")

    deleted = db.delete_file(file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File and embeddings deleted successfully"}


@app.delete("/api/files/all/{user_id}")
async def delete_all_user_files(user_id: str):
    files = db.get_files_for_user(user_id)
    indexer = VectorIndexer()
    for f in files:
        fid = f.get("file_id") or f.get("id")
        if fid:
            try:
                indexer.delete_embeddings_for_file(fid)
            except Exception:
                pass
            db.delete_file(fid)
    return {"status": "success", "message": "All files deleted."}


@app.get("/api/files/{file_id}/audit")
async def get_file_audit(file_id: str):
    try:
        metadata_list = db.get_files_for_user(
            None
        )  # Not ideal, but safe fallback if no direct get_file method exists
        metadata = next(
            (f for f in metadata_list if f.get("file_id") == file_id),
            {"file_id": file_id},
        )
    except Exception:
        metadata = {"file_id": file_id}

    try:
        questions = db.get_questions_for_file(file_id) or []
    except Exception:
        questions = []

    safe_questions = [
        {
            "question_id": q.get("question_id"),
            "question_text": q.get("question_text"),
            "difficulty": q.get("difficulty"),
            "question_type": q.get("question_type"),
        }
        for q in questions
    ]

    try:
        rejected = db.get_rejections_for_file(file_id) or []
    except AttributeError:
        rejected = []

    return {
        "metadata": metadata,
        "total_questions": len(safe_questions),
        "questions": safe_questions,
        "rejected": rejected,
    }


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
    _session_last_active[session_id] = time.time()
    await broadcast(
        f"[INFO] sessions: ✓ New session {session_id[:8]} started for user {req.user_id[:8]}"
    )
    return {"session_id": session_id, "first_question": first_question}


@app.post("/api/interview/evaluate")
async def evaluate_answer(req: EvaluateRequest):
    session = active_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _session_last_active[req.session_id] = time.time()
    return session.evaluate_turn(req.user_answer)


@app.get("/api/interview/{session_id}/status")
async def get_session_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    current_q = session.ctx.current_question
    return {
        "alive": True,
        "state": session.state.name,
        "current_question": current_q.text if current_q else None,
        "questions_answered": len(session.ctx.history),
    }


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
    _session_last_active.pop(session_id, None)

    if not active_sessions and _stt is not None:
        with _stt_lock:
            if not active_sessions and _stt is not None:
                _stt = None
                logger.info("Qwen3 ASR model unloaded — no active sessions.")
                await broadcast(
                    "[INFO] asr: ■ Qwen3 ASR model unloaded — no active sessions remain"
                )

    await broadcast(
        f"[INFO] sessions: ■ Session {session_id[:8]} ended ({len(active_sessions)} remaining)"
    )
    return {"message": "Session ended"}


# ==========================================
# VOICE API ROUTES
# ==========================================


@app.post("/api/analyze-speech")
async def analyze_speech(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        loop = asyncio.get_running_loop()
        stt = await loop.run_in_executor(None, get_stt)

        result = await loop.run_in_executor(
            None, stt.transcribe_and_analyze, audio_bytes
        )
        words = len((result.get("transcript") or "").split())
        await broadcast(
            f"[INFO] asr: ✓ Transcribed {words} words — "
            f"stutter={'yes' if result.get('stutter_flag') else 'no'}  "
            f"rate={result.get('disfluency_rate', 0):.2%}"
        )
        return result

    except FileNotFoundError as e:
        msg = "ffmpeg not found. Install it: sudo apt install ffmpeg (Linux) or brew install ffmpeg (Mac)."
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)
    except RuntimeError as e:
        if "ffmpeg" in str(e).lower():
            msg = f"ffmpeg decode failed: {e}. Check the audio format sent by the browser."
        else:
            msg = str(e)
        logger.exception(f"STT runtime error: {e}")
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        logger.exception(f"Speech analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        loop = asyncio.get_running_loop()
        tts = await loop.run_in_executor(None, get_tts)
        wav_bytes = await loop.run_in_executor(None, tts.generate_wav_bytes, req.text)

        audio_data = wav_bytes.read()
        await broadcast(
            f"[INFO] tts: ✓ Generated {len(audio_data) // 1024} KB of audio"
        )
        return PlainResponse(content=audio_data, media_type="audio/wav")
    except Exception as e:
        logger.exception(f"TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# SSE STATUS ENDPOINTS
# ==========================================


@app.get("/api/events")
async def sse_events(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_clients.add(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_clients.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
async def get_system_status():
    return {
        "llm_server_running": _llama_is_healthy(),
        "active_enrichments": _active_enrichments,
        "sessions_with_asr": 1 if _stt is not None else 0,
        "active_sessions": len(active_sessions),
    }


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)
