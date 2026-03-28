"""
rate_limiting.py  —  Drop-in rate-limiting helpers for main.py

Fix applied:
  B25 - No rate limiting on file ingest or enrichment endpoints.
        POST /api/files/ingest starts a Docling OCR background task with no
        per-user limit. POST /api/questions/generate has no guard against
        concurrent or duplicate enrichment of the same file.

HOW TO APPLY
────────────
1. pip install slowapi  (add to apps/orchestrator/pyproject.toml dependencies)
2. Copy the three blocks below into main.py where marked.

The changes to main.py are shown as diff-style excerpts so they can be applied
precisely without re-pasting the entire file.
"""

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — Add to imports at the top of main.py
# ──────────────────────────────────────────────────────────────────────────────

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — Add after `app = FastAPI(...)` in main.py
# ──────────────────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — Replace the two pipeline endpoint definitions in main.py
# ──────────────────────────────────────────────────────────────────────────────


# ── Ingest ──  (replaces the existing POST /api/files/ingest handler)
# @app.post("/api/files/ingest")
# @limiter.limit("5/minute")          # B25: max 5 uploads per minute per IP
# async def ingest_file(
#     request: Request,               # B25: slowapi requires Request in signature
#     user_id: str,
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
# ):
#     if not db.get_user_by_id(user_id):
#         raise HTTPException(status_code=404, detail="User not found")
#
#     temp_path = DATA_DIR / f"{uuid.uuid4()}_{file.filename}"
#     with open(temp_path, "wb") as f:
#         f.write(await file.read())
#
#     background_tasks.add_task(run_ingestion_task, temp_path, user_id)
#     return {
#         "status": "processing",
#         "message": "File uploaded. Ingestion running in background.",
#         "filename": file.filename,
#     }


# ── Generate questions ──  (replaces the existing POST /api/questions/generate)
# @app.post("/api/questions/generate")
# @limiter.limit("10/minute")          # B25: max 10 generation requests per minute per IP
# async def generate_questions(
#     request: Request,                # B25: slowapi requires Request in signature
#     req: GenerateRequest,
#     background_tasks: BackgroundTasks,
# ):
#     # B25: Skip enrichment if this file already has questions — prevents duplicate
#     #      concurrent enrichment of the same file draining CPU.
#     existing = db.get_questions_for_file(req.file_id)
#     if existing:
#         return {
#             "message": f"File {req.file_id[:8]} already has {len(existing)} question(s). "
#                        "Delete them first to re-enrich.",
#             "skipped": True,
#         }
#     background_tasks.add_task(run_enrichment_task, req.file_id)
#     return {"message": f"Enrichment and indexing started for file {req.file_id}"}
