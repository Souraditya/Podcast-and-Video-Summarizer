"""
Stage 4 — Output: FastAPI backend + video-synced UI.

Runs the same pipeline as the CLI in a background thread and serves the
result to web/index.html. Works on media with NO existing chapters: chapters
and summaries are generated from the transcript alone.

Run:
    uvicorn podsum.app:app --reload        # then open http://127.0.0.1:8000
Env (optional): PODSUM_COOKIES_FILE=youtube_cookies.txt   ANTHROPIC_API_KEY=...
"""
from __future__ import annotations

import os
import threading
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from podsum import config
from podsum.data.ingest import extract_youtube_id
from podsum.pipeline import run

RUNS = Path(config.DATA_DIR) / "runs" / "web"
INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
STAGES = ["Ingest", "Transcribe", "Clean", "Segment", "Summarize", "Report"]

app = FastAPI(title="Podcast / Video Summarizer")
JOBS: dict[str, dict] = {}


def _worker(job_id: str, source: str, method: str) -> None:
    job = JOBS[job_id]

    def on_progress(i: int, label: str) -> None:
        job.update(stage=i, label=label)

    try:
        result = run(source, str(RUNS / job_id), method=method,
                     cookies_file=os.getenv("PODSUM_COOKIES_FILE"), on_progress=on_progress, evaluate=False)
        job.update(status="done", stage=len(STAGES), result=result)
    except Exception as e:  # surfaced to the UI instead of dying silently in the thread
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX)


@app.post("/api/jobs")
async def create_job(url: str = Form(""), method: str = Form("semantic"),
                     file: UploadFile | None = File(None)) -> dict:
    if method not in ("semantic", "llm"):
        raise HTTPException(400, "method must be 'semantic' or 'llm'")
    job_id = uuid.uuid4().hex[:10]
    out_dir = RUNS / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if file is not None and file.filename:
        path = out_dir / f"original{Path(file.filename).suffix.lower()}"
        path.write_bytes(await file.read())
        source, media = str(path), {"type": "file", "url": f"/api/jobs/{job_id}/media"}
    elif url.strip():
        source = url.strip()
        yid = extract_youtube_id(source)
        media = {"type": "youtube", "id": yid} if yid else {"type": "none"}
    else:
        raise HTTPException(400, "provide a URL or upload a file")

    JOBS[job_id] = {"status": "running", "stage": 0, "label": "Queued", "stages": STAGES,
                    "error": None, "media": media, "result": None, "path": source}
    threading.Thread(target=_worker, args=(job_id, source, method), daemon=True).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "path"}


@app.get("/api/jobs/{job_id}/media")
def get_media(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or job["media"]["type"] != "file":
        raise HTTPException(404, "no media for this job")
    return FileResponse(job["path"])  # Starlette FileResponse honours Range requests, so seeking works
