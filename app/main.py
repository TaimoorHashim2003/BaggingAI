from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import shutil
import json
import asyncio
import os
from . import processor

BASE_DIR = Path(__file__).parent.parent
JOBS_DIR = BASE_DIR / "uploads" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="BaggingAI")

# serve static UI
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/clips", StaticFiles(directory=JOBS_DIR), name="clips")

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    labels: str = Form(""),
    keywords: str = Form(""),
    max_len: float = Form(30.0),
    padding: float = Form(4.0),
    aspect: str = Form("9:16")
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "input" + Path(file.filename).suffix
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = {
        "id": job_id,
        "input_filename": str(input_path.name),
        "labels": labels,
        "keywords": keywords,
        "max_len": float(max_len),
        "padding": float(padding),
        "aspect": aspect,
        "status": "queued",
        "progress": 0,
        "clips": []
    }
    (job_dir / "meta.json").write_text(json.dumps(meta))

    # launch background task
    background_tasks.add_task(processor.process_job, job_id, str(input_path), meta)

    return JSONResponse({"job_id": job_id})

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job_dir = JOBS_DIR / job_id
    meta_file = job_dir / "meta.json"
    if not meta_file.exists():
        return JSONResponse({"error": "job not found"}, status_code=404)
    meta = json.loads(meta_file.read_text())
    return JSONResponse(meta)

@app.get("/jobs/{job_id}/clips")
async def job_clips(job_id: str):
    job_dir = JOBS_DIR / job_id
    clips_dir = job_dir / "clips"
    if not clips_dir.exists():
        return JSONResponse({"clips": []})
    clips = [p.name for p in sorted(clips_dir.glob("*.mp4"))]
    # return URLs
    urls = [f"/clips/{job_id}/clips/{c}" for c in clips]
    return JSONResponse({"clips": urls})
