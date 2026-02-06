from __future__ import annotations

import os
import socket
import time
import json
import logging
import tempfile
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any
import ulid
import httpx
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "ASR Service (CPU)"
MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "auto")
ASYNC_THRESHOLD_SEC = int(os.getenv("ASR_ASYNC_THRESHOLD", "300"))
JOBS_DIR = os.getenv("ASR_JOBS_DIR", "/srv/jobs")

os.makedirs(JOBS_DIR, exist_ok=True)

# Global process pool executor for heavy CPU tasks.
# We use 1 worker for transcription. This process is SEPARATE from the API workers.
# This prevents the CPU-heavy transcription from starving the web server.
executor = ProcessPoolExecutor(max_workers=1)

app = FastAPI(title=APP_TITLE)

def transcribe_worker(file_path: str, model_size: str, compute_type: str, language: Optional[str], vad_filter: bool, timestamps: bool):
    """
    This function runs in a completely separate OS process.
    """
    start_time = time.time()
    try:
        # Load model locally in the worker process (required as it's a new process)
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        
        segments, info = model.transcribe(
            file_path,
            language=language,
            vad_filter=vad_filter,
        )

        seg_list = []
        full_text_parts = []
        for s in segments:
            txt = (s.text or "").strip()
            if txt:
                full_text_parts.append(txt)
            if timestamps:
                seg_list.append({"start": float(s.start), "end": float(s.end), "text": txt})

        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "result": {
                "text": " ".join(full_text_parts).strip(),
                "language": getattr(info, "language", None),
                "segments": seg_list,
                "metadata": {
                    "model_size": model_size,
                    "compute_type": compute_type,
                    "device": "cpu",
                    "processing_time": round(processing_time, 3),
                    "hostname": socket.gethostname()
                }
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class AsrMetadata(BaseModel):
    model_size: str
    compute_type: str
    device: str
    processing_time: float
    hostname: str


class AsrResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: list[SegmentOut]
    metadata: AsrMetadata


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "asr",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }


def save_job_result(job_id: str, data: Dict[str, Any], status: str = "completed"):
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    job_data = {
        "job_id": job_id,
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
        "result": data
    }
    with open(job_file, "w") as f:
        json.dump(job_data, f)


def get_job_data(job_id: str) -> Optional[Dict[str, Any]]:
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(job_file):
        return None
    with open(job_file, "r") as f:
        return json.load(f)


async def run_async_job(job_id: str, file_path: str, lang: Optional[str], vad: bool, ts: bool, callback: Optional[str]):
    loop = asyncio.get_running_loop()
    try:
        # Offload to the separate process
        future = executor.submit(transcribe_worker, file_path, MODEL_SIZE, COMPUTE_TYPE, lang, vad, ts)
        # Use run_in_executor to avoid blocking the event loop while waiting for the process result
        response = await loop.run_in_executor(None, future.result)

        if response["success"]:
            save_job_result(job_id, response["result"], "completed")
            if callback:
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(callback, json={"job_id": job_id, "status": "completed", "result": response["result"]})
                except Exception as e:
                    logger.error("Callback failed for job %s: %s", job_id, e)
        else:
            save_job_result(job_id, {"error": response["error"]}, "failed")
            
    except Exception as e:
        logger.error("Async job wrapper failed for %s: %s", job_id, e)
        save_job_result(job_id, {"error": str(e)}, "failed")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/v1/asr")
async def asr(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    language: str = Form(DEFAULT_LANGUAGE),
    timestamps: bool = Form(True),
    vad_filter: bool = Form(True),
    callback_url: Optional[str] = Form(None),
    force_async: bool = Form(False)
):
    logger.info("ASR request received file=%s file_url=%s", 
                file.filename if file else "None", 
                file_url)
    
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")
    
    # Save to temp file
    suffix = ".bin"
    if file:
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    elif file_url:
        # Try to extract suffix from URL
        from urllib.parse import urlparse
        path = urlparse(file_url).path
        suffix = os.path.splitext(path)[1] or ".bin"

    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        if file:
            with os.fdopen(temp_fd, 'wb') as tmp:
                tmp.write(await file.read())
        else:
            # Download from URL
            logger.info("Downloading file from %s", file_url)
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream("GET", file_url) as response:
                    if response.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {response.status_code}")
                    with os.fdopen(temp_fd, 'wb') as tmp:
                        async for chunk in response.aiter_bytes():
                            tmp.write(chunk)
        
        try:
            audio = AudioSegment.from_file(temp_path)
            duration_sec = len(audio) / 1000.0
            logger.info("Audio duration detected: %.2fs", duration_sec)
        except Exception as e:
            logger.warning("Could not detect duration: %s", e)
            duration_sec = 0

        lang_val = None if (language or "").lower() == "auto" else language

        if duration_sec > ASYNC_THRESHOLD_SEC or force_async:
            # Using ULID and hostname for the job_id
            job_id = f"{ulid.ULID()}-{socket.gethostname()}"
            logger.info("Switching to async mode for job %s", job_id)
            
            save_job_result(job_id, {}, "processing")
            
            # Fire and forget the background task
            asyncio.create_task(run_async_job(job_id, temp_path, lang_val, vad_filter, timestamps, callback_url))
            
            return JSONResponse(status_code=202, content={
                "job_id": job_id,
                "status_url": f"/v1/asr/status/{job_id}",
                "download_url": f"/v1/asr/download/{job_id}",
                "message": "Job accepted. Processing in background."
            })

        # Sync path: Also offload to the separate process to keep this worker's event loop free
        loop = asyncio.get_running_loop()
        future = executor.submit(transcribe_worker, temp_path, MODEL_SIZE, COMPUTE_TYPE, lang_val, vad_filter, timestamps)
        response = await loop.run_in_executor(None, future.result)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])
            
        return response["result"]

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("ASR endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/asr/status/{job_id}")
async def get_status(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "updated_at": data["updated_at"]
    }


@app.get("/v1/asr/download/{job_id}")
async def download_result(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {data['status']}")
    return data["result"]
