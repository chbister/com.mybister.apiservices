from __future__ import annotations

import os
import socket
import logging
import io
import json
import asyncio
import ulid
from datetime import datetime
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

from .engines.tesseract import TesseractEngine
from .engines.easy_ocr import EasyOCREngine
from .engines.paddle_ocr import PaddleOCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "OCR Service"
app = FastAPI(title=APP_TITLE)

# Configuration
OCR_ENGINE_TYPE = os.getenv("OCR_ENGINE", "EASYOCR").upper()
OCR_LANG = os.getenv("OCR_LANG", "de,en")
OCR_CONFIG = os.getenv("OCR_CONFIG")
OCR_PROFILE = os.getenv("OCR_PROFILE", "screenshot").lower()
OCR_MAX_IMAGE_WIDTH = int(os.getenv("OCR_MAX_IMAGE_WIDTH", "1600"))
OCR_MAX_IMAGE_HEIGHT = int(os.getenv("OCR_MAX_IMAGE_HEIGHT", "3000"))

# Role configuration
SERVICE_ROLE = os.getenv("SERVICE_ROLE", "API").upper() # API or WORKER

# Async Jobs Configuration
JOBS_DIR = os.getenv("OCR_JOBS_DIR", "/srv/jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")

# Singleton engines
default_engine = None
engines = {}

def get_engine(engine_type: str = None, lang: str = None, config: str = None):
    global default_engine
    
    etype = engine_type.upper() if engine_type else OCR_ENGINE_TYPE
    elang = lang if lang else OCR_LANG
    econfig = config if config else OCR_CONFIG

    if etype == OCR_ENGINE_TYPE and (elang is None or elang == OCR_LANG) and (econfig is None or econfig == OCR_CONFIG):
        if default_engine is None:
            logger.info("Initializing default OCR engine: %s", etype)
            default_engine = _create_engine(etype, elang, econfig)
        else:
            logger.info("Reusing initialized default OCR engine: %s", etype)
        return default_engine

    cache_key = f"{etype}_{elang}_{econfig}"
    if cache_key not in engines:
        logger.info("Initializing specialized OCR engine: %s (lang=%s, config=%s)", etype, elang, econfig)
        engines[cache_key] = _create_engine(etype, elang, econfig)
    else:
        logger.info("Reusing specialized OCR engine: %s", etype)
    return engines[cache_key]

def _create_engine(etype: str, elang: str, econfig: str):
    if etype == "PADDLE":
        return PaddleOCREngine(lang=elang or "latin", config=econfig)
    elif etype == "EASYOCR":
        return EasyOCREngine(lang=elang or "de,en", config=econfig)
    else:
        return TesseractEngine(lang=elang, config=econfig)

@app.on_event("startup")
async def startup_event():
    if SERVICE_ROLE == "WORKER":
        logger.info("Starting OCR worker warm-up...")
        get_engine()
        logger.info("OCR worker warm-up complete.")
    else:
        logger.info("OCR API started (Role: %s).", SERVICE_ROLE)

async def get_image_bytes(file: Optional[UploadFile], file_url: Optional[str]) -> bytes:
    if file:
        return await file.read()
    elif file_url:
        logger.info("Downloading image from %s", file_url)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(file_url, timeout=30.0)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {response.status_code}")
                return response.content
            except HTTPException:
                raise
            except Exception as e:
                logger.error("Download failed: %s", str(e))
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

class Metadata(BaseModel):
    ocr_engine: str
    ocr_language: str
    processing_time: Optional[float] = None

class OCRResponse(BaseModel):
    text: str
    lines: List[str]
    count: int
    hostname: str
    file_name: Optional[str] = None
    file_url: Optional[str] = None
    metadata: Metadata

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    download_url: str
    callback_url: Optional[str] = None
    message: str

def enqueue_job(job_id: str, contents: bytes, callback_url: Optional[str], file_name: Optional[str], file_url: Optional[str]):
    image_path = os.path.join(JOBS_DIR, f"{job_id}.img")
    with open(image_path, "wb") as f:
        f.write(contents)
        
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "payload": {
            "callback_url": callback_url,
            "file_name": file_name,
            "file_url": file_url
        }
    }
    with open(job_file, "w") as f:
        json.dump(job_data, f)
    logger.info("Job %s enqueued.", job_id)

def get_job_data(job_id: str) -> Optional[Dict[str, Any]]:
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(job_file):
        return None
    try:
        with open(job_file, "r") as f:
            return json.load(f)
    except Exception:
        return None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ocr",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
        "config": {
            "ocr_engine": OCR_ENGINE_TYPE,
            "ocr_language": OCR_LANG
        }
    }

@app.post("/v1/ocr", response_model=Union[OCRResponse, AsyncJobResponse], status_code=202)
async def process_image(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None)
):
    logger.info("OCR request received file=%s, callback=%s", 
                file.filename if file else "None", callback_url)
    
    contents = await get_image_bytes(file, file_url)
    file_name = file.filename if file else None
    
    job_id = f"{ulid.ULID()}-{socket.gethostname()}"
    enqueue_job(job_id, contents, callback_url, file_name, file_url)
    
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/ocr/v1/ocr/status/{job_id}",
            "download_url": f"/ocr/v1/ocr/download/{job_id}",
            "callback_url": callback_url,
            "message": "Job accepted and enqueued."
        }
    )

@app.get("/v1/ocr/status/{job_id}")
async def get_status(job_id: str):
    job_data = get_job_data(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "updated_at": job_data.get("updated_at", job_data.get("created_at"))
    }

@app.get("/v1/ocr/download/{job_id}")
async def download_result(job_id: str):
    job_data = get_job_data(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data["status"] == "completed":
        return job_data["result"]
    elif job_data["status"] == "failed":
        return JSONResponse(status_code=500, content={"detail": f"Job failed: {job_data.get('error', 'Unknown error')}"})
    else:
        return JSONResponse(status_code=200, content={"detail": f"Job status is {job_data['status']}"})
