from __future__ import annotations

import os
import time
import json
import logging
import socket
import io
import httpx
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional
from PIL import Image

# Import shared logic from app
from .engines.tesseract import TesseractEngine
from .engines.easy_ocr import EasyOCREngine
from .engines.paddle_ocr import PaddleOCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ocr-worker")

# Configuration (mirrored from main.py)
OCR_ENGINE_TYPE = os.getenv("OCR_ENGINE", "EASYOCR").upper()
OCR_LANG = os.getenv("OCR_LANG", "de,en")
OCR_CONFIG = os.getenv("OCR_CONFIG")
JOBS_DIR = os.getenv("OCR_JOBS_DIR", "/srv/jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# Worker-specific configuration
POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "1.0"))
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")
OCR_PROCESSING_TIMEOUT = int(os.getenv("OCR_PROCESSING_TIMEOUT", "120"))
OCR_MAX_IMAGE_WIDTH = int(os.getenv("OCR_MAX_IMAGE_WIDTH", "1600"))
OCR_MAX_IMAGE_HEIGHT = int(os.getenv("OCR_MAX_IMAGE_HEIGHT", "3000"))
OCR_REQUEUE_STALE_PROCESSING_JOBS = os.getenv("OCR_REQUEUE_STALE_PROCESSING_JOBS", "true").lower() == "true"

# Singleton engines in worker context
default_engine = None
engines = {}

def get_engine(engine_type: str = None, lang: str = None, config: str = None):
    global default_engine
    
    etype = engine_type.upper() if engine_type else OCR_ENGINE_TYPE
    elang = lang if lang else OCR_LANG
    econfig = config if config else OCR_CONFIG

    if etype == OCR_ENGINE_TYPE and (elang is None or elang == OCR_LANG) and (econfig is None or econfig == OCR_CONFIG):
        if default_engine is None:
            logger.info("Initializing default OCR engine in worker: %s", etype)
            default_engine = _create_engine(etype, elang, econfig)
        return default_engine

    cache_key = f"{etype}_{elang}_{econfig}"
    if cache_key not in engines:
        logger.info("Initializing specialized OCR engine in worker: %s (lang=%s, config=%s)", etype, elang, econfig)
        engines[cache_key] = _create_engine(etype, elang, econfig)
    return engines[cache_key]

def _create_engine(etype: str, elang: str, econfig: str):
    if etype == "PADDLE":
        return PaddleOCREngine(lang=elang or "latin", config=econfig)
    elif etype == "EASYOCR":
        return EasyOCREngine(lang=elang or "de,en", config=econfig)
    else:
        return TesseractEngine(lang=elang, config=econfig)

async def send_callback(url: str, payload: Dict[str, Any]):
    headers = {}
    if CALLBACK_AUTH_SECRET:
        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
    
    logger.info("Sending callback to %s", url)
    async with httpx.AsyncClient() as client:
        try:
            cb_response = await client.post(url, json=payload, headers=headers, timeout=15.0)
            logger.info("Callback sent. Status: %s", cb_response.status_code)
        except Exception as e:
            logger.error("Failed to send callback: %s", str(e))

def update_job_status(job_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    try:
        if os.path.exists(job_file):
            with open(job_file, "r") as f:
                data = json.load(f)
        else:
            data = {"job_id": job_id}
            
        data["status"] = status
        data["updated_at"] = datetime.utcnow().isoformat()
        if result is not None:
            data["result"] = result
        if error is not None:
            data["error"] = error
            
        with open(job_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error("Failed to update job status for %s: %s", job_id, str(e))

def resize_image_if_needed(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= OCR_MAX_IMAGE_WIDTH and height <= OCR_MAX_IMAGE_HEIGHT:
        return image
    
    logger.info("Resizing image from %dx%d", width, height)
    image.thumbnail((OCR_MAX_IMAGE_WIDTH, OCR_MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    new_width, new_height = image.size
    logger.info("Resized to %dx%d", new_width, new_height)
    return image

async def process_job(job_id: str, job_data: Dict[str, Any]):
    logger.info("Processing job %s", job_id)
    update_job_status(job_id, "processing")
    
    try:
        payload = job_data.get("payload", {})
        file_name = payload.get("file_name")
        file_url = payload.get("file_url")
        callback_url = payload.get("callback_url")
        
        # Load image
        image_path = os.path.join(JOBS_DIR, f"{job_id}.img")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found for job {job_id}")
            
        with open(image_path, "rb") as f:
            contents = f.read()
            
        start_time = time.time()
        image = Image.open(io.BytesIO(contents))
        
        # 0. Resize Step
        image = resize_image_if_needed(image)
        
        # 1. OCR Step
        engine = get_engine()
        raw_text = engine.extract_text(image)
        
        processing_time = time.time() - start_time
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        result = {
            "text": raw_text,
            "lines": lines,
            "count": len(lines),
            "hostname": socket.gethostname(),
            "file_name": file_name,
            "file_url": file_url,
            "metadata": {
                "ocr_engine": engine.__class__.__name__.replace("Engine", "").upper(),
                "ocr_language": engine.lang,
                "processing_time": round(processing_time, 4)
            }
        }
        
        update_job_status(job_id, "completed", result=result)
        logger.info("Job %s completed in %.2fs", job_id, processing_time)
        
        if callback_url:
            callback_payload = {
                "job_id": job_id,
                "status": "completed",
                "result": result
            }
            await send_callback(callback_url, callback_payload)
            
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, str(e), exc_info=True)
        update_job_status(job_id, "failed", error=str(e))
        
        callback_url = job_data.get("payload", {}).get("callback_url")
        if callback_url:
            callback_payload = {
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
            await send_callback(callback_url, callback_payload)

async def cleanup_stale_jobs():
    logger.info("Cleaning up stale jobs...")
    try:
        job_files = [f for f in os.listdir(JOBS_DIR) if f.endswith(".json")]
        for jf in job_files:
            job_path = os.path.join(JOBS_DIR, jf)
            try:
                with open(job_path, "r") as f:
                    job_data = json.load(f)
                
                if job_data.get("status") == "processing":
                    updated_at_str = job_data.get("updated_at")
                    if not updated_at_str:
                        updated_at_str = job_data.get("created_at")
                    
                    if updated_at_str:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        delta = (datetime.utcnow() - updated_at).total_seconds()
                        if delta > OCR_PROCESSING_TIMEOUT:
                            job_id = job_data.get("job_id")
                            if OCR_REQUEUE_STALE_PROCESSING_JOBS:
                                logger.info("Requeuing stale job %s (stale for %.2fs)", job_id, delta)
                                update_job_status(job_id, "queued")
                            else:
                                logger.info("Marking stale job %s as failed (stale for %.2fs)", job_id, delta)
                                update_job_status(job_id, "failed", error="Job timed out during processing (stale)")
            except Exception as e:
                logger.error("Error cleaning up stale job file %s: %s", jf, str(e))
    except Exception as e:
        logger.error("Error during stale job cleanup: %s", str(e))

async def worker_loop():
    logger.info("OCR Worker started. Polling directory: %s", JOBS_DIR)
    
    # Clean up stale jobs from previous runs
    await cleanup_stale_jobs()
    
    # Warm up default engine
    get_engine()
    logger.info("Worker warm-up complete.")
    
    running = True
    
    def signal_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received. Exiting...")
        running = False
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            # Look for .json files that have status "queued"
            job_files = sorted([f for f in os.listdir(JOBS_DIR) if f.endswith(".json")])
            for jf in job_files:
                job_path = os.path.join(JOBS_DIR, jf)
                try:
                    with open(job_path, "r") as f:
                        job_data = json.load(f)
                        
                    if job_data.get("status") == "queued":
                        job_id = job_data.get("job_id")
                        await process_job(job_id, job_data)
                except (json.JSONDecodeError, IOError):
                    continue
                except Exception as e:
                    logger.error("Error checking job file %s: %s", jf, str(e))
                    
        except Exception as e:
            logger.error("Error in worker loop: %s", str(e))
            
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    import asyncio
    asyncio.run(worker_loop())
