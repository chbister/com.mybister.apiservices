from __future__ import annotations

import os
import socket
import logging
import time
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, Dict, Any, Optional

import ulid
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "NER Service (CPU)"
MODEL_ID = os.getenv("NER_MODEL_ID", "Davlan/xlm-roberta-large-ner-hrl")
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")
JOBS_DIR = os.getenv("JOBS_DIR", "/tmp/ner_jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# Using a single-process executor for the CPU-bound task
executor = ProcessPoolExecutor(max_workers=1)

app = FastAPI(title=APP_TITLE)

# device=-1 => CPU
# aggregation_strategy="simple" or "first", "max", "average" 
# to group B-LOC, I-LOC into one entity
ner_pipe = None

def get_ner_pipe():
    global ner_pipe
    if ner_pipe is None:
        logger.info(f"Loading NER model: {MODEL_ID}")
        ner_pipe = pipeline("ner", model=MODEL_ID, device=-1, aggregation_strategy="simple")
    return ner_pipe

# Warm up the model on startup
@app.on_event("startup")
async def startup_event():
    get_ner_pipe()

class NerRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    truncation: bool = True
    max_length: int = 512
    callback_url: Optional[str] = None
    force_async: bool = False

class Entity(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int

class NerMetadata(BaseModel):
    model: str
    hostname: str
    processing_time: float

class NerResponse(BaseModel):
    entities: List[Entity]
    metadata: NerMetadata
    callback_url: Optional[str] = None

def ner_worker(text: str) -> Dict[str, Any]:
    """Function to be executed in the ProcessPoolExecutor."""
    start_time = time.time()
    try:
        # Re-initialize the pipeline in the child process if needed
        # (Though with max_workers=1 it might be pre-loaded if we use fork, 
        # but spawn is safer for CUDA/ML libs)
        pipe = get_ner_pipe()
        out = pipe(text)
        
        entities = []
        for e in out:
            entities.append({
                "entity_group": e["entity_group"],
                "score": float(e["score"]),
                "word": e["word"],
                "start": e["start"],
                "end": e["end"]
            })
            
        processing_time = time.time() - start_time
        return {
            "success": True,
            "result": {
                "entities": entities,
                "metadata": {
                    "model": MODEL_ID,
                    "hostname": socket.gethostname(),
                    "processing_time": round(processing_time, 4)
                }
            }
        }
    except Exception as e:
        logger.error(f"NER worker failed: {e}")
        return {"success": False, "error": str(e)}

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

async def run_async_job(job_id: str, text: str, callback: Optional[str]):
    loop = asyncio.get_running_loop()
    try:
        # Offload to the separate process
        future = executor.submit(ner_worker, text)
        response = await loop.run_in_executor(None, future.result)

        if response["success"]:
            save_job_result(job_id, response["result"], "completed")
            if callback:
                headers = {}
                if CALLBACK_AUTH_SECRET:
                    headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                    logger.info("Sending callback with Authorization header")

                logger.info("Sending callback to %s", callback)
                async with httpx.AsyncClient() as client:
                    try:
                        cb_response = await client.post(
                            callback, 
                            json={"job_id": job_id, "status": "completed", "result": response["result"]},
                            headers=headers,
                            timeout=30.0
                        )
                        cb_response.raise_for_status()
                        logger.info("Callback successful: %s", cb_response.status_code)
                    except Exception as e:
                        logger.error("Callback failed for job %s: %s", job_id, e)
        else:
            save_job_result(job_id, {"error": response["error"]}, "failed")
            if callback:
                headers = {}
                if CALLBACK_AUTH_SECRET:
                    headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"

                async with httpx.AsyncClient() as client:
                    await client.post(
                        callback, 
                        json={"job_id": job_id, "status": "failed", "error": response["error"]},
                        headers=headers,
                        timeout=30.0
                    )
            
    except Exception as e:
        logger.error("Async job wrapper failed for %s: %s", job_id, e)
        save_job_result(job_id, {"error": str(e)}, "failed")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ner",
        "model": MODEL_ID,
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.post("/v1/ner")
async def ner(req: NerRequest):
    if req.force_async or req.callback_url:
        job_id = f"{ulid.ULID()}-{socket.gethostname()}"
        logger.info("Switching to async mode for job %s", job_id)
        
        save_job_result(job_id, {}, "processing")
        
        asyncio.create_task(run_async_job(job_id, req.text, req.callback_url))
        
        content = {
            "job_id": job_id,
            "status_url": f"/v1/ner/status/{job_id}",
            "download_url": f"/v1/ner/download/{job_id}",
            "message": "Job accepted. Processing in background."
        }
        if req.callback_url:
            content["callback_url"] = req.callback_url
            
        return JSONResponse(status_code=202, content=content)

    # Sync path
    loop = asyncio.get_running_loop()
    future = executor.submit(ner_worker, req.text)
    response = await loop.run_in_executor(None, future.result)
    
    if not response["success"]:
        raise HTTPException(status_code=500, detail=response["error"])
        
    result = response["result"]
    if req.callback_url:
        result["callback_url"] = req.callback_url
        
    return result

@app.get("/v1/ner/status/{job_id}")
async def get_status(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "updated_at": data["updated_at"]
    }

@app.get("/v1/ner/download/{job_id}")
async def download_result(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {data['status']}")
    return data["result"]
