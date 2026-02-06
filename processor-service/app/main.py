from __future__ import annotations

import os
import socket
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "ASR Result Processor/Aggregator"
app = FastAPI(title=APP_TITLE)

# Internal service URLs (using network aliases)
SENTIMENT_URL = os.getenv("SENTIMENT_URL", "http://sentiment:8000/v1/sentiment")
EMOTION_BASE_URL = os.getenv("EMOTION_BASE_URL", "http://emotion:8000/v1/emotion")
EMOTION_NUANCE_URL = os.getenv("EMOTION_NUANCE_URL", "http://emotion-nuance:8000/v1/emotion")
EMOTION_VIBE_URL = os.getenv("EMOTION_VIBE_URL", "http://emotion-vibe:8000/v1/emotion")
EMOTION_STATE_URL = os.getenv("EMOTION_STATE_URL", "http://emotion-state:8000/v1/emotion")

class SegmentIn(BaseModel):
    start: float
    end: float
    text: str

class AsrMetadata(BaseModel):
    model_size: str
    compute_type: str
    device: str
    processing_time: float
    hostname: str

class AsrResultIn(BaseModel):
    text: str
    language: Optional[str] = None
    segments: List[SegmentIn]
    metadata: AsrMetadata

async def analyze_text(client: httpx.AsyncClient, text: str) -> Dict[str, Any]:
    """Helper to call sentiment and all emotion variants in parallel for a piece of text."""
    if not text.strip():
        return {
            "sentiment": None,
            "emotion_base": None,
            "emotion_nuance": None,
            "emotion_vibe": None,
            "emotion_state": None
        }
    
    # Define all parallel tasks
    tasks = [
        client.post(SENTIMENT_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_BASE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_NUANCE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_VIBE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_STATE_URL, json={"text": text}, timeout=10.0),
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    keys = ["sentiment", "emotion_base", "emotion_nuance", "emotion_vibe", "emotion_state"]
    output = {}
    
    for i, key in enumerate(keys):
        res = results[i]
        output[key] = None
        
        if isinstance(res, Exception):
            logger.error(f"{key} call failed for text '{text[:50]}...': {str(res)}")
        elif res.status_code != 200:
            logger.error(f"{key} call returned status {res.status_code} for text '{text[:50]}...': {res.text}")
        else:
            output[key] = res.json()
            
    return output

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "processor",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.post("/v1/process")
async def process_asr_result(asr_result: AsrResultIn):
    logger.info("Processing ASR result (len=%d, segments=%d)", len(asr_result.text), len(asr_result.segments))
    
    async with httpx.AsyncClient() as client:
        # 1. Analyze segments in batches (parallel)
        # This is the most important part and safe from the 20k char limit
        segment_tasks = [analyze_text(client, s.text) for s in asr_result.segments]
        segment_analyses = await asyncio.gather(*segment_tasks)
        
        # Simple aggregation logic
        # We aggregate all valid analysis types
        aggregated_analysis = {}
        keys = ["sentiment", "emotion_base", "emotion_nuance", "emotion_vibe", "emotion_state"]
        
        for key in keys:
            valid_results = [s[key] for s in segment_analyses if s.get(key)]
            if valid_results:
                aggregated_analysis[key] = {
                    "label": "aggregated", 
                    "detail": f"based on {len(valid_results)} segments"
                }
            else:
                aggregated_analysis[key] = None

        # 3. Rebuild enriched structure
        enriched_segments = []
        for i, seg in enumerate(asr_result.segments):
            enriched_segments.append({
                **seg.dict(),
                "analysis": segment_analyses[i]
            })
            
        return {
            "text": asr_result.text,
            "language": asr_result.language,
            "analysis": {
                **aggregated_analysis,
                "note": "Global analysis is aggregated from segments to avoid character limits."
            },
            "segments": enriched_segments,
            "asr_metadata": asr_result.metadata,
            "processor_hostname": socket.gethostname(),
            "processed_at": datetime.utcnow().isoformat()
        }
