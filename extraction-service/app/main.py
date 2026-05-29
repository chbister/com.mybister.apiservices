import os
import socket
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .extractors.base import BaseExtractor, ExtractionResult
from .extractors.name import NameExtractor
from .extractors.phone import PhoneExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Extraction Service")

# Registry of available extractors
STRATEGIES: Dict[str, BaseExtractor] = {
    "name": NameExtractor(),
    "phonenumber": PhoneExtractor(),
}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Extraction Service...")
    for name, strategy in STRATEGIES.items():
        logger.info(f"Preloading strategy: {name}")
        strategy.preload()
    logger.info("All strategies preloaded.")

class ExtractionRequest(BaseModel):
    text: str
    lines: List[str] = []
    mode: List[str]

class ExtractionMetadata(BaseModel):
    processing_time: float

class ExtractionResponse(BaseModel):
    results: List[ExtractionResult]
    count: int
    hostname: str
    metadata: ExtractionMetadata

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "extraction",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
        "config": {
            "supported_modes": list(STRATEGIES.keys())
        }
    }

@app.post("/v1/extract", response_model=ExtractionResponse)
async def extract(req: ExtractionRequest):
    start_time = time.time()
    all_results = []
    
    # Validation of modes
    unknown_modes = [m for m in req.mode if m not in STRATEGIES]
    if unknown_modes:
        raise HTTPException(status_code=400, detail=f"Unknown extraction modes: {unknown_modes}")

    # Execute requested strategies
    for mode in req.mode:
        strategy = STRATEGIES[mode]
        try:
            results = await strategy.extract(req.text, req.lines)
            all_results.extend(results)
        except Exception as e:
            logger.error(f"Error in strategy {mode}: {e}")
            # We might want to continue with other strategies or fail
            # For now, let's just log and continue
    
    processing_time = time.time() - start_time
    
    return ExtractionResponse(
        results=all_results,
        count=len(all_results),
        hostname=socket.gethostname(),
        metadata=ExtractionMetadata(
            processing_time=round(processing_time, 4)
        )
    )
