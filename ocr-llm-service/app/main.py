from __future__ import annotations

import os
import socket
import logging
import io
import json
from datetime import datetime
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from PIL import Image
try:
    import pytesseract
except ImportError:
    pytesseract = None
import httpx
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "OCR/LLM Contact Extraction Service"
app = FastAPI(title=APP_TITLE)

# Use a lightweight model for extraction. 
# For names, a NER model or a small T5/BART could work.
# Let's use a zero-shot-classification or a small QA model as a starting point,
# or better yet, a dedicated NER model if we only want names.
# For more general "modes", a small generative model or QA model is more flexible.
# We'll use a small NER model for 'name' mode and keep it extensible.
# NER_MODEL_ID = os.getenv("NER_MODEL_ID", "dbmdz/bert-large-cased-finetuned-conll03-english")
NER_MODEL_ID = os.getenv("NER_MODEL_ID", "Davlan/xlm-roberta-base-ner-hrl")
ner_pipeline = pipeline("ner", model=NER_MODEL_ID, aggregation_strategy="simple")

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
            except Exception as e:
                logger.error("Download failed: %s", str(e))
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

class OCRResult(BaseModel):
    data: str
    type: str

class RawOCR(BaseModel):
    text: str

class OCRResponse(BaseModel):
    results: List[OCRResult]
    raw_ocr: RawOCR
    count: int
    hostname: str
    file_name: Optional[str] = None
    file_url: Optional[str] = None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ocr-llm",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

def extract_names(text: str) -> List[str]:
    # Use NER to find person names
    ner_results = ner_pipeline(text)
    names = []
    for entity in ner_results:
        if entity['entity_group'] == 'PER':
            name = entity['word'].strip()
            if len(name) > 1 and name not in names:
                names.append(name)
    return names

@app.post("/v1/ocr-llm", response_model=OCRResponse)
async def process_image(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    mode: str = Form("name")
):
    logger.info("OCR-LLM request received file=%s, mode=%s", 
                file.filename if file else "None", mode)
    
    # Handle possible JSON-encoded mode (e.g. {"mode": "name"}) 
    # or combined modes (e.g. {"mode": ["name", "phonenumber"]})
    try:
        mode_data = json.loads(mode)
        if isinstance(mode_data, dict) and "mode" in mode_data:
            requested_modes = mode_data["mode"]
        else:
            requested_modes = mode_data
    except json.JSONDecodeError:
        requested_modes = mode

    if isinstance(requested_modes, str):
        requested_modes = [requested_modes]

    # Validate modes
    supported_modes = ["name"]
    for m in requested_modes:
        if m not in supported_modes:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {m}. Currently supported: {supported_modes}")

    try:
        contents = await get_image_bytes(file, file_url)
        image = Image.open(io.BytesIO(contents))
        
        # OCR Step
        if pytesseract is None:
             raise RuntimeError("pytesseract is not installed")
        extracted_text = pytesseract.image_to_string(image)
        logger.info("OCR completed. Extracted text length: %d", len(extracted_text))
        
        results = []
        
        # LLM/NER Extraction Step
        if "name" in requested_modes:
            names = extract_names(extracted_text)
            for name in names:
                results.append(OCRResult(data=name, type="NAME"))
        
        return OCRResponse(
            results=results,
            raw_ocr=RawOCR(text=extracted_text),
            count=len(results),
            hostname=socket.gethostname(),
            file_name=file.filename if file else None,
            file_url=file_url
        )
        
    except Exception as e:
        logger.error("OCR-LLM processing failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
