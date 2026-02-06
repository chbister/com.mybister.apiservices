from __future__ import annotations

import os
import socket
import logging
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "Emotion Detection Service"
# Default model: DistilBERT fine-tuned for emotion detection
MODEL_ID = os.getenv("EMOTION_MODEL_ID", "bhadresh-savani/distilbert-base-uncased-emotion")

app = FastAPI(title=APP_TITLE)

# Load model once at startup
# device=-1 => CPU
classifier = pipeline("text-classification", model=MODEL_ID, device=-1, top_k=None)

class EmotionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    truncation: bool = True
    max_length: int = 512

class EmotionScore(BaseModel):
    label: str
    score: float

class EmotionResponse(BaseModel):
    emotions: List[EmotionScore]
    top_emotion: str
    hostname: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "emotion",
        "model": MODEL_ID,
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.post("/v1/emotion", response_model=EmotionResponse)
def detect_emotion(req: EmotionRequest):
    logger.info("Emotion detection request received")
    try:
        results = classifier(
            req.text,
            truncation=req.truncation,
            max_length=req.max_length,
        )

        # Results for top_k=None is a list of dicts: [{'label': 'joy', 'score': 0.9}, ...]
        if isinstance(results, list) and len(results) > 0:
            # Handle potential list of lists if batching was used (here we do single)
            data = results[0] if isinstance(results[0], list) else results
            
            # Sort by score descending
            sorted_emotions = sorted(data, key=lambda x: x['score'], reverse=True)
            
            return EmotionResponse(
                emotions=[EmotionScore(label=e['label'], score=e['score']) for e in sorted_emotions],
                top_emotion=sorted_emotions[0]['label'],
                hostname=socket.gethostname()
            )

        raise RuntimeError("Unexpected pipeline output")
        
    except Exception as e:
        logger.error("Emotion detection failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
