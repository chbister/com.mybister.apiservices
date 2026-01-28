from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

APP_TITLE = "Sentiment Service (CPU)"
MODEL_ID = os.getenv("SENTIMENT_MODEL_ID", "cardiffnlp/twitter-xlm-roberta-base-sentiment")

app = FastAPI(title=APP_TITLE)

# Lädt Modell einmal beim Start.
# device=-1 => CPU
clf = pipeline("text-classification", model=MODEL_ID, device=-1, top_k=None)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    truncation: bool = True
    max_length: int = 512  # für transformer input
    return_all_scores: bool = False


class SentimentResponse(BaseModel):
    label: str
    score: float
    all_scores: list[dict] | None = None

@app.get("/health")
def health():
    return {"status": "ok", "service": "sentiment"}

@app.post("/v1/sentiment", response_model=SentimentResponse)
def sentiment(req: SentimentRequest):
    try:
        # pipeline liefert je nach top_k/return_all_scores etwas unterschiedlich,
        # wir normalisieren auf: best label + score
        out = clf(
            req.text,
            truncation=req.truncation,
            max_length=req.max_length,
        )

        # out kann sein:
        # - list[dict] für single best
        # - list[list[dict]] wenn top_k/all scores
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            best = out[0]
            return SentimentResponse(label=best["label"], score=float(best["score"]))
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
            scores = out[0]
            best = max(scores, key=lambda x: x["score"])
            return SentimentResponse(
                label=best["label"],
                score=float(best["score"]),
                all_scores=scores if req.return_all_scores else None,
            )

        raise RuntimeError(f"Unexpected pipeline output: {type(out)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment failed: {e}")
