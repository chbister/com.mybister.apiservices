from __future__ import annotations

import os
import socket
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from app.transcript import build_transcript

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "Debug ASR Result Processor/Aggregator"
# --- GlitchTip (Sentry-kompat): Errors + Traces -----------------------------
# Nur aktiv, wenn SENTRY_DSN gesetzt ist (siehe .env.example / docker-compose).
# Ohne DSN bzw. ohne installiertes sentry-sdk verhält sich der Service wie bisher.
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

if sentry_sdk is not None and (_sentry_dsn := os.getenv("SENTRY_DSN")):
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "dev"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
        send_default_pii=False,
    )
    sentry_sdk.set_tag("service", "processor-service")
    if os.getenv("SENTRY_SEND_TEST_EVENT"):
        sentry_sdk.capture_message("processor-service: GlitchTip test event")

app = FastAPI(title=APP_TITLE)

# Internal service URLs (using network aliases)
SENTIMENT_URL = os.getenv("SENTIMENT_URL", "http://sentiment:8000/v1/sentiment")
EMOTION_BASE_URL = os.getenv("EMOTION_BASE_URL", "http://emotion:8000/v1/emotion")
EMOTION_NUANCE_URL = os.getenv("EMOTION_NUANCE_URL", "http://emotion-nuance:8000/v1/emotion")
EMOTION_VIBE_URL = os.getenv("EMOTION_VIBE_URL", "http://emotion-vibe:8000/v1/emotion")
EMOTION_STATE_URL = os.getenv("EMOTION_STATE_URL", "http://emotion-state:8000/v1/emotion")
NER_URL = os.getenv("NER_URL", "http://ner:8000/v1/ner")

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
            "emotion_state": None,
            "ner": None
        }
    
    # Define all parallel tasks
    tasks = [
        client.post(SENTIMENT_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_BASE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_NUANCE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_VIBE_URL, json={"text": text}, timeout=10.0),
        client.post(EMOTION_STATE_URL, json={"text": text}, timeout=10.0),
        client.post(NER_URL, json={"text": text}, timeout=10.0),
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    keys = ["sentiment", "emotion_base", "emotion_nuance", "emotion_vibe", "emotion_state", "ner"]
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
        "service": "debug-processor",
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
        keys = ["sentiment", "emotion_base", "emotion_nuance", "emotion_vibe", "emotion_state", "ner"]
        
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


# ===========================================================================
# /v1/transcript - neue Route: fusioniert ASR-Segmente mit Diarization-Turns
# zu einem Sprecher-Protokoll. Rein JSON-basiert, keine neuen Abhaengigkeiten.
# Der bestehende /v1/process-Endpunkt bleibt unveraendert.
# ===========================================================================


class TranscriptSegmentIn(BaseModel):
    start: float
    end: float
    text: str = ""


class TranscriptTurnIn(BaseModel):
    start: float
    end: float
    speaker: str


class TranscriptRequest(BaseModel):
    # Variante A: komplette Antwort-Objekte der beiden anderen Services
    asr_result: Optional[Dict[str, Any]] = None        # Antwort von POST /v1/asr
    diarization: Optional[Dict[str, Any]] = None       # Antwort von POST /v1/diarize
    # Variante B: flache Listen
    segments: Optional[List[TranscriptSegmentIn]] = None
    turns: Optional[List[TranscriptTurnIn]] = None
    # Optionen
    speaker_names: Optional[Dict[str, str]] = None     # {"SPEAKER_00": "Anna"}
    format: str = "json"                               # "json" | "text"


def _extract_segments(req: TranscriptRequest) -> List[Dict[str, Any]]:
    if req.segments is not None:
        return [s.model_dump() for s in req.segments]
    if req.asr_result is not None:
        raw = req.asr_result.get("segments")
        if raw is None:
            raise HTTPException(
                status_code=400,
                detail="asr_result muss ein 'segments'-Array enthalten.",
            )
        try:
            return [
                {"start": float(s["start"]), "end": float(s["end"]),
                 "text": str(s.get("text") or "")}
                for s in raw
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Ungueltiges asr_result.segments: {e}"
            )
    raise HTTPException(
        status_code=400,
        detail="Entweder 'segments' oder 'asr_result' muss uebergeben werden.",
    )


def _extract_turns(req: TranscriptRequest) -> List[Dict[str, Any]]:
    if req.turns is not None:
        return [t.model_dump() for t in req.turns]
    if req.diarization is not None:
        raw = req.diarization.get("speakers")
        if raw is None:
            raise HTTPException(
                status_code=400,
                detail="diarization muss ein 'speakers'-Array enthalten.",
            )
        try:
            return [
                {"start": float(t["start"]), "end": float(t["end"]),
                 "speaker": str(t["speaker"])}
                for t in raw
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Ungueltiges diarization.speakers: {e}"
            )
    return []  # ohne Diarization: Sprecher bleiben None, Protokoll ohne Labels


@app.post("/v1/transcript")
async def create_transcript(req: TranscriptRequest):
    logger.info(
        "Transcript request (format=%s, speaker_names=%s)",
        req.format, bool(req.speaker_names),
    )

    segments = _extract_segments(req)
    turns = _extract_turns(req)

    language = None
    if req.asr_result is not None:
        language = req.asr_result.get("language")

    result = build_transcript(
        segments, turns, speaker_names=req.speaker_names, language=language
    )
    result["metadata"] = {
        "processor_hostname": socket.gethostname(),
        "processed_at": datetime.utcnow().isoformat(),
        "diarization_provided": bool(turns),
    }

    if (req.format or "json").lower() == "text":
        return PlainTextResponse(
            content=result["transcript"],
            media_type="text/plain; charset=utf-8",
        )

    return result
