from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

APP_TITLE = "ASR Service (CPU)"
MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")  # tiny|base|small|medium|large-v3 (CPU large = langsam)
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")  # int8 empfohlen auf CPU
DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "auto")

app = FastAPI(title=APP_TITLE)

# Modell einmal beim Start laden (wichtig!)
# device="cpu" explizit; threads/num_workers kannst du später tunen.
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class AsrResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: list[SegmentOut]

@app.get("/health")
def health():
    return {"status": "ok", "service": "asr"}

@app.post("/v1/asr", response_model=AsrResponse)
async def asr(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),  # "auto" oder ISO-639-1 wie "de", "en", "es"
    timestamps: bool = Form(True),
    vad_filter: bool = Form(True),
):
    # Wir speichern Upload kurz in eine Datei; faster-whisper kann dann ffmpeg intern nutzen.
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp.write(content)
            tmp.flush()

            # language=None => auto; sonst "de"/"en"/...
            lang = None if (language or "").lower() == "auto" else language

            segments, info = model.transcribe(
                tmp.name,
                language=lang,
                vad_filter=vad_filter,
            )

            seg_list: list[SegmentOut] = []
            full_text_parts = []
            for s in segments:
                txt = (s.text or "").strip()
                if txt:
                    full_text_parts.append(txt)
                if timestamps:
                    seg_list.append(SegmentOut(start=float(s.start), end=float(s.end), text=txt))

            return AsrResponse(
                text=" ".join(full_text_parts).strip(),
                language=getattr(info, "language", None),
                segments=seg_list if timestamps else [],
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")
