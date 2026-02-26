from __future__ import annotations

import os
import socket
import time
import tempfile
import logging
import json
import asyncio
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speaker Check Service (CPU)")

DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
PYANNOTE_TOKEN = os.getenv("PYANNOTE_TOKEN") or os.getenv("HF_TOKEN")

MIN_SPEAKER_SEC = float(os.getenv("DIARIZATION_MIN_SPEAKER_SEC", "10"))
MIN_SPEAKER_SHARE = float(os.getenv("DIARIZATION_MIN_SPEAKER_SHARE", "0.15"))

MAX_ANALYZE_SEC = float(os.getenv("DIARIZATION_MAX_ANALYZE_SEC", "0"))  # 0 = whole file

# Global cached pipeline per process
_PIPELINE: Optional[Pipeline] = None
_PIPELINE_LOCAL_DIR: Optional[str] = None
_PIPELINE_ERROR: Optional[str] = None

# Async jobs and callbacks
JOBS_DIR = os.getenv("DIAR_JOBS_DIR", "/srv/jobs")
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")

os.makedirs(JOBS_DIR, exist_ok=True)
# Separate process pool for heavy diarization work
executor = ProcessPoolExecutor(max_workers=1)


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


async def run_async_job(job_id: str, file_path: str, callback: Optional[str], file_name: Optional[str] = None, file_url: Optional[str] = None):
    loop = asyncio.get_running_loop()
    try:
        # Offload to separate process; worker will ensure pipeline is loaded
        future = executor.submit(speaker_check_worker, file_path, file_name, file_url)
        response = await loop.run_in_executor(None, future.result)

        if response.get("success"):
            result = response["result"]
            save_job_result(job_id, result, "completed")
            if callback:
                try:
                    headers = {}
                    if CALLBACK_AUTH_SECRET:
                        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                        logger.info("Sending callback with Authorization header")

                    logger.info("Sending callback to %s", callback)
                    logger.debug("Callback payload: %s", result)

                    async with httpx.AsyncClient() as client:
                        cb_response = None
                        try:
                            cb_response = await client.post(
                                callback,
                                json={"job_id": job_id, "status": "completed", "result": result},
                                headers=headers,
                            )
                            cb_response.raise_for_status()
                            logger.info("Callback response status: %s", cb_response.status_code)
                            logger.debug("Callback response body: %s", cb_response.text)
                        except httpx.HTTPStatusError as e:
                            logger.error("Callback failed for job %s with status %s: %s", job_id, e.response.status_code, e.response.text)
                        except Exception as e:
                            if cb_response:
                                logger.error("Callback failed for job %s: %s, Response: %s %s", job_id, e, cb_response.status_code, cb_response.text)
                            else:
                                logger.error("Callback failed for job %s: %s", job_id, e)
                except Exception as e:
                    logger.error("Callback logic failed for job %s: %s", job_id, e)
        else:
            save_job_result(job_id, {"error": response.get("error")}, "failed")
    except Exception as e:
        logger.error("Async job wrapper failed for %s: %s", job_id, e)
        save_job_result(job_id, {"error": str(e)}, "failed")
    finally:
        # Ensure temp file is removed after async processing
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


def normalize_to_wav_16k_mono(input_path: str) -> str:
    audio = AudioSegment.from_file(input_path)

    if MAX_ANALYZE_SEC and MAX_ANALYZE_SEC > 0:
        audio = audio[: int(MAX_ANALYZE_SEC * 1000)]

    audio = audio.set_channels(1).set_frame_rate(16000)

    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio.export(out_path, format="wav")
    return out_path


def load_pipeline() -> None:
    global _PIPELINE, _PIPELINE_LOCAL_DIR, _PIPELINE_ERROR

    if _PIPELINE is not None:
        return

    if not PYANNOTE_TOKEN:
        _PIPELINE_ERROR = "Missing PYANNOTE_TOKEN/HF_TOKEN."
        return

    try:
        logger.info("Diarization model: %s", DIARIZATION_MODEL)
        logger.info("Downloading model snapshot (if not cached yet)...")

        # Cache füllen (verhindert 'cannot find requested files')
        _PIPELINE_LOCAL_DIR = snapshot_download(
            repo_id=DIARIZATION_MODEL,
            token=PYANNOTE_TOKEN,
            max_workers=1,
        )

        logger.info("Snapshot ready at: %s", _PIPELINE_LOCAL_DIR)
        logger.info("Loading pyannote Pipeline from Hub (will reuse local cache)...")

        # WICHTIG: pyannote erwartet repo_id, kein lokaler Pfad
        # Token-Parameter ist versionsabhängig -> robust behandeln
        try:
            _PIPELINE = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=PYANNOTE_TOKEN)
        except TypeError:
            # Falls diese pyannote-Version kein use_auth_token kennt:
            _PIPELINE = Pipeline.from_pretrained(DIARIZATION_MODEL)

        _PIPELINE_ERROR = None
        logger.info("pyannote Pipeline loaded successfully.")

    except Exception as e:
        _PIPELINE = None
        _PIPELINE_ERROR = str(e)
        logger.exception("Pipeline load failed: %s", e)


def speaker_check_worker(file_path: str, file_name: Optional[str] = None, file_url: Optional[str] = None) -> Dict[str, Any]:
    start = time.time()
    wav_path = None

    try:
        if _PIPELINE is None:
            # Sollte in der Regel durch Startup vorgewärmt sein.
            # Falls nicht, versuchen wir einmal zu laden.
            load_pipeline()

        if _PIPELINE is None:
            raise RuntimeError(_PIPELINE_ERROR or "Pipeline not loaded.")

        wav_path = normalize_to_wav_16k_mono(file_path)

        annotation = _PIPELINE(wav_path)

        speaker_durations: Dict[str, float] = {}
        total_speech = 0.0

        for segment, _, label in annotation.itertracks(yield_label=True):
            dur = max(0.0, float(segment.end) - float(segment.start))
            speaker_durations[label] = speaker_durations.get(label, 0.0) + dur
            total_speech += dur

        qualifying = []
        for spk, dur in speaker_durations.items():
            share = (dur / total_speech) if total_speech > 0 else 0.0
            if dur >= MIN_SPEAKER_SEC or share >= MIN_SPEAKER_SHARE:
                qualifying.append(spk)

        number_speakers = len(qualifying)
        multi_speaker = number_speakers >= 2

        processing_time = time.time() - start

        metadata = {
            "diarization_model": DIARIZATION_MODEL,
            "snapshot_dir": _PIPELINE_LOCAL_DIR,
            "min_speaker_sec": MIN_SPEAKER_SEC,
            "min_speaker_share": MIN_SPEAKER_SHARE,
            "max_analyze_sec": MAX_ANALYZE_SEC,
            "processing_time": round(processing_time, 3),
            "hostname": socket.gethostname(),
            "datetime": datetime.utcnow().isoformat(),
        }
        if file_name:
            metadata["file_name"] = file_name
        if file_url:
            metadata["file_url"] = file_url

        return {
            "success": True,
            "result": {
                "multi_speaker": multi_speaker,
                "number_speakers": number_speakers,
                "metadata": metadata,
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.on_event("startup")
def startup():
    # Pipeline beim Start laden: verhindert 500 während Download/Cache-States
    load_pipeline()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "speaker_check",
        "diarization_model": DIARIZATION_MODEL,
        "has_token": bool(PYANNOTE_TOKEN),
        "pipeline_loaded": _PIPELINE is not None,
        "pipeline_error": _PIPELINE_ERROR,
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
    }


@app.get("/v1/speaker-check/status/{job_id}")
async def get_status(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "updated_at": data["updated_at"],
    }


@app.get("/v1/speaker-check/download/{job_id}")
async def download_result(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {data['status']}")
    return data["result"]


@app.post("/v1/speaker-check")
async def speaker_check(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    force_async: bool = Form(False),
):
    logger.info("Speaker-check request received file=%s file_url=%s callback_url=%s force_async=%s",
                file.filename if file else "None", file_url, callback_url, force_async)

    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

    if _PIPELINE is None:
        # Kein 500, sondern sauber 503 -> Client kann retryen
        raise HTTPException(status_code=503, detail=_PIPELINE_ERROR or "Pipeline not ready yet.")

    suffix = ".bin"
    if file and file.filename:
        _, ext = os.path.splitext(file.filename)
        suffix = ext or ".bin"
    elif file_url:
        from urllib.parse import urlparse
        ext = os.path.splitext(urlparse(file_url).path)[1]
        suffix = ext or ".bin"

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    remove_after = True

    try:
        if file:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(await file.read())
        else:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                async with client.stream("GET", file_url) as resp:
                    if resp.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download file: {resp.status_code}")
                    with os.fdopen(fd, "wb") as tmp:
                        async for chunk in resp.aiter_bytes():
                            tmp.write(chunk)

        # Decide path: async if forced or callback given
        if force_async or callback_url:
            job_id = f"{uuid.uuid4()}-{socket.gethostname()}"
            logger.info("Switching to async mode for job %s", job_id)
            save_job_result(job_id, {}, "processing")

            # Schedule background task
            asyncio.create_task(run_async_job(job_id, temp_path, callback_url, file.filename if file else None, file_url))
            remove_after = False  # async runner will remove

            content = {
                "job_id": job_id,
                "status_url": f"/v1/speaker-check/status/{job_id}",
                "download_url": f"/v1/speaker-check/download/{job_id}",
                "message": "Job accepted. Processing in background.",
            }
            if callback_url:
                content["callback_url"] = callback_url
            return JSONResponse(status_code=202, content=content)

        # Sync path
        try:
            audio = AudioSegment.from_file(temp_path)
            duration_sec = len(audio) / 1000.0
        except Exception:
            duration_sec = None

        # Offload sync processing to separate process as well
        loop = asyncio.get_running_loop()
        future = executor.submit(speaker_check_worker, temp_path, file.filename if file else None, file_url)
        response = await loop.run_in_executor(None, future.result)

        if not response.get("success"):
            raise HTTPException(status_code=500, detail=response.get("error"))

        if duration_sec is not None:
            response["result"]["metadata"]["input_duration_sec"] = round(duration_sec, 3)

        # Attach callback_url if provided (even in sync response, for traceability)
        result = response["result"]
        if callback_url:
            result["callback_url"] = callback_url
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Speaker-check endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if remove_after:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass