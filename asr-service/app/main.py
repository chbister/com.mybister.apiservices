from __future__ import annotations

import os
import socket
import time
import json
import logging
import tempfile
import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any
import ulid
import httpx
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "ASR Service (CPU)"
MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
DEFAULT_LANGUAGE = os.getenv("ASR_DEFAULT_LANGUAGE", "auto")
ASYNC_THRESHOLD_SEC = int(os.getenv("ASR_ASYNC_THRESHOLD", "300"))
JOBS_DIR = os.getenv("ASR_JOBS_DIR", "/srv/jobs")
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")

os.makedirs(JOBS_DIR, exist_ok=True)

# Global process pool executor for heavy CPU tasks.
# We use 1 worker for transcription. This process is SEPARATE from the API workers.
# This prevents the CPU-heavy transcription from starving the web server.
executor = ProcessPoolExecutor(max_workers=1)

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
    sentry_sdk.set_tag("service", "asr-service")
    if os.getenv("SENTRY_SEND_TEST_EVENT"):
        sentry_sdk.capture_message("asr-service: GlitchTip test event")

app = FastAPI(title=APP_TITLE)

def transcribe_worker(file_path: str, model_size: str, compute_type: str, language: Optional[str], vad_filter: bool, timestamps: bool, file_name: Optional[str] = None, file_url: Optional[str] = None):
    """
    This function runs in a completely separate OS process.
    """
    start_time = time.time()
    try:
        # Load model locally in the worker process (required as it's a new process)
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        
        segments, info = model.transcribe(
            file_path,
            language=language,
            vad_filter=vad_filter,
        )

        seg_list = []
        full_text_parts = []
        for s in segments:
            txt = (s.text or "").strip()
            if txt:
                full_text_parts.append(txt)
            if timestamps:
                seg_list.append({"start": float(s.start), "end": float(s.end), "text": txt})

        processing_time = time.time() - start_time
        
        metadata = {
            "model_size": model_size,
            "compute_type": compute_type,
            "device": "cpu",
            "processing_time": round(processing_time, 3),
            "hostname": socket.gethostname()
        }
        if file_name:
            metadata["file_name"] = file_name
        if file_url:
            metadata["file_url"] = file_url

        return {
            "success": True,
            "result": {
                "text": " ".join(full_text_parts).strip(),
                "language": getattr(info, "language", None),
                "segments": seg_list,
                "metadata": metadata
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ===========================================================================
# /v1/asr/gpu - neue Route: lokale GPU-Variante mit Apples MLX-Backend.
# ctranslate2/faster-whisper ist auf macOS CPU-only, deshalb nutzt dieser
# Endpunkt mlx-whisper (Metal). Nur lokaler Betrieb (Makefile: local-asr);
# in Docker fehlt mlx-whisper und der Endpunkt liefert eine klare Meldung.
# Der bestehende /v1/asr-Endpunkt bleibt unveraendert.
# ===========================================================================

MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

DEFAULT_GPU_MODEL_SIZE = os.getenv("ASR_GPU_MODEL_SIZE", "small")


def resolve_mlx_repo(model_size: str) -> str:
    """Mappt tiny|small|medium|large-v3 auf ein mlx-community-Repo.

    Vollstaendige Repo-Namen (mit '/') werden durchgereicht.
    """
    if model_size in MLX_MODEL_REPOS:
        return MLX_MODEL_REPOS[model_size]
    if "/" in model_size:
        return model_size
    raise ValueError(
        f"Unbekannte model_size '{model_size}'. "
        f"Erlaubt: {', '.join(MLX_MODEL_REPOS)} oder ein vollstaendiger "
        "HF-Repo-Name (z.B. mlx-community/whisper-small-mlx)."
    )


def load_audio_16k_mono_np(file_path: str):
    """Dekodiert Audio zu 16 kHz mono float32 (numpy) - ohne system-ffmpeg.

    Reihenfolge: soundfile (wav/flac/ogg/mp3) -> PyAV (m4a/aac/webm/...).
    """
    import numpy as np

    try:
        import soundfile as sf

        samples, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        if sample_rate != 16000:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(16000, int(sample_rate))
            samples = resample_poly(samples, 16000 // g, int(sample_rate) // g).astype(
                np.float32
            )
        return samples
    except Exception as e:
        logger.info("soundfile decode failed (%s), trying PyAV...", e)

    import av

    with av.open(file_path) as container:
        stream = next((s for s in container.streams.audio if s.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio stream found in file")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                chunks.append(out_frame.to_ndarray().astype(np.float32).reshape(-1))
        for out_frame in resampler.resample(None):
            chunks.append(out_frame.to_ndarray().astype(np.float32).reshape(-1))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def audio_duration_sec(file_path: str) -> Optional[float]:
    """Audiodauer via PyAV (funktioniert ohne system-ffmpeg)."""
    try:
        import av

        with av.open(file_path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is not None and stream.duration is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
    except Exception as e:
        logger.warning("Could not detect duration via PyAV: %s", e)
    return None


def transcribe_worker_gpu(
    file_path: str,
    model_size: str,
    compute_type: str,
    language: Optional[str],
    vad_filter: bool,
    timestamps: bool,
    file_name: Optional[str] = None,
    file_url: Optional[str] = None,
    word_timestamps: bool = False,
):
    """Transkription mit mlx-whisper auf der Apple-GPU (separater Prozess).

    Signatur analog zu transcribe_worker, damit run_async_job_gpu die
    ueblichen Job-/Callback-Routinen wiederverwenden kann. 'compute_type'
    und 'vad_filter' werden vom MLX-Backend nicht unterstuetzt (siehe metadata).
    """
    start_time = time.time()
    try:
        try:
            import mlx_whisper
        except ImportError:
            return {
                "success": False,
                "error": (
                    "mlx-whisper ist nicht installiert. Der GPU-Endpunkt "
                    "/v1/asr/gpu ist nur fuer den lokalen Betrieb gedacht "
                    "(pip install -r requirements-local.txt)."
                ),
            }

        repo = resolve_mlx_repo(model_size)
        audio = load_audio_16k_mono_np(file_path)
        if audio.shape[0] == 0:
            return {"success": False, "error": "Audio file contains no samples."}

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=repo,
            language=None if (language or "").lower() == "auto" else language,
            word_timestamps=bool(word_timestamps),
            verbose=None,
        )

        seg_list = []
        full_text_parts = []
        raw_segments = result.get("segments") or []
        for s in raw_segments:
            txt = (s.get("text") or "").strip()
            if txt:
                full_text_parts.append(txt)
            if timestamps:
                entry = {
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": txt,
                }
                if word_timestamps and s.get("words"):
                    entry["words"] = [
                        {
                            "word": str(w.get("word", "")),
                            "start": float(w.get("start", 0.0)),
                            "end": float(w.get("end", 0.0)),
                        }
                        for w in s["words"]
                    ]
                seg_list.append(entry)

        full_text = (result.get("text") or "").strip()
        if not full_text:
            full_text = " ".join(full_text_parts).strip()

        processing_time = time.time() - start_time
        metadata = {
            "model_size": model_size,
            "model_repo": repo,
            "backend": "mlx-whisper",
            "device": "apple-gpu",
            "compute_type": compute_type,
            "vad_filter": False,
            "vad_filter_note": "vad_filter wird vom MLX-Backend nicht unterstuetzt",
            "word_timestamps": bool(word_timestamps),
            "processing_time": round(processing_time, 3),
            "hostname": socket.gethostname(),
        }
        if file_name:
            metadata["file_name"] = file_name
        if file_url:
            metadata["file_url"] = file_url

        return {
            "success": True,
            "result": {
                "text": full_text,
                "language": result.get("language"),
                "segments": seg_list,
                "metadata": metadata,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_async_job_gpu(
    job_id: str,
    file_path: str,
    model_size: str,
    lang: Optional[str],
    ts: bool,
    callback: Optional[str],
    file_name: Optional[str] = None,
    file_url: Optional[str] = None,
    word_timestamps: bool = False,
):
    """Async-Runner fuer /v1/asr/gpu (gleiche Job-/Callback-Semantik)."""
    loop = asyncio.get_running_loop()
    try:
        future = executor.submit(
            transcribe_worker_gpu,
            file_path,
            model_size,
            COMPUTE_TYPE,
            lang,
            True,
            ts,
            file_name,
            file_url,
            word_timestamps,
        )
        response = await loop.run_in_executor(None, future.result)

        if response["success"]:
            save_job_result(job_id, response["result"], "completed")
            if callback:
                try:
                    headers = {}
                    if CALLBACK_AUTH_SECRET:
                        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                    async with httpx.AsyncClient() as client:
                        try:
                            cb_response = await client.post(
                                callback,
                                json={
                                    "job_id": job_id,
                                    "status": "completed",
                                    "result": response["result"],
                                },
                                headers=headers,
                            )
                            cb_response.raise_for_status()
                            logger.info(
                                "GPU callback response status: %s",
                                cb_response.status_code,
                            )
                        except Exception as e:
                            logger.error(
                                "GPU callback failed for job %s: %s", job_id, e
                            )
                except Exception as e:
                    logger.error("GPU callback logic failed for job %s: %s", job_id, e)
        else:
            save_job_result(job_id, {"error": response["error"]}, "failed")
            if callback:
                try:
                    headers = {}
                    if CALLBACK_AUTH_SECRET:
                        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            callback,
                            json={
                                "job_id": job_id,
                                "status": "failed",
                                "error": response["error"],
                            },
                            headers=headers,
                        )
                except Exception as e:
                    logger.error("GPU error callback failed for job %s: %s", job_id, e)

    except Exception as e:
        logger.error("GPU async job wrapper failed for %s: %s", job_id, e)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(e)

        save_job_result(job_id, {"error": str(e)}, "failed")
        if callback:
            try:
                headers = {}
                if CALLBACK_AUTH_SECRET:
                    headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                async with httpx.AsyncClient() as client:
                    await client.post(
                        callback,
                        json={"job_id": job_id, "status": "failed", "error": str(e)},
                        headers=headers,
                    )
            except Exception as ex:
                logger.error("GPU error callback failed for job %s: %s", job_id, ex)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/v1/asr/gpu")
async def asr_gpu(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    language: str = Form(DEFAULT_LANGUAGE),
    timestamps: bool = Form(True),
    vad_filter: bool = Form(True),
    word_timestamps: bool = Form(False),
    model_size: str = Form(DEFAULT_GPU_MODEL_SIZE),
    callback_url: Optional[str] = Form(None),
    force_async: bool = Form(False),
):
    logger.info(
        "GPU ASR request received file=%s file_url=%s model_size=%s word_timestamps=%s",
        file.filename if file else "None",
        file_url,
        model_size,
        word_timestamps,
    )

    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

    try:
        resolve_mlx_repo(model_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suffix = ".bin"
    if file:
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    elif file_url:
        from urllib.parse import urlparse

        path = urlparse(file_url).path
        suffix = os.path.splitext(path)[1] or ".bin"

    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)

    lang_val = None if (language or "").lower() == "auto" else language

    try:
        if file:
            with open(temp_path, "wb") as tmp:
                tmp.write(await file.read())
        else:
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                try:
                    check_res = await client.get(
                        file_url, headers={"Range": "bytes=0-0"}
                    )
                    check_res.raise_for_status()
                except Exception as e:
                    logger.error("File check failed for %s: %s", file_url, e)
                    status_code = 400
                    if hasattr(e, "response") and e.response is not None:
                        status_code = e.response.status_code
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(
                        status_code=status_code,
                        detail=f"File check failed at URL: {file_url} (Error: {e})",
                    )

            job_id = f"{ulid.ULID()}-{socket.gethostname()}"
            logger.info("GPU: switching to async mode for file_url job %s", job_id)
            save_job_result(job_id, {}, "processing")

            async def download_and_run():
                try:
                    async with httpx.AsyncClient(follow_redirects=True) as client:
                        async with client.stream("GET", file_url) as response:
                            if response.status_code != 200:
                                raise Exception(
                                    f"Failed to download file from URL: {response.status_code}"
                                )
                            with open(temp_path, "wb") as tmp:
                                async for chunk in response.aiter_bytes():
                                    tmp.write(chunk)

                    await run_async_job_gpu(
                        job_id, temp_path, model_size, lang_val, timestamps,
                        callback_url, None, file_url, word_timestamps,
                    )
                except Exception as e:
                    logger.error("GPU async download/job failed: %s", e)
                    if sentry_sdk is not None:
                        sentry_sdk.capture_exception(e)

                    save_job_result(job_id, {"error": str(e)}, "failed")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            asyncio.create_task(download_and_run())

            return JSONResponse(
                status_code=202,
                content={
                    "job_id": job_id,
                    "status_url": f"/v1/asr/status/{job_id}",
                    "download_url": f"/v1/asr/download/{job_id}",
                    "message": "File download and processing started in background.",
                },
            )

        duration_sec = audio_duration_sec(temp_path) or 0.0
        logger.info("GPU: audio duration detected: %.2fs", duration_sec)

        if duration_sec > ASYNC_THRESHOLD_SEC or force_async:
            job_id = f"{ulid.ULID()}-{socket.gethostname()}"
            logger.info("GPU: switching to async mode for job %s", job_id)
            save_job_result(job_id, {}, "processing")

            asyncio.create_task(
                run_async_job_gpu(
                    job_id, temp_path, model_size, lang_val, timestamps,
                    callback_url, file.filename if file else None, file_url,
                    word_timestamps,
                )
            )

            response_content = {
                "job_id": job_id,
                "status_url": f"/v1/asr/status/{job_id}",
                "download_url": f"/v1/asr/download/{job_id}",
                "message": "Job accepted. Processing in background.",
            }
            if callback_url:
                response_content["callback_url"] = callback_url

            return JSONResponse(status_code=202, content=response_content)

        # Sync path
        loop = asyncio.get_running_loop()
        future = executor.submit(
            transcribe_worker_gpu,
            temp_path,
            model_size,
            COMPUTE_TYPE,
            lang_val,
            vad_filter,
            timestamps,
            file.filename if file else None,
            file_url,
            word_timestamps,
        )
        response = await loop.run_in_executor(None, future.result)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])

        result = response["result"]
        if callback_url:
            result["callback_url"] = callback_url

        return result

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("GPU ASR endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


class AsrMetadata(BaseModel):
    model_size: str
    compute_type: str
    device: str
    processing_time: float
    hostname: str
    file_name: Optional[str] = None
    file_url: Optional[str] = None


class AsrResponse(BaseModel):
    text: str
    language: Optional[str] = None
    segments: list[SegmentOut]
    metadata: AsrMetadata


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "asr",
        "model": MODEL_SIZE,
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }


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


async def run_async_job(job_id: str, file_path: str, lang: Optional[str], vad: bool, ts: bool, callback: Optional[str], file_name: Optional[str] = None, file_url: Optional[str] = None):
    loop = asyncio.get_running_loop()
    try:
        # Offload to the separate process
        future = executor.submit(transcribe_worker, file_path, MODEL_SIZE, COMPUTE_TYPE, lang, vad, ts, file_name, file_url)
        # Use run_in_executor to avoid blocking the event loop while waiting for the process result
        response = await loop.run_in_executor(None, future.result)

        if response["success"]:
            save_job_result(job_id, response["result"], "completed")
            if callback:
                try:
                    headers = {}
                    if CALLBACK_AUTH_SECRET:
                        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                        logger.info("Sending callback with Authorization header")

                    logger.info("Sending callback to %s", callback)
                    logger.debug("Callback payload: %s", response)
                        
                    async with httpx.AsyncClient() as client:
                        cb_response = None
                        try:
                            cb_response = await client.post(
                                callback, 
                                json={"job_id": job_id, "status": "completed", "result": response["result"]},
                                headers=headers
                            )
                            cb_response.raise_for_status()
                            logger.info("Callback response status: %s", cb_response.status_code)
                            logger.debug("Callback response body: %s", cb_response.text)
                        except httpx.HTTPStatusError as e:
                            logger.error("HTTPStatusError, Callback failed for job %s with status %s: %s", job_id, e.response.status_code, e.response.text)
                        except Exception as e:
                            if cb_response:
                                logger.error("Exception A, Callback failed for job %s: %s, Response: %s %s", job_id, e, cb_response.status_code, cb_response.text)
                            else:
                                logger.error("Exception B, Callback failed for job %s: %s", job_id, e)
                except Exception as e:
                    logger.error("Exception C, Callback logic failed for job %s: %s", job_id, e)
        else:
            save_job_result(job_id, {"error": response["error"]}, "failed")
            if callback:
                try:
                    headers = {}
                    if CALLBACK_AUTH_SECRET:
                        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            callback,
                            json={"job_id": job_id, "status": "failed", "error": response["error"]},
                            headers=headers
                        )
                except Exception as e:
                    logger.error("Error callback failed for job %s: %s", job_id, e)
            
    except Exception as e:
        logger.error("Async job wrapper failed for %s: %s", job_id, e)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(e)

        save_job_result(job_id, {"error": str(e)}, "failed")
        if callback:
            try:
                headers = {}
                if CALLBACK_AUTH_SECRET:
                    headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
                async with httpx.AsyncClient() as client:
                    await client.post(
                        callback,
                        json={"job_id": job_id, "status": "failed", "error": str(e)},
                        headers=headers
                    )
            except Exception as ex:
                logger.error("Error callback failed for job %s: %s", job_id, ex)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/v1/asr")
async def asr(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    language: str = Form(DEFAULT_LANGUAGE),
    timestamps: bool = Form(True),
    vad_filter: bool = Form(True),
    callback_url: Optional[str] = Form(None),
    force_async: bool = Form(False)
):
    logger.info("ASR request received file=%s file_url=%s",
                file.filename if file else "None", 
                file_url)
    logger.info("ASR request received callback_url=%s force_async=%s", callback_url, force_async)
    
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")
    
    # Save to temp file
    suffix = ".bin"
    if file:
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    elif file_url:
        # Try to extract suffix from URL
        from urllib.parse import urlparse
        path = urlparse(file_url).path
        suffix = os.path.splitext(path)[1] or ".bin"

    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd) # Close it as we will open it with 'wb' or via os.fdopen
    
    lang_val = None if (language or "").lower() == "auto" else language

    try:
        if file:
            with open(temp_path, 'wb') as tmp:
                tmp.write(await file.read())
        else:
            # Download from URL
            logger.info("Checking if file exists at %s", file_url)
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                try:
                    check_res = await client.get(file_url, headers={"Range": "bytes=0-0"})
                    check_res.raise_for_status()
                except Exception as e:
                    logger.error("File check failed for %s: %s", file_url, e)
                    status_code = 400
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(status_code=status_code, detail=f"File check failed at URL: {file_url} (Error: {e})")

            # Switch to async for file_url to avoid timeout
            job_id = f"{ulid.ULID()}-{socket.gethostname()}"
            logger.info("Switching to async mode for file_url job %s", job_id)
            save_job_result(job_id, {}, "processing")
            
            # Use a wrapper that downloads and then runs the job
            async def download_and_run():
                try:
                    async with httpx.AsyncClient(follow_redirects=True) as client:
                        async with client.stream("GET", file_url) as response:
                            if response.status_code != 200:
                                raise Exception(f"Failed to download file from URL: {response.status_code}")
                            with open(temp_path, 'wb') as tmp:
                                async for chunk in response.aiter_bytes():
                                    tmp.write(chunk)
                    
                    await run_async_job(job_id, temp_path, lang_val, vad_filter, timestamps, callback_url, 
                                       None, file_url)
                except Exception as e:
                    logger.error("Async download/job failed: %s", e)
                    if sentry_sdk is not None:
                        sentry_sdk.capture_exception(e)

                    save_job_result(job_id, {"error": str(e)}, "failed")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            asyncio.create_task(download_and_run())
            
            return JSONResponse(status_code=202, content={
                "job_id": job_id,
                "status_url": f"/v1/asr/status/{job_id}",
                "download_url": f"/v1/asr/download/{job_id}",
                "message": "File download and processing started in background."
            })
        
        try:
            audio = AudioSegment.from_file(temp_path)
            duration_sec = len(audio) / 1000.0
            logger.info("Audio duration detected: %.2fs", duration_sec)
        except Exception as e:
            logger.warning("Could not detect duration: %s", e)
            duration_sec = 0

        if duration_sec > ASYNC_THRESHOLD_SEC or force_async:
            # Using ULID and hostname for the job_id
            job_id = f"{ulid.ULID()}-{socket.gethostname()}"
            logger.info("Switching to async mode for job %s", job_id)
            
            save_job_result(job_id, {}, "processing")
            
            # Fire and forget the background task
            asyncio.create_task(run_async_job(job_id, temp_path, lang_val, vad_filter, timestamps, callback_url, 
                                             file.filename if file else None, file_url))
            
            response_content = {
                "job_id": job_id,
                "status_url": f"/v1/asr/status/{job_id}",
                "download_url": f"/v1/asr/download/{job_id}",
                "message": "Job accepted. Processing in background."
            }
            if callback_url:
                response_content["callback_url"] = callback_url
                
            return JSONResponse(status_code=202, content=response_content)

        # Sync path: Also offload to the separate process to keep this worker's event loop free
        loop = asyncio.get_running_loop()
        future = executor.submit(transcribe_worker, temp_path, MODEL_SIZE, COMPUTE_TYPE, lang_val, vad_filter, timestamps,
                                 file.filename if file else None, file_url)
        response = await loop.run_in_executor(None, future.result)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])
            
        result = response["result"]
        if callback_url:
            result["callback_url"] = callback_url
            
        return result

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("ASR endpoint failed: %s, %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/asr/status/{job_id}")
async def get_status(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "updated_at": data["updated_at"]
    }


@app.get("/v1/asr/download/{job_id}")
async def download_result(job_id: str):
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {data['status']}")
    return data["result"]
