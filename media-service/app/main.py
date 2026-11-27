"""Media-Resolve-Service: Web-Medien-Link -> gehostete Audio-URL.

Nutzer uebergeben eine ihnen bekannte Seiten-URL (z.B. YouTube-Link). Der Service
laedt die beste Audiospur via yt-dlp, hostet sie mit TTL und gibt eine file_url
zurueck, die direkt an asr-/diarization- & co. weitergereicht werden kann
(fuer asr z.B. als file_url-Parameter von POST /v1/asr).

Endpoints:
    POST /v1/media/resolve          -> 202 {job_id, status_url, file_url}
    GET  /v1/media/status/{job_id}  -> Jobstatus (+ Ergebnis bzw. Fehler)
    GET  /v1/media/files/{job_id}   -> gehostete Audiodatei
    GET  /health
"""

import asyncio
import json
import logging
import mimetypes
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import ulid

# --- GlitchTip (Sentry-kompat): Errors + Traces -----------------------------
# Nur aktiv, wenn SENTRY_DSN gesetzt ist (siehe .env.example / docker-compose).
# Ohne DSN bzw. ohne installiertes sentry-sdk verhaelt sich der Service wie bisher.
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except ImportError:
    sentry_sdk = None  # type: ignore[assignment]

logging.basicConfig(
    level=os.getenv("UVICORN_LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

APP_TITLE = "Media Resolve Service"

JOBS_DIR = os.getenv("MEDIA_JOBS_DIR", "/srv/jobs")
FILES_DIR = os.getenv("MEDIA_FILES_DIR", "/srv/files")
PUBLIC_BASE_URL = os.getenv("MEDIA_PUBLIC_BASE_URL", "").rstrip("/")
TTL_HOURS_DEFAULT = int(os.getenv("MEDIA_TTL_HOURS", "24"))
TTL_HOURS_MAX = int(os.getenv("MEDIA_TTL_HOURS_MAX", "72"))
MAX_BYTES = int(os.getenv("MEDIA_MAX_BYTES", str(500 * 1024 * 1024)))
SWEEP_INTERVAL_SEC = int(os.getenv("MEDIA_SWEEP_INTERVAL_SEC", "600"))
DOWNLOAD_FORMAT = os.getenv("MEDIA_DOWNLOAD_FORMAT", "bestaudio[ext=m4a]/bestaudio")
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)

_sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_sdk is not None and _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "dev"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        integrations=[FastApiIntegration()],
        send_default_pii=False,
    )
    sentry_sdk.set_tag("service", "media-service")
    if os.getenv("SENTRY_SEND_TEST_EVENT"):
        sentry_sdk.capture_message("media-service: GlitchTip test event")

app = FastAPI(title=APP_TITLE)

# Downloads sind IO-gebunden; 2 Threads reichen (1 Request pro Nutzer ueblich).
executor = ThreadPoolExecutor(max_workers=2)

JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# mimetypes kennt in slim-Images oft keine Audio-Typen; feste Map als Fallback.
EXT_MIME_TYPES = {
    "m4a": "audio/mp4",
    "mp4": "video/mp4",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "webm": "audio/webm",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


class ResolveRequest(BaseModel):
    url: str
    ttl_hours: Optional[int] = None
    callback_url: Optional[str] = None


# --- Job-Speicherung (Muster analog asr-service) ----------------------------

def save_job_result(job_id: str, data: Dict[str, Any], status: str = "completed"):
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    job_data = {
        "job_id": job_id,
        "status": status,
        "updated_at": datetime.utcnow().isoformat(),
        "result": data,
    }
    with open(job_file, "w") as f:
        json.dump(job_data, f)


def get_job_data(job_id: str) -> Optional[Dict[str, Any]]:
    job_file = os.path.join(JOBS_DIR, f"{job_id}.json")
    if not os.path.exists(job_file):
        return None
    with open(job_file, "r") as f:
        return json.load(f)


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    return job_id


def find_media_file(job_id: str) -> Optional[str]:
    """Liefert den Pfad der gehosteten Mediendatei zu einem Job (falls vorhanden)."""
    for name in sorted(os.listdir(FILES_DIR)):
        if name.startswith(f"{job_id}."):
            path = os.path.join(FILES_DIR, name)
            if os.path.isfile(path):
                return path
    return None


def cleanup_job_files(job_id: str):
    path = find_media_file(job_id)
    if path:
        os.remove(path)


# --- Download (yt-dlp, in Worker-Thread) ------------------------------------

def download_media(job_id: str, url: str, ttl_hours: int, base: str) -> Dict[str, Any]:
    import yt_dlp

    outtmpl = os.path.join(FILES_DIR, f"{job_id}.%(ext)s")
    opts = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "socket_timeout": 30,
        "max_filesize": MAX_BYTES,
    }
    logger.info("Downloading media for job %s from %s", job_id, url)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)

    if not os.path.exists(file_path):
        # Fallback (z.B. bei Format-Remix): erste Datei zum Job suchen
        file_path = find_media_file(job_id) or ""
    if not file_path or not os.path.exists(file_path):
        raise RuntimeError("yt-dlp finished but produced no output file")

    ext = os.path.splitext(file_path)[1].lstrip(".") or "bin"
    expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
    return {
        "source_url": url,
        "file_url": f"{base}/v1/media/files/{job_id}",
        "title": info.get("title"),
        "duration": info.get("duration"),
        "format": ext,
        "mime_type": EXT_MIME_TYPES.get(ext) or mimetypes.guess_type(file_path)[0] or "application/octet-stream",
        "bytes": os.path.getsize(file_path),
        "expires_at": expires_at.isoformat(),
    }


# --- Async-Job + Callback ---------------------------------------------------

async def send_callback(callback_url: Optional[str], job_id: str, status: str, result: Dict[str, Any]):
    if not callback_url:
        return
    headers = {}
    if CALLBACK_AUTH_SECRET:
        headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
    if status == "completed":
        payload = {"job_id": job_id, "status": status, "result": result}
    else:
        payload = {"job_id": job_id, "status": status, "error": result.get("error")}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info("Callback response status: %s", response.status_code)
    except Exception as e:
        logger.error("Callback failed for job %s: %s", job_id, e)


async def run_resolve_job(job_id: str, url: str, ttl_hours: int, base: str, callback_url: Optional[str]):
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, download_media, job_id, url, ttl_hours, base)
        save_job_result(job_id, result, "completed")
        logger.info("Media resolve completed for job %s (%s bytes)", job_id, result.get("bytes"))
        await send_callback(callback_url, job_id, "completed", result)
    except Exception as e:
        logger.error("Media resolve failed for job %s: %s", job_id, e)
        if sentry_sdk is not None:
            sentry_sdk.capture_exception(e)
        save_job_result(job_id, {"error": str(e)}, "failed")
        cleanup_job_files(job_id)
        await send_callback(callback_url, job_id, "failed", {"error": str(e)})


# --- TTL-Sweeper ------------------------------------------------------------

def sweep_expired() -> int:
    """Loescht abgelaufene Mediendateien inkl. Job-JSON. Liefert Anzahl entfernter Jobs."""
    now = datetime.utcnow()
    removed = 0
    for name in os.listdir(JOBS_DIR):
        if not name.endswith(".json"):
            continue
        job_id = name[:-5]
        job_file = os.path.join(JOBS_DIR, name)
        try:
            with open(job_file, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        result = data.get("result") or {}
        expires_at = None
        if isinstance(result, dict) and result.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(result["expires_at"])
            except ValueError:
                expires_at = None
        if expires_at is None:
            # processing/failed: anhand updated_at + Default-TTL
            try:
                expires_at = datetime.fromisoformat(data["updated_at"]) + timedelta(hours=TTL_HOURS_DEFAULT)
            except Exception:
                continue

        if expires_at < now:
            cleanup_job_files(job_id)
            if os.path.exists(job_file):
                os.remove(job_file)
            removed += 1
            logger.info("TTL sweep removed job %s (expired %s)", job_id, expires_at.isoformat())
    return removed


@app.on_event("startup")
async def start_sweeper():
    async def _loop():
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(executor, sweep_expired)
            except Exception as e:
                logger.error("TTL sweep failed: %s", e)
            await asyncio.sleep(SWEEP_INTERVAL_SEC)
    asyncio.create_task(_loop())


# --- Endpunkte --------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "media",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
    }


@app.post("/v1/media/resolve")
async def resolve_media(req: ResolveRequest, request: Request):
    url = (req.url or "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url must be an http(s) URL")

    ttl_hours = min(max(req.ttl_hours or TTL_HOURS_DEFAULT, 1), TTL_HOURS_MAX)
    base = PUBLIC_BASE_URL or str(request.base_url).rstrip("/")

    job_id = f"{ulid.ULID()}-{socket.gethostname()}"
    logger.info("Media resolve job %s for %s (ttl=%sh)", job_id, url, ttl_hours)
    save_job_result(job_id, {"source_url": url}, "processing")

    asyncio.create_task(run_resolve_job(job_id, url, ttl_hours, base, req.callback_url))

    content = {
        "job_id": job_id,
        "status_url": f"/v1/media/status/{job_id}",
        "file_url": f"{base}/v1/media/files/{job_id}",
        "message": "Media download started in background.",
    }
    if req.callback_url:
        content["callback_url"] = req.callback_url
    return JSONResponse(status_code=202, content=content)


@app.get("/v1/media/status/{job_id}")
async def get_status(job_id: str):
    validate_job_id(job_id)
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    response = {
        "job_id": data["job_id"],
        "status": data["status"],
        "updated_at": data["updated_at"],
    }
    result = data.get("result")
    if isinstance(result, dict):
        if data["status"] == "failed" and result.get("error"):
            response["error"] = result["error"]
        elif data["status"] == "completed":
            response["result"] = result
    return response


@app.get("/v1/media/files/{job_id}")
async def get_file(job_id: str):
    validate_job_id(job_id)
    data = get_job_data(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {data['status']}")
    path = find_media_file(job_id)
    if not path:
        raise HTTPException(status_code=404, detail="File not found (maybe expired)")
    result = data.get("result") or {}
    return FileResponse(
        path,
        media_type=result.get("mime_type") or "application/octet-stream",
        filename=os.path.basename(path),
    )
