import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
import magic
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_NAME = "metadata-service"
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(2000 * 1024 * 1024)))
DOWNLOAD_TIMEOUT_S = float(os.getenv("DOWNLOAD_TIMEOUT_S", "30"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
CALLBACK_AUTH_SECRET = os.getenv("CALLBACK_AUTH_SECRET")

app = FastAPI(title=APP_NAME, version="1.0.0")

# Very simple in-memory job store (good enough behind a gateway for lightweight async).
# If you need persistence, swap this for Redis / DB.
JOBS: Dict[str, Dict[str, Any]] = {}


class ExtractRequest(BaseModel):
    file_url: HttpUrl
    callback_url: Optional[HttpUrl] = None
    force_async: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_iso_date(raw: Any) -> str:
    """
    Attempts to parse a date string and return it in ISO 8601 format.
    Exiftool often returns 'YYYY:MM:DD HH:MM:SS' or similar.
    """
    if not isinstance(raw, str):
        return str(raw) if raw is not None else ""

    raw = raw.strip()
    if not raw:
        return ""

    # Common Exif format: YYYY:MM:DD HH:MM:SS
    # Sometimes it has a timezone suffix: YYYY:MM:DD HH:MM:SS+01:00 or YYYY:MM:DD HH:MM:SSZ
    try:
        # Match YYYY:MM:DD HH:MM:SS...
        if re.match(r"^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", raw):
            # Replace the first two colons with dashes to make it more ISO-like
            # e.g. "2026:02:14 14:12:04" -> "2026-02-14 14:12:04"
            # and then handle the space with T for standard ISO
            iso_candidate = raw[:4] + "-" + raw[5:7] + "-" + raw[8:10] + "T" + raw[11:]
            # Now try to parse it
            return datetime.fromisoformat(iso_candidate).isoformat()
    except Exception:
        pass

    # Generic fallback: try fromisoformat directly
    try:
        return datetime.fromisoformat(raw).isoformat()
    except Exception:
        pass

    return raw


def _safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "file"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_size(path: str) -> int:
    return os.path.getsize(path)


def _detect_mime(path: str) -> str:
    # libmagic-based detection
    m = magic.Magic(mime=True)
    return m.from_file(path) or "application/octet-stream"


def _run_cmd_json(cmd: list, timeout: int = 60) -> Optional[dict]:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if p.returncode != 0:
            return None
        out = p.stdout.decode("utf-8", errors="replace").strip()
        if not out:
            return None
        return json.loads(out)
    except Exception:
        return None


def _exiftool_metadata(path: str) -> Dict[str, Any]:
    # -json gives structured output
    # -G includes group names (EXIF, XMP, File, QuickTime, PDF, etc.)
    # -n avoids human formatting for numeric values (GPS, etc.)
    cmd = ["exiftool", "-json", "-G", "-n", path]
    data = _run_cmd_json(cmd, timeout=60)
    if isinstance(data, list) and data:
        return data[0]
    return {}


def _ffprobe_metadata(path: str) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    data = _run_cmd_json(cmd, timeout=60)
    return data or {}


def _pdf_docinfo(path: str) -> Dict[str, Any]:
    # pypdf docinfo can be useful and consistent for PDFs
    try:
        r = PdfReader(path)
        info = r.metadata or {}
        out: Dict[str, Any] = {}
        for k, v in info.items():
            key = str(k).lstrip("/")  # "/Title" -> "Title"
            out[key] = str(v) if v is not None else None
        out["Pages"] = len(r.pages)
        return out
    except Exception:
        return {}


def _normalize_common(
    mime: str,
    exif: Dict[str, Any],
    ffprobe: Dict[str, Any],
    pdfinfo: Dict[str, Any],
    original_name: Optional[str] = None,
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a compact "normalized" section with the most commonly-used fields.
    We keep raw tool outputs too.
    """
    norm: Dict[str, Any] = {}

    # Timestamps (best-effort across groups)
    candidates = [
        "EXIF:DateTimeOriginal",
        "XMP:CreateDate",
        "QuickTime:CreateDate",
        "RIFF:DateCreated",
        "ID3:Date",
        "PDF:CreateDate",
        "File:FileModifyDate",
    ]
    for c in candidates:
        if c in exif:
            norm["created_at_raw"] = _to_iso_date(exif.get(c))
            break

    # Image dimensions
    for k in ["EXIF:ImageWidth", "File:ImageWidth", "PNG:ImageWidth"]:
        if k in exif:
            norm["width"] = exif.get(k)
            break
    for k in ["EXIF:ImageHeight", "File:ImageHeight", "PNG:ImageHeight"]:
        if k in exif:
            norm["height"] = exif.get(k)
            break

    # GPS
    lat = exif.get("EXIF:GPSLatitude") or exif.get("Composite:GPSLatitude")
    lon = exif.get("EXIF:GPSLongitude") or exif.get("Composite:GPSLongitude")
    if lat is not None and lon is not None:
        gps_info = {"lat": lat, "lon": lon}
        
        # Altitude
        alt = exif.get("EXIF:GPSAltitude") or exif.get("Composite:GPSAltitude")
        if alt is not None:
            gps_info["altitude"] = alt
            
        # Bearing / Direction
        bearing = exif.get("EXIF:GPSImgDirection") or exif.get("Composite:GPSImgDirection")
        if bearing is not None:
            gps_info["bearing"] = bearing
            
        norm["gps"] = gps_info

    # Camera / device
    make = exif.get("EXIF:Make")
    model = exif.get("EXIF:Model")
    if make or model:
        norm["device"] = {"make": make, "model": model}

    # Input source info
    if source and source.get("type") == "url":
        norm["file_url"] = source.get("file_url")
    if original_name:
        norm["file_name"] = original_name

    # Duration for A/V (ffprobe is best)
    if ffprobe.get("format", {}).get("duration") is not None:
        try:
            norm["duration_seconds"] = float(ffprobe["format"]["duration"])
        except Exception:
            norm["duration_seconds"] = ffprobe["format"]["duration"]

    # Codec summary
    if "streams" in ffprobe and isinstance(ffprobe["streams"], list):
        codecs = []
        for s in ffprobe["streams"]:
            c = s.get("codec_name")
            t = s.get("codec_type")
            if c or t:
                codecs.append({"type": t, "codec": c})
        if codecs:
            norm["codecs"] = codecs

    # PDF doc fields
    if mime == "application/pdf" or pdfinfo:
        if pdfinfo:
            norm["pdf"] = {
                "title": pdfinfo.get("Title"),
                "author": pdfinfo.get("Author"),
                "creator": pdfinfo.get("Creator"),
                "producer": pdfinfo.get("Producer"),
                "pages": pdfinfo.get("Pages"),
            }

    return norm


def _standard_response(
    *,
    source: Dict[str, Any],
    file_info: Dict[str, Any],
    tools: Dict[str, Any],
    normalized: Dict[str, Any],
    warnings: Optional[list] = None,
) -> Dict[str, Any]:
    return {
        "service": APP_NAME,
        "version": "1.0.0",
        "timestamp_utc": _utc_now_iso(),
        "source": source,         # where input came from
        "file": file_info,        # sha256, size, mime, etc.
        "normalized": normalized, # compact common fields
        "tools": tools,           # raw outputs (exiftool/ffprobe/pdf)
        "warnings": warnings or [],
    }


async def _download_to_temp(url: str, dirpath: str) -> Tuple[str, str]:
    """
    Download URL to a temp file with basic safety: size limit, timeout.
    Returns (path, filename).
    """
    filename = "downloaded_file"
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()

        # Try to infer filename
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^"]+)"?', cd, re.IGNORECASE)
        if m:
            filename = _safe_filename(m.group(1))

        # Enforce size limit while streaming
        logger.info("Size limit %s", MAX_DOWNLOAD_BYTES)
        out_path = os.path.join(dirpath, filename)
        total = 0
        with open(out_path, "wb") as f:
            async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="AAADownloaded file exceeds MAX_DOWNLOAD_BYTES limit")
                f.write(chunk)

    return out_path, filename


def _extract_all(path: str, original_name: str, source: Dict[str, Any]) -> Dict[str, Any]:
    mime = _detect_mime(path)
    size = _file_size(path)
    sha256 = _sha256_file(path)

    exif = _exiftool_metadata(path)

    # Only run ffprobe when it likely makes sense; harmless if it fails though.
    ffprobe = {}
    if mime.startswith("video/") or mime.startswith("audio/"):
        ffprobe = _ffprobe_metadata(path)
    else:
        # Some containers / odd types might still be media
        maybe = _ffprobe_metadata(path)
        # keep it only if it looks real
        if maybe.get("streams") or maybe.get("format"):
            ffprobe = maybe

    pdfinfo = {}
    if mime == "application/pdf":
        pdfinfo = _pdf_docinfo(path)

    normalized = _normalize_common(mime, exif, ffprobe, pdfinfo, original_name, source)

    file_info = {
        "original_name": original_name,
        "sha256": sha256,
        "size_bytes": size,
        "mime_type": mime,
    }

    tools = {
        "exiftool": exif,
        "ffprobe": ffprobe,
        "pdf_docinfo": pdfinfo,
    }

    return _standard_response(
        source=source,
        file_info=file_info,
        tools=tools,
        normalized=normalized,
        warnings=[],
    )


async def _post_callback(callback_url: str, payload: Dict[str, Any]) -> None:
    logger.info("Sending callback to %s", callback_url)
    try:
        headers = {}
        if CALLBACK_AUTH_SECRET:
            headers["Authorization"] = f"Bearer {CALLBACK_AUTH_SECRET}"
            logger.info("Sending callback with Authorization header")

        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            cb_response = await client.post(str(callback_url), json=payload, headers=headers)
            cb_response.raise_for_status()
            logger.info("Callback response status: %s", cb_response.status_code)
            logger.debug("Callback response body: %s", cb_response.text)
    except httpx.HTTPStatusError as e:
        logger.error("Callback failed for job %s with status %s: %s", payload.get("job_id"), e.response.status_code, e.response.text)
    except Exception as e:
        # Callback failures should not crash the job; store warning in job record if desired.
        logger.error("Callback failed for job %s: %s", payload.get("job_id"), e)


async def _run_job(job_id: str, tmpdir: str, path: str, original_name: str, source: Dict[str, Any], callback_url: Optional[str]):
    logger.info("Starting job %s", job_id)
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at_utc"] = _utc_now_iso()

    try:
        payload = await asyncio.to_thread(_extract_all, path, original_name, source)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["finished_at_utc"] = _utc_now_iso()
        JOBS[job_id]["result"] = payload
        logger.info("Job %s completed successfully", job_id)

        if callback_url:
            await _post_callback(callback_url, {"job_id": job_id, "status": "done", "result": payload})

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e)
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["finished_at_utc"] = _utc_now_iso()
        JOBS[job_id]["error"] = str(e)
        if callback_url:
            await _post_callback(callback_url, {"job_id": job_id, "status": "error", "error": str(e)})

    finally:
        # Cleanup temp dir
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug("Cleaned up temp directory %s", tmpdir)
        except Exception as e:
            logger.warning("Failed to cleanup temp directory %s: %s", tmpdir, e)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "metadata",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }


@app.get("/metadata/status/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    # Keep it stable and DB-friendly
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "created_at_utc": job.get("created_at_utc"),
        "started_at_utc": job.get("started_at_utc"),
        "finished_at_utc": job.get("finished_at_utc"),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.get("/metadata/download/{job_id}")
def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail=f"Job status is {job.get('status')}")
    return job.get("result")


@app.post("/metadata")
async def extract_metadata(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
    force_async: bool = Form(False),
):
    """
    Unified metadata extraction endpoint:
      - file: multipart upload
      - file_url: URL to download
      - callback_url: optional callback
      - force_async: force background processing
    """
    logger.info("Metadata request received file=%s, file_url=%s, callback_url=%s, force_async=%s",
                file.filename if file else "None", file_url, callback_url, force_async)

    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

    job_id = str(uuid.uuid4())
    tmpdir = tempfile.mkdtemp(prefix="meta_")
    
    try:
        if file:
            safe_name = _safe_filename(file.filename or "upload")
            path = os.path.join(tmpdir, safe_name)
            with open(path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    if f.tell() > MAX_DOWNLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Uploaded file exceeds MAX_DOWNLOAD_BYTES limit")
            source = {"type": "upload"}
            original_name = safe_name
        else:
            # Download from URL
            logger.info("Downloading file from %s", file_url)
            source = {"type": "url", "file_url": str(file_url)}
            path, original_name = await _download_to_temp(str(file_url), tmpdir)

        wants_async = bool(force_async or callback_url)
        if wants_async:
            JOBS[job_id] = {"status": "queued", "created_at_utc": _utc_now_iso()}
            background_tasks.add_task(_run_job, job_id, tmpdir, path, original_name, source, callback_url)
            return JSONResponse(
                status_code=202, 
                content={
                    "job_id": job_id, 
                    "status": "queued",
                    "status_url": f"/metadata/status/{job_id}",
                    "download_url": f"/metadata/download/{job_id}"
                }
            )

        # Sync path
        try:
            payload = await asyncio.to_thread(_extract_all, path, original_name, source)
            logger.info("Metadata extracted successfully (sync)")
            return payload
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug("Cleaned up temp directory %s", tmpdir)

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if isinstance(e, HTTPException):
            raise e
        logger.error("Metadata endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
