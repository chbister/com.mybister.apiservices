import asyncio
import hashlib
import json
import os
import re
import shutil
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

APP_NAME = "metadata-service"
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB default
DOWNLOAD_TIMEOUT_S = float(os.getenv("DOWNLOAD_TIMEOUT_S", "30"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))

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
            norm["created_at_raw"] = exif.get(c)
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
        norm["gps"] = {"lat": lat, "lon": lon}

    # Camera / device
    make = exif.get("EXIF:Make")
    model = exif.get("EXIF:Model")
    if make or model:
        norm["device"] = {"make": make, "model": model}

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
        out_path = os.path.join(dirpath, filename)
        total = 0
        with open(out_path, "wb") as f:
            async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Downloaded file exceeds MAX_DOWNLOAD_BYTES limit")
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

    normalized = _normalize_common(mime, exif, ffprobe, pdfinfo)

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
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            await client.post(str(callback_url), json=payload)
    except Exception:
        # Callback failures should not crash the job; store warning in job record if desired.
        pass


async def _run_job(job_id: str, tmpdir: str, path: str, original_name: str, source: Dict[str, Any], callback_url: Optional[str]):
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at_utc"] = _utc_now_iso()

    try:
        payload = await asyncio.to_thread(_extract_all, path, original_name, source)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["finished_at_utc"] = _utc_now_iso()
        JOBS[job_id]["result"] = payload

        if callback_url:
            await _post_callback(callback_url, {"job_id": job_id, "status": "done", "result": payload})

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["finished_at_utc"] = _utc_now_iso()
        JOBS[job_id]["error"] = str(e)
        if callback_url:
            await _post_callback(callback_url, {"job_id": job_id, "status": "error", "error": str(e)})

    finally:
        # Cleanup temp dir
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME, "time_utc": _utc_now_iso()}


@app.get("/meta/status/{job_id}")
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


@app.post("/meta/extract")
async def extract_metadata(
    background_tasks: BackgroundTasks,
    # multipart option
    file: UploadFile = File(default=None),
    # multipart fields OR query-style form fields
    callback_url: Optional[str] = Form(default=None),
    force_async: bool = Form(default=False),
    # JSON option is handled by a separate route below for clean typing
):
    """
    Multipart upload endpoint:
      - file=<uploaded>
      - callback_url (optional)
      - force_async (optional)
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded. Use file upload or POST /meta/extract-from-url.")

    job_id = str(uuid.uuid4())
    tmpdir = tempfile.mkdtemp(prefix="meta_")
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

    wants_async = bool(force_async or callback_url)
    if wants_async:
        JOBS[job_id] = {"status": "queued", "created_at_utc": _utc_now_iso()}
        background_tasks.add_task(_run_job, job_id, tmpdir, path, original_name, source, callback_url)
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    # sync
    try:
        payload = await asyncio.to_thread(_extract_all, path, original_name, source)
        return payload
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/meta/extract-from-url")
async def extract_from_url(req: ExtractRequest, background_tasks: BackgroundTasks):
    """
    JSON URL endpoint:
      {
        "file_url": "https://...",
        "callback_url": "https://... (optional)",
        "force_async": false
      }
    """
    job_id = str(uuid.uuid4())
    tmpdir = tempfile.mkdtemp(prefix="meta_")

    source = {"type": "url", "file_url": str(req.file_url)}

    # Download first (still required for sync/async)
    path, filename = await _download_to_temp(str(req.file_url), tmpdir)

    wants_async = bool(req.force_async or req.callback_url)
    if wants_async:
        JOBS[job_id] = {"status": "queued", "created_at_utc": _utc_now_iso()}
        background_tasks.add_task(_run_job, job_id, tmpdir, path, filename, source, str(req.callback_url) if req.callback_url else None)
        return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})

    # sync
    try:
        payload = await asyncio.to_thread(_extract_all, path, filename, source)
        return payload
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
