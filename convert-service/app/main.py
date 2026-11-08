"""Convert-Service: Audio-Konvertierung (ffmpeg) als HTTP-Endpunkt.

Laeuft ausschliesslich als Docker-Container - auf dem Host wird bewusst kein
ffmpeg installiert. Andere Services und Clients rufen diesen Dienst auf, um
Dateien z.B. nach WAV 16 kHz mono zu normalisieren.
"""

from __future__ import annotations

import logging
import os
import socket
import tempfile
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydub import AudioSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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
    sentry_sdk.set_tag("service", "convert-service")
    if os.getenv("SENTRY_SEND_TEST_EVENT"):
        sentry_sdk.capture_message("convert-service: GlitchTip test event")

app = FastAPI(title="Audio Convert Service (ffmpeg)")

# Erlaubte Ziel-Formate -> (pydub-Format, Content-Type, Dateiendung)
TARGET_FORMATS = {
    "wav": ("wav", "audio/wav", "wav"),
    "mp3": ("mp3", "audio/mpeg", "mp3"),
    "flac": ("flac", "audio/flac", "flac"),
    "ogg": ("ogg", "audio/ogg", "ogg"),
    "m4a": ("mp4", "audio/mp4", "m4a"),  # AAC in MP4-Container
}

MAX_SAMPLE_RATE = 192000
MIN_SAMPLE_RATE = 4000


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "convert",
        "formats": sorted(TARGET_FORMATS.keys()),
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat(),
    }


async def _read_input(
    file: Optional[UploadFile], file_url: Optional[str]
) -> tuple[bytes, str]:
    """Liest entweder den Upload oder laedt die Datei von der URL."""
    if file:
        data = await file.read()
        name = file.filename or "input"
        return data, name

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            check = await client.get(file_url, headers={"Range": "bytes=0-0"})
            check.raise_for_status()
        except Exception as e:
            status_code = 400
            if hasattr(e, "response") and e.response is not None:
                status_code = e.response.status_code
            raise HTTPException(
                status_code=status_code,
                detail=f"File check failed at URL: {file_url} (Error: {e})",
            )

        resp = await client.get(file_url)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Failed to download file: {resp.status_code}",
            )
        name = os.path.basename(urlparse(file_url).path) or "input"
        return resp.content, name


@app.post("/v1/convert")
async def convert(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    format: str = Form("wav"),
    sample_rate: int = Form(16000),
    mono: bool = Form(True),
):
    """Konvertiert eine Audiodatei und gibt sie als Binary zurueck.

    Parameter:
      - file | file_url : Quelldatei
      - format          : wav | mp3 | flac | ogg | m4a  (default wav)
      - sample_rate     : Ziel-Samplerate (default 16000)
      - mono            : auf 1 Kanal mischen (default true)

    Antwort: die konvertierte Datei als Binary + Header
      X-Original-Name, X-Duration-Seconds, X-Sample-Rate, X-Channels.
    """
    if not file and not file_url:
        raise HTTPException(
            status_code=400, detail="Either 'file' or 'file_url' must be provided."
        )

    fmt = (format or "wav").lower().lstrip(".")
    if fmt not in TARGET_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported format '{format}'. "
                f"Supported: {', '.join(sorted(TARGET_FORMATS))}"
            ),
        )

    if not isinstance(sample_rate, int) or not (
        MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE
    ):
        raise HTTPException(
            status_code=400,
            detail=f"sample_rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}.",
        )

    data, original_name = await _read_input(file, file_url)

    fd, in_path = tempfile.mkstemp()
    os.close(fd)
    out_path = None
    try:
        with open(in_path, "wb") as fh:
            fh.write(data)

        audio = AudioSegment.from_file(in_path)
        original_duration = len(audio) / 1000.0

        if mono:
            audio = audio.set_channels(1)
        if sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        pydub_fmt, media_type, ext = TARGET_FORMATS[fmt]
        out_fd, out_path = tempfile.mkstemp(suffix=f".{ext}")
        os.close(out_fd)

        export_kwargs = {"format": pydub_fmt}
        if fmt == "m4a":
            export_kwargs["codec"] = "aac"
        audio.export(out_path, **export_kwargs)

        with open(out_path, "rb") as fh:
            content = fh.read()

        base = os.path.splitext(os.path.basename(original_name))[0] or "audio"
        headers = {
            "Content-Disposition": f'attachment; filename="{base}.{ext}"',
            "X-Original-Name": original_name,
            "X-Duration-Seconds": f"{original_duration:.3f}",
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": str(1 if mono else audio.channels),
        }
        logger.info(
            "Converted %s (%.1fs) -> %s (%d bytes)",
            original_name,
            original_duration,
            fmt,
            len(content),
        )
        return Response(content=content, media_type=media_type, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Conversion failed for %s: %s", original_name, e)
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        for path in (in_path, out_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
