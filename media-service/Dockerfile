FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEDIA_JOBS_DIR=/srv/jobs \
    MEDIA_FILES_DIR=/srv/files \
    MEDIA_TTL_HOURS=24 \
    MEDIA_MAX_BYTES=524288000 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_RELOAD=0 \
    UVICORN_LOG_LEVEL=info \
    UVICORN_WORKERS=1

# ffmpeg wird von yt-dlp fuer Schnitte/Format-Remixe benoetigt
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Dependencies separat, damit Builds gecached bleiben
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

# GlitchTip/Sentry-SDK als eigener Layer: der requirements-Layer (yt-dlp etc.)
# bleibt damit gecacht und muss bei einem SDK-Update nicht neu gebaut werden.
# Pin <3: sentry-sdk 3.x kennt 'send_default_pii' nicht mehr (siehe asr-service).
RUN pip install --no-cache-dir "sentry-sdk[fastapi]>=2.26,<3"

# Prod-ready: Code ist im Image enthalten.
# In Dev wird /srv/app durch den Bind-Mount ueberschrieben.
COPY app /srv/app

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
