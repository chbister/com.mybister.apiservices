FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASR_MODEL_SIZE=small \
    ASR_COMPUTE_TYPE=int8 \
    ASR_DEFAULT_LANGUAGE=auto \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    UVICORN_RELOAD=0 \
    UVICORN_LOG_LEVEL=info \
    UVICORN_WORKERS=1 \
    DIARIZATION_MAX_ANALYZE_SEC=120 \
    DIARIZATION_MIN_SPEAKER_SEC=10 \
    DIARIZATION_MIN_SPEAKER_SHARE=0.15

# Install ffmpeg for pydub and faster-whisper (mp3/m4a/ogg/webm etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

# Dependencies separat, damit Builds gecached bleiben
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

RUN python -c "import pyannote.audio; from pyannote.audio import Pipeline; print('pyannote OK')"
RUN python -c "import pkgutil; print('pyannote' in [m.name for m in pkgutil.iter_modules()])"

# Prod-ready: Code ist im Image enthalten.
# In Dev wird /srv/app durch den Bind-Mount überschrieben.
COPY app /srv/app

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
