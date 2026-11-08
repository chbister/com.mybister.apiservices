FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# - exiftool: broad metadata extraction
# - ffmpeg: ffprobe for audio/video metadata
# - libmagic1: python-magic backend
# - ca-certificates: HTTPS downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    exiftool \
    ffmpeg \
    libmagic1 \
    ca-certificates \
    iproute2 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt


# GlitchTip/Sentry-SDK als eigener Layer: der requirements-Layer (torch/whisper etc.)
# bleibt damit gecacht und muss bei einem SDK-Update nicht neu gebaut werden.
RUN pip install --no-cache-dir "sentry-sdk[fastapi]>=2.0"
COPY app /srv/app

# Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
CMD ["/entrypoint.sh"]
