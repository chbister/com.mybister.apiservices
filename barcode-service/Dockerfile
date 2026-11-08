FROM python:3.10-slim

# Install system dependencies for zbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# GlitchTip/Sentry-SDK als eigener Layer: der requirements-Layer (torch/whisper etc.)
# bleibt damit gecacht und muss bei einem SDK-Update nicht neu gebaut werden.
RUN pip install --no-cache-dir "sentry-sdk[fastapi]>=2.0"
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
