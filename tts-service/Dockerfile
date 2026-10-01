FROM python:3.10-slim

# Install system dependencies for TTS and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install CPU-only PyTorch first to avoid large GPU-enabled binaries
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
