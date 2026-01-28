FROM python:3.11-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# (Optional) CPU-only torch wheels - je nach Plattform brauchst du ggf. andere Pins
RUN pip install --no-cache-dir --upgrade pip

# Repo holen & editable installieren (enthält demo/ scripts)
RUN git clone --depth 1 https://github.com/microsoft/VibeVoice.git /srv/VibeVoice
WORKDIR /srv/VibeVoice
RUN pip install --no-cache-dir -e .

# API deps
WORKDIR /srv/app
COPY requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

COPY app /srv/app

ENV MODEL_PATH="microsoft/VibeVoice-ASR"
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
