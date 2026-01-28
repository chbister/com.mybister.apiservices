FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASR_MODEL_SIZE=small \
    ASR_COMPUTE_TYPE=int8

WORKDIR /srv

# ffmpeg ist wichtig, weil Inputs oft mp3/m4a/ogg sind
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
