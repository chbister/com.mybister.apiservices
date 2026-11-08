FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESUME_NER_MODEL_ID=Davlan/xlm-roberta-base-ner-hrl \
    RESUME_NER_ENABLED=false

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt


# GlitchTip/Sentry-SDK als eigener Layer: der requirements-Layer (torch/whisper etc.)
# bleibt damit gecacht und muss bei einem SDK-Update nicht neu gebaut werden.
RUN pip install --no-cache-dir "sentry-sdk[fastapi]>=2.0"
COPY app /srv/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
