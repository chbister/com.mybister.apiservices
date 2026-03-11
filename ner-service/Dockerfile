FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NER_MODEL_ID=Davlan/xlm-roberta-large-ner-hrl \
    JOBS_DIR=/srv/jobs

WORKDIR /srv

RUN mkdir -p /srv/jobs && chmod 777 /srv/jobs

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
