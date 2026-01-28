FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTIMENT_MODEL_ID=cardiffnlp/twitter-xlm-roberta-base-sentiment

WORKDIR /srv

COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
