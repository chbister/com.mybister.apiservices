# OCR/LLM Contact Extraction Service

A Python-based service that performs OCR on uploaded images and uses a candidate validation pipeline to extract structured data.

## Features

* Extract WhatsApp group member names from screenshots using a robust validation pipeline.
* Extensible architecture for future extraction modes (e.g., phone numbers).
* FastAPI-based with `/health` and `/v1/ocr-llm` endpoints.

## Usage

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "ocr-llm",
  "hostname": "b89ce9a97ef0",
  "datetime": "2026-05-21T21:01:33.530219"
}
```

### OCR/LLM Extraction

```http
POST /v1/ocr-llm
```

**Form Data:**
* `file`: (Optional) Image file upload.
* `file_url`: (Optional) URL to an image file.
* `mode`: (Optional, default: "name") Extraction mode. Supports:
    * `"name"`
    * `{"mode": "name"}`
    * `{"mode": ["name"]}`

**Response:**
```json
{
  "results": [
    {
      "data": "Max Mustermann",
      "type": "NAME"
    },
    {
      "data": "Erika Musterfrau",
      "type": "NAME"
    }
  ],
  "raw_ocr": {
    "text": "Max Mustermann\nErika Musterfrau\n+49 170 1234567\nGroup members"
  },
  "count": 2,
  "hostname": "b89ce9a97ef0",
  "file_name": "whatsapp-group-members.png",
  "file_url": null
}
```

## Environment Variables

* `OCR_ENGINE`: `TESSERACT` (default), `EASYOCR`, or `PADDLE`.
* `EXTRACTION_STRATEGY`: `DEFAULT` (default).
* `NER_MODEL_ID`: Hugging Face model ID for NER (default: `Davlan/xlm-roberta-base-ner-hrl`).
* `OCR_CLEANUP_REMOVE_UI`: `true` (default) or `false`.
* `OCR_CLEANUP_REMOVE_STATUS`: `true` (default) or `false`.
* `OCR_CLEANUP_REMOVE_URLS`: `true` (default) or `false`.
* `OCR_CLEANUP_REMOVE_PHONE_NUMBERS`: `true` (default) or `false`.
* `VALIDATION_MAX_NAME_WORDS`: Maximum words in a name (default: `4`).
* `VALIDATION_MAX_LINE_LENGTH`: Maximum characters in a name line (default: `50`).
* `VALIDATION_REMOVE_UI`: `true` (default) or `false`.
* `VALIDATION_REMOVE_STATUS`: `true` (default) or `false`.

## Development

Run via Docker Compose:

```bash
docker compose --profile dev up ocr-llm-dev
```

## Async OCR (`/v1/ocr`) — Bilder und PDFs

Der aktuelle Service nimmt über **einen** Endpoint alles entgegen und erkennt
den Typ selbst anhand der Magic Bytes (Extension/MIME sind unzuverlässig):

* `image` (PNG, JPEG, WEBP, GIF, BMP, TIFF) → direkt durch die OCR-Engine.
* `pdf` → Seiten werden gerendert (pypdfium2) und einzeln erkannt, danach mit
  Form-Feed (`\f`) konkateniert. Antwort enthält zusätzlich `pages`.
* sonst → Job schlägt sauber fehl (`failed` + Fehlermeldung im Callback).

```http
POST /v1/ocr
```

**Form Data:**
* `file`: Datei-Upload (Bild oder PDF).
* `file_url`: alternativ URL zur Datei.
* `callback_url`: Webhook für das Ergebnis (`{job_id, status, result|error}`).
* `max_pages`: (optional) Seiten-Limit für PDFs, gedeckelt durch `OCR_MAX_PAGES`.

Antwort ist immer `202` mit `job_id`, `status_url`, `download_url`.
Ergebnis-`metadata` enthält `detected_type`, `page_count`
(bei Kürzung zusätzlich `truncated: true`, `total_pages`).

## Environment Variables (Worker)

* `OCR_ENGINE`: `EASYOCR` (default), `TESSERACT` oder `PADDLE`.
* `OCR_LANG`: z. B. `de,en`.
* `OCR_MAX_PAGES`: Seiten-Limit für PDFs (default: `10`).
