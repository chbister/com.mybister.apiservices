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
