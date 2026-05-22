# API Services Project

This project provides a robust, containerized suite of AI microservices for Speech-to-Text (ASR), Sentiment Analysis, Emotion Detection, Barcode Recognition, and Result Aggregation.

## 🚀 Architecture Overview

The system is designed with a **Traefik-based routing** pattern. All external traffic flows through Traefik, which routes requests to internal services based on the PathPrefix.

### Key Components:
1.  **Traefik**: Central entry point and reverse proxy. Handles routing and middlewares like `StripPrefix` and `ForwardAuth`.
2.  **ASR Services**: Multiple instances of Faster-Whisper (Tiny to Large-v3) supporting both synchronous and asynchronous (Pull/Push) processing.
3.  **Analysis Services**: Multilingual Sentiment and Emotion detection (Nuance, Vibe, and State variants).
4.  **Processor Service**: An aggregator that orchestrates calls between ASR and Analysis services to produce enriched results.
5.  **Barcode Service**: High-contrast recognition of QR and barcodes from images or URLs.

---

## 🛠 Services & Endpoints

### 1. Entry Point (Traefik)
Acts as the reverse proxy.
- **Production**: Uses a central Traefik instance.
- **Development**: Uses a project-specific Traefik instance on port 80.
- **URL Pattern**: `{{baseUrl}}/service-name/v1/...`

### 2. ASR (Speech-to-Text)
Supports `tiny`, `small`, `medium`, and `large-v3` models.
- **Endpoint**: `POST /asr-<model>/v1/asr`
- **Features**: 
    - `file_url` parameter for remote downloads.
    - **Async Mode**: Automatically switches to background processing for files > 5 minutes.
    - **Push Method**: Optional `callback_url` for webhooks. Supports `CALLBACK_AUTH_SECRET` for Bearer token authentication.
- **Status/Download**: `GET /asr-<model>/v1/asr/status/{job_id}` and `GET /asr-<model>/v1/asr/download/{job_id}`.

### 3. Analysis Services (Multilingual)
- **Sentiment**: `POST /sentiment/v1/sentiment`
- **Emotion (Nuance)**: `POST /emotion-nuance/v1/emotion` (28 fine-grained emotions).
- **Emotion (State)**: `POST /emotion-state/v1/emotion` (7 core psychological states).
- **Emotion (Vibe)**: `POST /emotion-vibe/v1/emotion` (1-5 star rating).

### 4. Processor (The Aggregator)
Combines ASR results with all analysis services.
- **Endpoint**: `POST /processor/v1/process`
- **Logic**: It accepts an ASR JSON, processes segments in parallel, and returns a fully enriched "Voice Diary" object.

### 5. Barcode Service
- **Endpoint**: `POST /barcode/v1/barcode`
- **Highlight**: `POST /barcode/v1/barcode/highlight` (Returns image with high-contrast borders).

### 6. Callback Tester
Simulates a webhook receiver for testing asynchronous ASR callbacks.
- **Endpoint**: `POST /callback-tester/listen`
- **Features**: 
    - Captures any incoming POST request.
    - Stores the payload and headers as JSON files.
- **List Received**: `GET /callback-tester/list`

---

## 🔄 Correlations & Relations

1.  **Environment Separation**: The project uses Docker Compose **profiles** (`prod` vs `dev`). 
    - `prod`: Stable models, optimized logging, no code mounts.
    - `dev`: Live-reloading enabled, debug logging, local code mounting for rapid development.
2.  **Internal Networking**: Services communicate over a shared `edge` network (for Traefik) and a private `app` network using DNS aliases (e.g., `asr-tiny-link`).
3.  **Resource Sharing**: All ASR and Analysis services share a common Hugging Face cache (`hf_cache`) and CTranslate2 cache (`ct2_cache`) to prevent redundant model downloads.
4.  **Async Orchestration**: ASR services use a `ProcessPoolExecutor` to handle heavy CPU tasks without blocking the web workers, while the `processor` uses `asyncio` to call multiple analysis models simultaneously.

---

## 🚦 Getting Started

### Prerequisites
- **OrbStack** (recommended for macOS) or **Docker Desktop**.
- If using OrbStack, ensure the Docker command is available in your shell.

### Development Mode
```bash
make up-dev
```

### Production Mode
```bash
make up-prod
```

### Checking Status
Each service has a `/health` endpoint that returns the **hostname/container ID**, which is useful for verifying load balancing or gateway routing.
