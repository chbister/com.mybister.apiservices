# API Services Project

This project provides a robust, containerized suite of AI microservices covering **Speech-to-Text (ASR), Speaker Diarization, Text-to-Speech, Sentiment & Emotion Analysis, OCR, Document Extraction, NER, Resume Parsing, QR/Barcode Recognition, Audio Conversion, and Result Aggregation** – all routed via Traefik and orchestrated with Docker Compose.

## 🚀 Architecture Overview

The system is designed with a **Traefik-based routing** pattern. All external traffic flows through Traefik, which routes requests to internal services based on the PathPrefix.

### Key Components:
1.  **Traefik**: Central entry point and reverse proxy. Handles routing and middlewares like `StripPrefix` and `ForwardAuth`.
2.  **ASR Services**: Multiple instances of Faster-Whisper (Tiny to Large-v3) supporting both synchronous and asynchronous (Pull/Push) processing, plus a local Apple-GPU endpoint (`/v1/asr/gpu`, mlx-whisper).
3.  **Diarization Service**: Speaker detection (`speaker-check`) and full speaker timelines (`diarize`) via pyannote.
4.  **Analysis Services**: Multilingual Sentiment and Emotion detection (base, Nuance, Vibe, State, and Hate variants).
5.  **Processor Service**: An aggregator that orchestrates calls between ASR and Analysis services (`/v1/process`) and merges ASR + diarization into speaker protocols (`/v1/transcript`).
6.  **Speech Services**: Text-to-Speech (XTTS v2 with voice cloning) and audio conversion (`convert`, ffmpeg in-container).
7.  **Document Services** (CareerGraph profile): OCR (async, with dedicated worker), Resume parsing, Text/URL extraction, NER.
8.  **Recognition Services**: Barcode/QR detection from images, QR code generation, food recognition (zero-shot).

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
- **GPU Endpoint (lokal)**: `POST /asr-<model>/v1/asr/gpu` – same contract, but transcribes with
  [mlx-whisper](https://github.com/ml-explore/mlx-whisper) on the Apple GPU (ctranslate2 is CPU-only
  on macOS). Supports `model_size` (`tiny|small|medium|large-v3` or a full HF repo) and
  `word_timestamps` (word-level timestamps à la WhisperX). Run it outside Docker via
  `make local-asr` (see "Local GPU runs" below).

### 2b. Diarization (Speaker)
- **Speaker check**: `POST /diarization/v1/speaker-check` – returns only `multi_speaker` + `number_speakers`.
- **Full timeline**: `POST /diarization/v1/diarize` – returns the complete speaker timeline:
  `{"speakers": [{"start": 72.1, "end": 76.4, "speaker": "SPEAKER_00"}, ...],
  "number_speakers": 2, "multi_speaker": true, "metadata": {...}}`
- **Status/Download**: `GET /diarization/v1/diarize/status/{job_id}` and `.../download/{job_id}`
  (same job storage as `speaker-check`).
- Audio decoding works without host ffmpeg (soundfile → PyAV → pydub fallback chain).

### 2c. Convert (ffmpeg, container-only)
- **Endpoint**: `POST /convert/v1/convert` – converts audio and returns it as a binary response.
- **Params**: `file` | `file_url`, `format` (`wav|mp3|flac|ogg|m4a`, default `wav`),
  `sample_rate` (default `16000`), `mono` (default `true`).
- **Response headers**: `X-Original-Name`, `X-Duration-Seconds`, `X-Sample-Rate`, `X-Channels`.
- ffmpeg lives only inside this container – the host never needs it.

### 3. Analysis Services (Multilingual)
All emotion services share the same code base; the model is selected via `EMOTION_MODEL_ID`.
- **Sentiment**: `POST /sentiment/v1/sentiment` – multilingual sentiment scoring.
- **Emotion (base)**: `POST /emotion/v1/emotion` – 6 basic emotions (DistilBERT default).
- **Emotion (Nuance)**: `POST /emotion-nuance/v1/emotion` – 28 fine-grained emotions.
- **Emotion (State)**: `POST /emotion-state/v1/emotion` – 7 core psychological states.
- **Emotion (Vibe)**: `POST /emotion-vibe/v1/emotion` – 1-5 star rating.
- **Emotion (Hate)**: `POST /emotion-hate/v1/emotion` – hate-speech classifier (`cardiffnlp/twitter-roberta-base-hate`).

### 4. Processor (The Aggregator)
Combines ASR results with all analysis services.
- **Endpoint**: `POST /processor/v1/process`
- **Logic**: It accepts an ASR JSON, processes segments in parallel, and returns a fully enriched "Voice Diary" object.
- **Transcript endpoint**: `POST /processor/v1/transcript` – merges ASR segments with the
  diarization timeline into a speaker protocol:
  - Input: `asr_result` (answer of `/v1/asr`) + `diarization` (answer of `/v1/diarize`), or flat
    `segments` + `turns`; optional `speaker_names` (`{"SPEAKER_00": "Anna"}`) and
    `format` (`json` | `text`).
  - Output: `segments` (with `speaker`, split at speaker changes), `utterances` (grouped),
    `speakers` (summary) and `transcript`:
    ```text
    [00:01:12] Anna: Wann hat das Gespräch begonnen?
    [00:01:16] Sprecher 2: Kurz nach neun.
    ```
  - Step-by-step guide (German): [docs/transkription-anleitung.md](docs/transkription-anleitung.md).

### 5. TTS (Text-to-Speech)
- **Endpoint**: `POST /tts/v1/tts`
- **Models**: Coqui **XTTS v2** (multilingual, voice cloning) with GPU detection;
  automatic CPU fallback to German Tacotron2 (`tts_models/de/thorsten/tacotron2-DDS_German`).
- Configurable voices via `TTS_DEFAULT_VOICE` / `TTS_DEFAULT_MALE_VOICE`, model root via `TTS_MODEL_ROOT`.

### 6. OCR (CareerGraph profile)
Async document OCR with a dedicated worker container (`ocr-worker`).
- **Endpoint**: `POST /ocr/v1/ocr` (UploadFile or `file_url`) – returns a job immediately for larger documents.
- **Status/Download**: `GET /ocr/v1/ocr/status/{job_id}` and `GET /ocr/v1/ocr/download/{job_id}`.
- Deployed in the `careergraph` profile (`make up-careergraph-prod|dev`), production host `ask.vereinstalk.de`.

### 7. Extraction & NER
- **Extraction**: `POST /extraction/v1/extract` – extracts structured data (names, phone numbers, …) from text/lines via pluggable extractor strategies (`mode`). Production host `ask.vereinstalk.de`.
- **NER**: `POST /ner/v1/ner` – named entity recognition, async with
  `GET /ner/v1/ner/status/{job_id}` and `.../download/{job_id}`.

### 8. Resume Parsing (CareerGraph profile)
- **Endpoint**: `POST /resume/v1/resume/extract` – parses CVs (stations, dates, contact data) with an optional NER model (`NER_ENABLED`/`NER_MODEL_ID`).
- Part of the `careergraph` profile alongside `ocr`/`ocr-worker`.

### 9. Metadata
- **Endpoint**: `POST /metadata/metadata` (Traefik strips the `/metadata` prefix, the app route itself is `/metadata`, hence the doubled path) – generates/returns metadata for media files, async with `GET /metadata/metadata/status/{job_id}` and `GET /metadata/metadata/download/{job_id}`.
- Config: `MAX_DOWNLOAD_BYTES`, `DOWNLOAD_TIMEOUT_S`.

### 10. Barcode & QR
- **Barcode detection**: `POST /barcode/v1/barcode` (image upload or URL).
- **Highlight**: `POST /barcode/v1/barcode/highlight` (returns image with high-contrast borders).
- **QR generation**: `GET /qrcode/v1/qrcode?data=...` – renders a QR code as image stream (generation, not detection).

### 11. Food Recognition *(code ready, not yet in compose)*
- **Endpoint**: `POST /food-recognition/v1/detect` – zero-shot food item detection (GoutWatcher component list).
- The service code exists (`food-recognition-service/`), but its compose entries are currently commented out.

### 12. Statusbar *(code ready, not yet in compose)*
- **Endpoint**: `POST /statusbar/v1/statusbar/edit` – server-side status bar image editing (PIL).
- Not wired into `docker-compose.yml` yet (commented out).

### 13. Callback Tester
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

### Profiles
| Profile | Services |
|---|---|
| `prod` / `dev` | Main suite (ASR, diarization, analysis, processor, TTS, OCR, conversion, …) |
| `careergraph-prod` / `careergraph-dev` | CareerGraph subset: `ocr`, `ocr-worker`, `resume` |
| `kickfriends-prod` / `kickfriends-dev` | Kickfriends subset |

### Development Mode
```bash
make up-dev
```

### Production Mode
```bash
make up-prod
```

### CareerGraph / Kickfriends
```bash
make up-careergraph-prod   # or up-careergraph-dev, up-kickfriends-prod, ...
```

### Checking Status
Each service has a `/health` endpoint that returns the **hostname/container ID**, which is useful for verifying load balancing or gateway routing.

---

## 🔗 DNS Management (Hetzner CNAMEs)

Every independently deployable service gets a CNAME
`<service-id>.api.mybister.com → <host>.mybister.com.` in the Hetzner Cloud
DNS zone `mybister.com` (CNAME values are written absolute, i.e. with a
trailing dot). The repo ships a small CLI/library for that:

```bash
export HETZNER_API_TOKEN="..."            # Hetzner Cloud API token (Read & Write)

python -m tools.hetzner_dns list                    # show current CNAMEs
python -m tools.hetzner_dns bulk --host b03         # dry-run: plan all services
python -m tools.hetzner_dns bulk --host b03 --apply # create missing CNAMEs
python -m tools.hetzner_dns set asr-small b02 --apply
python -m tools.hetzner_dns get asr-small
```

`set` and `bulk` are dry-runs by default; diverging CNAMEs are only
overwritten in bulk with an explicit `--force`. The central service list
lives in `tools/service_registry.json` and is reused by the tool.
Full documentation (config, exit codes, JSON output, Python API,
GitHub Actions): **[docs/hetzner-dns.md](docs/hetzner-dns.md)**.

---

## 🍎 Local GPU Runs (no Docker)

> **Step-by-step guide (German):** [docs/transkription-anleitung.md](docs/transkription-anleitung.md)
> – complete walkthrough from audio file to speaker protocol, incl. timing measurements.

Docker on macOS has no GPU access, so the GPU-capable endpoints are started directly on the host.
Only Python libraries and models are required – no additional applications (ffmpeg stays in the
`convert` container).

### One-time setup
```bash
pip install -r requirements-local.txt   # pyannote 3.3.1 (Docker parity) + mlx-whisper
pip install python-ulid pytest          # if not already present in the venv
```
Models are downloaded lazily on first use (Whisper via mlx-community, pyannote 3.1 via HF_TOKEN).

### Start services (one terminal each)
```bash
make local-asr           # POST /v1/asr + /v1/asr/gpu        → http://127.0.0.1:8100
make local-diarization   # POST /v1/speaker-check + /v1/diarize → http://127.0.0.1:8101
make local-processor     # POST /v1/process + /v1/transcript   → http://127.0.0.1:8102
```

### End-to-end example (audio → speaker protocol)
```bash
# 1) Transkription auf der Apple-GPU
curl -X POST http://127.0.0.1:8100/v1/asr/gpu \
  -F "file=@meeting.m4a" -F "language=de" -F "word_timestamps=false" -o asr.json

# 2) Sprecher-Zeitlinie
curl -X POST http://127.0.0.1:8101/v1/diarize \
  -F "file=@meeting.m4a" -o diar.json

# 3) Merge zum Protokoll
curl -X POST http://127.0.0.1:8102/v1/transcript \
  -H "Content-Type: application/json" \
  -d "{\"asr_result\": $(cat asr.json), \"diarization\": $(cat diar.json),
       \"speaker_names\": {\"SPEAKER_00\": \"Anna\"}, \"format\": \"text\"}"
```

### Notes
- **GPU**: Only `mlx-whisper` uses the Apple GPU; `faster-whisper`/ctranslate2 is CPU-only on macOS.
  Diarization runs on CPU (MPS is supported but slower for pyannote).
- **ffmpeg**: Not installed on the host. The GPU endpoints decode audio via `soundfile`/`PyAV`
  (bundled ffmpeg libs). For converting files, use the `convert` container.
- **Docker parity**: `requirements-local.txt` pins the same pyannote/torch versions as
  `diarization-service/requirements.txt`, so local and container results match.
- **Docker vs. local**: `/v1/asr/gpu` also exists in the container and returns a clear error there
  (mlx-whisper is intentionally not part of the Docker image).
