# Transkription mit Sprecher-Zuordnung – Schritt-für-Schritt-Anleitung

Diese Anleitung beschreibt, wie aus einer Audiodatei (z. B. ein Meeting-Mitschnitt als
`.m4a`, `.wav`, `.mp3`, `.flac`) ein **Sprecher-Protokoll** im Format

```text
[00:01:12] Sprecher 1: Wann hat das Gespräch begonnen?
[00:01:16] Sprecher 2: Kurz nach neun.
```

erzeugt wird. Der Ablauf sind **3 getrennte API-Aufrufe** (Client-Orchestrierung,
kein interner Workflow):

1. **Transkribieren** → `POST /v1/asr/gpu` (Apple-GPU, mlx-whisper)
2. **Sprecher bestimmen** → `POST /v1/diarize` (CPU, pyannote)
3. **Zusammenführen** → `POST /v1/transcript` (Merge zum Protokoll)

---

## 0. Voraussetzungen (einmalig)

- macOS mit Apple Silicon (GPU-Zugriff über MLX)
- Projekt-venv: `.venv` im Projektwurzelverzeichnis
- Abhängigkeiten installieren:

```bash
pip install -r requirements-local.txt   # pyannote 3.3.1 (Docker-Parität) + mlx-whisper
pip install python-ulid pytest          # falls nicht schon vorhanden
```

- `HF_TOKEN` muss in `.env` stehen (wird für das pyannote-Modell gebraucht;
  das Makefile-Ziel `local-diarization` lädt es automatisch).
- Modelle werden beim ersten Lauf automatisch von Hugging Face geladen
  (Whisper `medium` ≈ 1,5 GB, pyannote 3.1 ≈ einige hundert MB).
- **Kein ffmpeg auf dem Host nötig** – der Decode läuft über soundfile/PyAV;
  ffmpeg existiert nur im Docker-`convert`-Container.

---

## 1. Dienste starten (je ein Terminal)

```bash
make local-asr           # POST /v1/asr + /v1/asr/gpu        → http://127.0.0.1:8100
make local-diarization   # POST /v1/speaker-check + /v1/diarize → http://127.0.0.1:8101
make local-debug-processor     # POST /v1/process + /v1/transcript   → http://127.0.0.1:8102
```

Bereitschaft prüfen:

```bash
curl http://127.0.0.1:8100/health   # → {"status":"ok", ...}
curl http://127.0.0.1:8101/health   # → {"status":"ok", "has_token":true, ...}
curl http://127.0.0.1:8102/health   # → {"status":"ok", ...}
```

---

## 2. Transkribieren (GPU)

```bash
curl -X POST http://127.0.0.1:8100/v1/asr/gpu \
  -F "file=@/Pfad/zur/Datei.m4a" \
  -F "language=auto" \
  -F "model_size=medium" \
  -F "timestamps=true" \
  -o asr.json
```

| Parameter | Werte | Hinweis |
|---|---|---|
| `language` | `auto`, `de`, `en`, … | `auto` erkennt die Sprache selbst |
| `model_size` | `tiny`, `small`, `medium`, `large-v3` oder vollständiger HF-Repo-Name | `medium` ist eine gute Balance; `large-v3` = beste Qualität |
| `word_timestamps` | `true`/`false` | Wortgenauigkeit à la WhisperX (braucht `word_timestamps=true` im Merge nicht zwingend) |

**Verhalten:** Dateien länger als **5 Minuten** wechseln automatisch in den
Async-Modus – dann kommt sofort eine Job-ID zurück und man pollt:

```bash
# Antwort bei async:
# {"job_id": "...", "status_url": "...", "download_url": "..."}

curl "http://127.0.0.1:8100/v1/asr/status/<JOB_ID>"      # → {"status":"processing"|"completed"|"failed"}
curl "http://127.0.0.1:8100/v1/asr/download/<JOB_ID>" -o asr.json
```

**Gemessene Laufzeiten** (Apple-GPU, `medium`): ca. **10× schneller als Echtzeit**
(57-min-Datei ≈ 5,5 Min.). Auf CPU (Docker-Standard `faster-whisper`) wäre es
~8× langsamer.

---

## 3. Sprecher bestimmen (Diarization)

```bash
curl -X POST http://127.0.0.1:8101/v1/diarize \
  -F "file=@/Pfad/zur/Datei.m4a" \
  -F "force_async=true" \
  -o diar.json
```

`force_async=true` empfiehlich ab ~5 Min. Audio – die Antwort enthält wie oben
`job_id`/`status_url`/`download_url`. Status abfragen:

```bash
curl "http://127.0.0.1:8101/v1/diarize/status/<JOB_ID>"
curl "http://127.0.0.1:8101/v1/diarize/download/<JOB_ID>" -o diar.json
```

**Laufzeit:** pyannote läuft auf **CPU** (MPS ist langsamer) mit ~2× Echtzeit,
skaliert bei langen Dateien aber überproportional:
- 14 s Testdatei ≈ 6–8 s
- **57 Min. Datei ≈ 60 Min.** (910 Turns, 5 Sprecher)

Ergebnis-Format:

```json
{"speakers": [{"start": 72.1, "end": 76.4, "speaker": "SPEAKER_00"}, ...],
 "number_speakers": 5, "multi_speaker": true, "metadata": {...}}
```

Hinweis: In `docker-compose.yml` ist `DIARIZATION_MAX_ANALYZE_SEC: 120` gesetzt
(Docker analysiert nur die ersten 2 Min.). Lokal ist der Cap ausgeschaltet
(`0.0` = ganze Datei).

---

## 4. Protokoll erzeugen (Merge)

```bash
curl -X POST http://127.0.0.1:8102/v1/transcript \
  -H "Content-Type: application/json" \
  -d "{
    \"asr_result\": $(cat asr.json),
    \"diarization\": $(cat diar.json),
    \"format\": \"text\"
  }" -o protokoll.txt
```

Optional **echte Namen** vergeben (Schlüssel = `SPEAKER_xx` aus der Diarization):

```bash
-d "{
  \"asr_result\": $(cat asr.json),
  \"diarization\": $(cat diar.json),
  \"speaker_names\": {\"SPEAKER_00\": \"Anna\", \"SPEAKER_01\": \"Bernd\"},
  \"format\": \"text\"
}"
```

| Parameter | Werte | Hinweis |
|---|---|---|
| `format` | `json` (Default) \| `text` | `text` liefert nur den Protokolltext als `text/plain` |
| `speaker_names` | `{SPEAKER_xx: "Name"}` | fehlende Sprecher werden als `Sprecher 1..n` nummeriert |

Die `json`-Antwort enthält zusätzlich:
- `segments` – ASR-Segmente, an Sprecherwechseln gesplittet
- `utterances` – gruppierte Äußerungen gleicher Sprecher
- `speakers` – Zusammenfassung (Redezeit, Anzahl Äußerungen, erstes Auftreten)
- `counts`, `language`, `metadata`

**Sprecher-Namen herausfinden:** In der `json`-Antwort stehen unter `speakers`
Redezeit und `first_appearance` – so lässt sich z. B. der Hauptreferent
(längste Redezeit) identifizieren und per `speaker_names` benennen.

---

## 5. Dienste stoppen

```bash
lsof -ti:8100 -ti:8101 -ti:8102 | xargs kill
```

---

## Vollständiges Beispiel (so wurde GTLS_Tests_MMmT_9369.m4a transkribiert)

```bash
# Terminal 1-3:
make local-asr & make local-diarization & make local-debug-processor &   # oder einzeln

FILE="/Users/christianbister/Downloads/GTLS_Tests_MMmT_9369.m4a"   # 57 Min., Mono-AAC

# 1) Transkription (GPU, ~5,5 Min.)
curl -s -X POST http://127.0.0.1:8100/v1/asr/gpu \
  -F "file=@$FILE" -F "language=auto" -F "model_size=medium" -F "timestamps=true"
# → 202 + job_id; pollen bis "completed", dann:
#   curl -s "http://127.0.0.1:8100/v1/asr/download/<JOB_ID>" -o asr.json

# 2) Diarization (CPU, ~60 Min. bei 57 Min. Audio)
curl -s -X POST http://127.0.0.1:8101/v1/diarize \
  -F "file=@$FILE" -F "force_async=true"
# → 202 + job_id; pollen bis "completed", dann:
#   curl -s "http://127.0.0.1:8101/v1/diarize/download/<JOB_ID>" -o diar.json

# 3) Protokoll
curl -s -X POST http://127.0.0.1:8102/v1/transcript \
  -H "Content-Type: application/json" \
  -d "{\"asr_result\": $(cat asr.json), \"diarization\": $(cat diar.json), \"format\": \"text\"}" \
  -o protokoll.txt
```

Ergebnis dieser Datei: **englisch**, 747 Segmente, 910 Turns, **5 Sprecher**,
120 Äußerungen, Protokoll ≈ 45 KB.

---

## Fehlerbehebung

| Symptom | Ursache/Lösung |
|---|---|
| `Missing PYANNOTE_TOKEN/HF_TOKEN.` | `local-diarization` ohne Token gestartet – `make local-diarization` nutzen (lädt `.env`) |
| `/v1/asr/gpu` liefert Fehler „mlx-whisper ist nicht installiert" | läuft in Docker – nur lokaler Betrieb, `pip install -r requirements-local.txt` |
| `pyannote 4.x` → 403 gated `community-1` | lokal pyannote **3.3.1** nutzen (`requirements-local.txt` pint das) |
| m4a wird nicht dekodiert | kein ffmpeg nötig – soundfile→PyAV→pydub-Fallback greift automatisch |
| Diarization sehr langsam bei langen Dateien | erwartet (CPU); ggf. Datei vorher splitten oder `DIARIZATION_MAX_ANALYZE_SEC` setzen |
| Port schon belegt | `lsof -ti:8100 \| xargs kill` (bzw. 8101/8102) |

Weitere Details: README-Abschnitt „🍎 Local GPU Runs (no Docker)".
