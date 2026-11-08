# Fehler-Tracking mit GlitchTip

Alle Services melden Exceptions und Performance-Traces an die selbst gehostete
GlitchTip-Instanz unter <https://glitchtip.mybister.com> (Sentry-kompatibel,
gleiches Event-Format, gleiche SDKs).

## Die zwei URLs

| URL | Zweck |
|---|---|
| `https://<key>@glitchtip.mybister.com/2` | **DSN** – das ist die Adresse, die `sentry_sdk.init(dsn=…)` bzw. `Sentry.init({dsn})` braucht. `2` ist die Project-ID. |
| `https://glitchtip.mybister.com/api/2/security/?glitchtip_key=<key>` | Nur für **Browser/JS-SDKs** (u. a. CSP-Reports). In diesem Repo aktuell **nicht** verwenden – es gibt kein Frontend. |

Die DSN ist ein Secret (Write-Zugang zum Projekt). Sie steht ausschließlich in
der lokalen `.env` bzw. in der `.env` des Deployment-Hosts – `.env` ist
gitignored, `.env.example` enthält nur Platzhalter.

## Konfiguration (Umgebungsvariablen)

| Variable | Default | Bedeutung |
|---|---|---|
| `SENTRY_DSN` | *(leer)* | DSN des GlitchTip-Projekts. **Leer = Tracking komplett aus.** |
| `SENTRY_ENVIRONMENT` | `dev` | Environment-Tag, z. B. `prod` auf dem Deployment-Host. |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Anteil der Requests, die als Performance-Traces gesendet werden (0 = nur Errors). |
| `SENTRY_RELEASE` | *(leer)* | Optionaler Release-Tag, z. B. `git rev-parse --short HEAD`. |
| `SENTRY_SEND_TEST_EVENT` | *(leer)* | `1` → beim Start einmal ein Test-Event senden. |

In `docker-compose.yml` werden die Variablen über den Anchor `x-common-env`
an alle Services durchgereicht (plus eigene Blöcke für `barcode`, `qrcode`,
`callback-tester`, `debug-processor` und `ocrpdf`/`ocrpdf-dev`).

## Was wird gemeldet?

- **FastAPI-Services (18 Stück)**: Init-Block direkt vor `app = FastAPI(…)` in
  `*/app/main.py` – guarded, d. h. ohne `SENTRY_DSN` (oder ohne installiertes
  `sentry-sdk`) verhält sich der Service exakt wie bisher. Integrations:
  `FastApiIntegration` (Request-Kontext, 500er, Stacktraces),
  `send_default_pii=False`.
- **Background-Jobs**, die nicht im Request-Handler landen, melden explizit:
  `asr-service` (`run_async_job`, `run_async_job_gpu`, Download-Handler),
  `diarization-service`, `ner-service`, `ocr-service/app/worker.py`
  (eigener Prozess, eigener Init-Block) und der TTS-Modell-Ladefehler im
  `lifespan` – jeweils `sentry_sdk.capture_exception(e)`.
- **`ocrpdf-service` (Node/Express)**: `@sentry/node` in `server.js`
  (`Sentry.init`, `captureException` im OCR-catch-Block,
  `setupExpressErrorHandler`), Abhängigkeit liegt in `package.json`.
- Zusätzlich fängt die SDK-Standard-`LoggingIntegration` alle `ERROR`-Logs als
  Events ab (INFO-Logs werden nur als Breadcrumbs mitgeschickt).

## Docker-Layer

`sentry-sdk[fastapi]` wird **nicht** in die `requirements.txt` eingetragen,
sondern als eigener Layer direkt danach installiert:

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "sentry-sdk[fastapi]>=2.0"
```

Damit bleibt der große requirements-Layer (torch, whisper, pyannote …)
gecacht und muss bei einem SDK-Update nicht neu gebaut werden.

## Verifikation

```bash
# 1. DSN in der .env setzen, dann z. B. den qrcode-Service lokal starten:
SENTRY_SEND_TEST_EVENT=1 uvicorn app.main:app --port 8000

# 2. Test-Event sollte in der GlitchTip-UI unter "Issues" erscheinen
#    (ggf. bis zu 30 s warten, Events werden gebatcht).

# 3. Echten Fehler provozieren – Stacktrace + Request-Kontext prüfen:
curl "http://127.0.0.1:8000/v1/qrcode?data="   # 422/500 je nach Endpoint

# 4. Mit Umgebungsvariable erzwingen:
SENTRY_DSN=https://…@glitchtip.mybister.com/2 SENTRY_ENVIRONMENT=dev …
```

Auf dem Deployment-Host gilt dasselbe mit `SENTRY_ENVIRONMENT=prod` in der
`.env`; nach dem Deploy einmal `SENTRY_SEND_TEST_EVENT=1` setzen, um die
Erreichbarkeit des Hosts (aus dem Container-Netzwerk) zu bestätigen.

## Datenschutz / Datenminimierung

- `send_default_pii=False`: keine IPs, Cookies, Nutzer-IDs.
- Request-Inhalte (Transkripte, Bilder, PDF-Texte) werden **nicht** als Event
  gesendet – nur URL, Method, Statuscode, Stacktrace und Breadcrumbs
  (u. a. `logger`-Aufrufe).
- Vor dem Hochladen in einen öffentlichen GlitchTip ggf. `before_send`
  ergänzen, falls doch mal ein Payload-Fragment in einem Log-Breadcrumb landet.
- Die DSN behandeln wie ein Passwort; sie erlaubt das *Schreiben* von Events
  in das Projekt (Lesen nur mit GlitchTip-Login).

## Troubleshooting

| Symptom | Ursache |
|---|---|
| Keine Events | `SENTRY_DSN` leer/ungültig, Service nicht neu gebaut, oder Container kann `glitchtip.mybister.com` nicht erreichen (DNS/Proxy). |
| Doppelte Events pro Fehler | Einmal von der ASGI-/FastAPI-Integration, einmal vom `logger.error`-Logging-Integration – gleicher Stacktrace → gleiche Issue-Gruppe. |
| Keine Traces | `SENTRY_TRACES_SAMPLE_RATE=0` oder Rate zu gering. |
| Lokal `ModuleNotFoundError: sentry_sdk` | `pip install -r requirements-local.txt` (enthält `sentry-sdk[fastapi]`) ausführen – der Init-Block ist guarded und deaktiviert sich andernfalls still. |
