# Resume Extraction Service

Strukturiert OCR-/PDF-Text von Lebensläufen in Profildaten und Sektionen
(vergleichbar mit dem `extraction-service`-Muster: FastAPI, `/health`,
`/v1/...`, Startup-Preload, `hf_cache`-Volume).

Verbraucht wird er von CareerGraph (`OcrCallbackController` → `ImportDraft`):
Der Service liefert genau die Felder, die `ImportDraftData` + Profil erwarten.

## Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "resume",
  "hostname": "a1b2c3d4e5f6",
  "datetime": "2026-10-06T12:00:00.000000",
  "config": {
    "supported_locales": ["de", "en"],
    "ner_enabled": false,
    "ner_model": "Davlan/xlm-roberta-base-ner-hrl"
  }
}
```

### Resume Extraction

```http
POST /v1/resume/extract
```

**Body:**
```json
{
  "text": "LEBENSLAUF\nMax Mustermann\n...",
  "lines": ["LEBENSLAUF", "Max Mustermann", "..."],
  "locale": "de"
}
```

**Response (Auszug):**
```json
{
  "profile": {
    "display_name": "Max Mustermann",
    "email": "max@example.de",
    "phone": "+491701234567",
    "location": "Stuttgart",
    "summary": "Product Owner mit ..."
  },
  "experiences": [
    {
      "organization": "MUSTER AG",
      "role": "Product Owner",
      "started_on": "2021-12-01",
      "ended_on": "2026-08-01",
      "description": "Produktverantwortung ..."
    }
  ],
  "educations": [],
  "certifications": [],
  "projects": [],
  "warnings": [],
  "count": 1,
  "hostname": "a1b2c3d4e5f6",
  "metadata": {"processing_time": 0.0123, "model": "rule-based", "locale": "de"}
}
```

Datumsformat: ISO (`YYYY-MM-DD`, Monatserster bei `MM/YYYY`).
`heute` / `present` / `now` bedeuten "laufend" → `ended_on: null`.
Nicht zuordenbare Zeilen landen in `warnings`, statt still zu scheitern.

## Arbeitsweise

Deterministische Pipeline (CPU, keine Downloads im Standardbetrieb):

1. Sektions-Split an de/en-Überschriften
   (`Berufliche Stationen`, `Ausbildung & Zertifikate`, `Kompetenzen`, …).
   Zeilen mit Datums-Range sind nie Überschriften.
2. Kontakt-Header (vor der ersten Überschrift): Name, E-Mail (Regex),
   Telefon (`phonenumbers`, Region DE), Ort (`A | B | C`-Segmentierung).
3. Erfahrungen: Header-Zeile mit Range (`12/2021 - 08/2026`) → Rolle/Org-Split
   (camelCase-Fragment + Folgezeile, AG/GmbH-Suffixe, Freelance → `Selbstständig`).
4. Bildung: `Qualifikation, Institution (MM/YYYY - MM/YYYY)`, auch umbrochen.
5. Zertifikate: `Name (JJJJ)`-Listen, inkl. umbrochener Klammern;
   Abschluss-Zeilen werden per Stichwort-Guard ausgeschlossen.
6. Projekte: Einträge mit Range aus der Projekt-Sektion.

## Optionale HF-Verstärkung

Ohne NER läuft alles offline. Mit NER werden ORG-Kandidaten in
Erfahrungs-Headings per freiem Hugging-Face-Modell bestätigt:

```bash
RESUME_NER_ENABLED=true
RESUME_NER_MODEL_ID=Davlan/xlm-roberta-base-ner-hrl   # Default, frei
```

Das Modell wird beim Startup vorgeladen (analog `extraction-service`).
Mehrwort-Org-Namen jenseits der Suffix-Heuristik sind der Hauptgewinn;
ohne Modell bleibt der deterministische Fallback aktiv.

## Environment Variables

* `RESUME_NER_ENABLED`: `true` oder `false` (default).
* `RESUME_NER_MODEL_ID`: HF-Modell für ORG-Bestätigung
  (default: `Davlan/xlm-roberta-base-ner-hrl`).

## Development

```bash
# Tests (keine Modell-Downloads nötig)
cd resume-service && python3 -m pytest tests/ -q

# Lokal starten
uvicorn app.main:app --host 0.0.0.0 --port 8000

# via Compose (dev-Profil, Live-Reload, Gateway-Route /resume)
docker compose --profile dev up -d
```
