# Audit der Service-Autarkie (Analysebericht)

> **Status:** Analyse abgeschlossen, **keine** funktionalen Änderungen umgesetzt.
> **Datum:** 05.11.2026
> **Grenze:** Diese Prüfung bewertet und dokumentiert ausschließlich. Es wurden keine
> funktionalen Abhängigkeiten entfernt, keine Workflows umstrukturiert und keine
> Service-Verantwortlichkeiten verändert. Empfehlungen sind **nicht** implementiert.
> **Abgrenzung:** Die beauftragte HTTPS-Umstellung (dedizierte HTTPS-Service-URLs,
> Traefik-Routing, API-Header-Auth) bleibt Teil der Implementierung – siehe Abschnitt 10.

---

## 1. Untersuchungsumfang und Methode

Geprüft wurden:

- `docker-compose.yml` (63 Einträge: Prod-, Dev-, Builder-Service) und `docker-compose.ci.yml`
- der Quellcode aller 18 Deployments (`asr-service`, `diarization-service`, `sentiment-service`,
  `emotion-service`, `ner-service`, `extraction-service`, `resume-service`, `ocr-service`,
  `metadata-service`, `barcode-service`, `qrcode-service`, `convert-service`, `tts-service`,
  `processor-service`, `callback-tester`, `food-recognition-service`, `statusbar-service`,
  `asr-service-vibevoice`)
- `tools/service_registry.json` (21 eigenständig deploybare Services)

Je Service wurden die zehn Prüffragen (Abschnitt 2) beantwortet und die Kopplungen in
drei Kategorien eingeordnet (Abschnitt 5): **technisch**, **funktional**,
**infrastrukturbedingt** – jeweils mit der Bewertung *notwendig / optional /
möglicherweise vermeidbar*.

Die getroffenen Entscheidungen sind im **Entscheidungsprotokoll** (Abschnitt 6) festgehalten.

---

## 2. Antworten auf die zehn Prüffragen

1. **Kann der Service unabhängig von anderen API-Services gestartet werden?**
   Ja – **kein einziges aktives `depends_on`** in der Compose (nur zwei auskommentierte
   Zeilen). Jeder Service startet isoliert und hat einen eigenen `/health`-Endpunkt.

2. **Kann er seine Kernfunktion ohne andere API-Services ausführen?**
   19 von 21 Ja. Ausnahmen:
   - `processor` (→ `debug-processor`): die Kernfunktion „vollständige Textanalyse"
     braucht 6 Downstream-Services (Start/Health bleiben autark, Fehler werden toleriert).
   - `ocr`: der Async-Pfad braucht zwingend den `ocr-worker`.

3. **Ruft er andere API-Services direkt auf?**
   Nur `processor` (6 HTTP-POSTs). Alle anderen laden ausschließlich **client-zugewiesene**
   Adressen (`file_url`, `callback_url`) – keine Service-zu-Service-Abhängigkeit.

4. **Gibt es hart codierte URLs, Docker-DNS-Aliase oder interne HTTP-Adressen?**
   Ja, mehrere – Details in T1–T6 (Abschnitt 5.1): processor-Defaults auf Docker-DNS-Namen,
   Host-Regeln `ask.mybister.com` bzw. `ask.vereinstalk.de`, ForwardAuth-Adresse
   `http://apikey-auth:8000/`, deklarierte aber im Code ungenutzte `*-link`-Aliase,
   host-spezifischer docker.sock-Pfad in `traefik-dev`.

5. **Existieren `depends_on`-Beziehungen oder gemeinsame Docker-Netze?**
   `depends_on`: keine aktiv. Gemeinsame Netze: `edge` (external,
   `EDGE_NETWORK_NAME`, Default `dev-shared-network`, geteilt mit fremden Stacks) und
   das interne Netz `app` mit `*-link`-Aliassen.

6. **Werden gemeinsame Datenbanken, Volumes, Caches oder andere Infrastrukturkomponenten benötigt?**
   **Keine** Datenbank, kein Redis/Objektspeicher, kein Celery/Kafka. Nur Volumes:
   `hf_cache` (12+ Services), `ct2_cache` (asr×4 + diarization), `paddle_cache` (ocr-Paar),
   `asr_jobs` (asr×4), `ocr_jobs` (ocr + worker), `callback_data` (callback-tester).

7. **Gibt es funktionale Abhängigkeiten, die eine unabhängige Bereitstellung verhindern?**
   Ja, zwei:
   - `processor` → `sentiment`, 4× `emotion`, `ner` (weich: Ausfälle werden toleriert,
     die Analysefelder werden `null`, kein Crash – aber die Ergebnisqualität leidet).
   - `ocr` → `ocr-worker` (hart: ohne Worker bleiben Async-Jobs dauerhaft `queued`,
     es gibt keinen Sync-Pfad).

8. **Gibt es Abhängigkeiten zwischen unterschiedlichen Modellvarianten?**
   Keine funktionalen Calls zwischen Varianten. Geteilt werden nur Caches bzw. bei ASR
   der Job-Statusordner. `processor` ruft 4 der 5 Emotion-Varianten (+ `sentiment` + `ner`)
   auf – **`emotion-hate` wird nicht genutzt**.

9. **Sind Service-URLs und Zugangsdaten konfigurierbar?**
   Model-IDs, Secrets (`HF_TOKEN`, `CALLBACK_AUTH_SECRET`, `API_KEY`) und alle
   Service-URLs sind prinzipiell per Env konfigurierbar. Die 6 Processor-URLs werden in
   der Compose allerdings **nicht gesetzt** – sie sind damit faktisch hardcoded (T1).

10. **Könnte der Service ohne Codeänderung auf einen anderen Docker-Host verschoben werden?**
    Bedingt:
    - Services ohne geteilte Volumes: ja (nur Traefik-Router/CNAME anpassen).
    - `processor`: nur mit gesetzten URL-Variablen.
    - `ocr`/asr×4/diarization: nur mit Mitführen der jeweiligen Volumes.
    - Kein Service veröffentlicht Ports; Erreichbarkeit läuft ausschließlich über Traefik
      im `edge`-Netz.

---

## 3. Ergebnistabelle

| Service | Eigenständig startbar | Kernfunktion autark | Andere API-Abhängigkeiten | Multi-Host-fähig | Handlungsbedarf |
|---|---|---|---|---|---|
| asr-tiny | Ja | Ja | keine (teilt nur Volumes) | Bedingt¹ | Job-Statusordner `asr_jobs` mit 3 Geschwistern geteilt – zunächst akzeptiert (T6) |
| asr-small | Ja | Ja | keine | Bedingt¹ | wie asr-tiny |
| asr-medium | Ja | Ja | keine | Bedingt¹ | wie asr-tiny |
| asr-large | Ja | Ja | keine | Bedingt¹ | wie asr-tiny |
| diarization | Ja | Ja | keine | Ja² | `hf_cache`/`ct2_cache` gemeinsam (optional) |
| sentiment | Ja | Ja | keine ausgehend; **wird von processor gerufen** | Ja² | – |
| emotion | Ja | Ja | keine ausgehend; wird von processor gerufen | Ja² | – |
| emotion-nuance | Ja | Ja | keine ausgehend; wird von processor gerufen | Ja² | – |
| emotion-vibe | Ja | Ja | keine ausgehend; wird von processor gerufen | Ja² | – |
| emotion-state | Ja | Ja | keine ausgehend; wird von processor gerufen | Ja² | – |
| emotion-hate | Ja | Ja | keine ausgehend; **nicht** von processor genutzt | Ja² | Design-Lücke dokumentieren (P7) |
| processor *(→ debug-processor)* | Ja | **Nein**³ | ruft sentiment, emotion(base/nuance/vibe/state), ner auf (Docker-DNS-Defaults) | **Nein ohne Konfig**⁴ | Umbenennung zur Klarstellung (T1) + 6 `*_URL`-Variablen bei der HTTPS-Umstellung setzen |
| tts | Ja | Ja | keine | Ja² | – |
| convert | Ja | Ja | keine | Ja² | nur `edge`-Netz, kein `app`-Netz |
| ocr | Ja | **Nein**⁵ | zwingend `ocr-worker` über Volume-Queue `ocr_jobs` | **Nein**⁶ | Worker-Pflichtigkeit dokumentieren (P2) |
| extraction | Ja | Ja | keine | Ja² | – |
| ner | Ja | Ja | keine ausgehend; wird von processor gerufen | Ja² | – |
| resume | Ja | Ja | keine (NER optional **in-process**, prod deaktiviert) | Ja² | – |
| metadata | Ja | Ja | keine (lädt nur Client-URLs) | Ja² | – |
| barcode | Ja | Ja | keine | Ja² | 2 Replikate, teilen nichts außer Traefik |
| qrcode | Ja | Ja | keine | Ja² | – |

**Fußnoten**

¹ `asr_jobs` + `hf_cache` + `ct2_cache` müssen mitwandern bzw. pro Host neu aufgesetzt werden.
² Nur optionale Model-Cache-Volumes; ohne Cache läuft der Service, nur langsamer beim ersten Start.
³ Start/Health sind autark; die Kernfunktion liefert ohne die 6 Downstreams nur partielle
  Ergebnisse (Fehler werden toleriert, keine Crashes).
⁴ Die 6 URL-Defaults (`http://sentiment:8000/…`) existieren nur im Compose-Netz; ein Umzug
  verlangt gesetzte `SENTIMENT_URL`, `EMOTION_*_URL`, `NER_URL`.
⁵ `/v1/ocr` ausschließlich als Async-Queue (HTTP 202, Status `queued`); ohne Worker passiert
  nie etwas.
⁶ Die Job-Queue ist ein host-lokales Dateisystem-Volume; der Worker muss denselben Host/das
  gleiche Volume haben.

**Ergänzend (nicht in der Registry):** `ocr-worker` ist korrekt ausgeschlossen, aber
funktional Teil des ocr-Deployments.

---

## 4. Antworten zu Multi-Host und Konfigurierbarkeit (Kurzfassung)

- **Öffentliche Adresse:** künftig einheitlich `<service-id>.api.mybister.com`
  (DNS-CNAMEs über `tools/hetzner_dns` verwaltet). Die Host-Regeln
  `ask.mybister.com` und `ask.vereinstalk.de` sind damit **obsolet** (T2).
- **Ports:** kein Service veröffentlicht Ports; alles läuft über Traefik im `edge`-Netz.
- **Secrets/Model-IDs:** vollständig per Env konfiguriert (`.env` bzw. Compose).
- **Service-URLs:** nur `processor` hat welche – konfigurierbar, aktuell aber ungesetzt.

---

## 5. Kopplungen im Detail

### 5.1 Technische Kopplung

| # | Befund | Bewertung | Status |
|---|---|---|---|
| T1 | processor-Defaults `http://<service>:8000/…` (Docker-DNS); Kommentar verweist auf Aliase, Code nutzt Service-Namen | möglicherweise vermeidbar | ✅ entschieden: Debug-Dienst, Umbenennung zu `debug-processor` (siehe Abschnitt 6) |
| T2 | Host-Regeln `ask.mybister.com` (18 Services) bzw. `ask.vereinstalk.de` (ocr, extraction) | notwendig heute | ✅ entschieden: obsolet, werden entfernt; künftig nur `<service-id>.api.mybister.com` |
| T3 | External-Netz `edge` (`EDGE_NETWORK_NAME`, Default `dev-shared-network`), geteilt mit fremden Stacks | notwendig fürs Routing | ✅ entschieden: bleibt unverändert |
| T4 | `*-link`-Aliase in der Compose, im Code ungenutzt | möglicherweise vermeidbar (tote Konfiguration) | ⏳ später |
| T5 | `traefik-dev` bindet docker.sock über host-spezifischen OrbStack-Pfad | vermeidbar (dev-only) | ⏳ später |
| T6 | `asr_jobs` geteilt über alle 4 ASR-Varianten (Statuspolling beliebig über Varianten hinweg; `job_id` enthält den Hostnamen) | notwendig fürs einheitliche Polling, Design alternativ vermeidbar | ✅ entschieden: kann zunächst bleiben |

### 5.2 Funktionale Kopplung

| # | Befund | Bewertung | Status |
|---|---|---|---|
| F1 | `processor` → `sentiment` + `emotion` (base/nuance/vibe/state) + `ner`, parallel via `asyncio.gather(..., return_exceptions=True)`; Ausfälle werden zu `null` toleriert | notwendig für das Kernfeature (Ausfall = Qualitätseinbruch, kein Crash) | dokumentiert; Transportadressen werden bei HTTPS-Umstellung auf die neuen HTTPS-Endpunkte umgestellt |
| F2 | `ocr`-API → `ocr-worker` (Filesystem-Queue, kein Sync-Pfad) | notwendig für Async-OCR; der Mechanismus (FS-Queue) ist möglicherweise vermeidbar | dokumentiert (P2), keine Änderung |
| F3 | Sonstige Service-zu-Service-Calls | **keine vorhanden** | – |

### 5.3 Infrastrukturabhängigkeit

| # | Befund | Bewertung | Status |
|---|---|---|---|
| I1 | Traefik + `apikey-auth`-ForwardAuth (`API_KEY`) für alle Prod-Router; Service-Health ist davon unabhängig | notwendig für den öffentlichen Zugang, nicht für Start/Kernfunktion | beibehalten |
| I2 | `hf_cache` + `HF_TOKEN` über 12+ Services | optional (Performance) | beibehalten |
| I3 | `ct2_cache` (asr×4 + diarization) | optional | beibehalten |
| I4 | `paddle_cache` + `ocr_jobs` (ocr + worker) | notwendig innerhalb des Paars | beibehalten |
| I5 | `CALLBACK_AUTH_SECRET` über alle Services (Callback-Authentifizierung) | notwendig | beibehalten |
| I6 | Keine Datenbanken/Redis/Objektspeicher/Celery | – | – |

---

## 6. Entscheidungsprotokoll

| # | Befund | Entscheidung | Status |
|---|---|---|---|
| T1 | processor-Defaults hardcoded | Der Service dient dem **Debugging** und wird zur Klarstellung zu **`debug-processor`** umbenannt; URL-Klarheit herstellen | ✅ entschieden (Umsetzung separat) |
| T2 | Host-Regeln `ask.mybister.com`, `ask.vereinstalk.de` | **Obsolet** – künftig nur `<service-id>.api.mybister.com`; beide ask-Hosts werden **entfernt** (Teil der HTTPS-Umstellung) | ✅ entschieden |
| T3 | External-Netz `edge` | **Bleibt unverändert** | ✅ akzeptiert |
| T4 | `*-link`-Aliase (im Code ungenutzt) | Später prüfen | ⏳ aufgeschoben |
| T5 | docker.sock-Pfad in `traefik-dev` (dev-only) | Später | ⏳ aufgeschoben |
| T6 | `asr_jobs` über 4 ASR-Varianten geteilt | **Kann zunächst bleiben** | ✅ bewusst akzeptiert |

**Umsetzungsaufwand Umbenennung `processor` → `debug-processor` (dokumentiert, nicht ausgeführt):**

- Registry-ID in `tools/service_registry.json`
- DNS-CNAME `debug-processor.api.mybister.com` neu anlegen (`processor.api.mybister.com`
  ggf. entfernen) – **keine** produktiven DNS-Änderungen ohne expliziten `--apply`-Lauf
- Traefik-Router/PathPrefix `/debug-processor` in `docker-compose.yml`
- Compose-Service-Name und zugehörige `-dev`-/Builder-Einträge
- `tests/test_services.py` (`SERVICES`-Liste)
- README- und Doku-Verweise

---

## 7. Nicht deployte / nicht registrierte Komponenten

| Komponente | Status |
|---|---|
| `callback-tester` | Webhook-Testempfänger (interne Hilfskomponente), nutzt eigenes Volume `callback_data` |
| `apikey-auth` | Traefik-ForwardAuth-Middleware (Infrastruktur), braucht `API_KEY` |
| `gateway` | nginx-Gateway in der Compose auskommentiert |
| `food-recognition` | Code vorhanden, nicht in `docker-compose.yml` deploybar |
| `statusbar` | Code vorhanden, nicht in `docker-compose.yml` deploybar; Test nutzt `localhost:8001` |
| `asr-service-vibevoice` | Code vorhanden, nicht in der Compose dieses Repos hinterlegt |
| `ocr-worker` | Deployed, aber ohne öffentlichen HTTP-Endpunkt (korrekt aus der Registry ausgeschlossen) |

---

## 8. Empfehlungen (nicht implementiert)

1. **R1** – `debug-processor`: die 6 `*_URL`-Variablen in der Compose explizit setzen;
   bei der HTTPS-Umstellung auf die konfigurierbaren HTTPS-Endpunkte umstellen
   (**Transportadresse ja, fachliche Kette unverändert**).
2. **R2** – `asr_jobs`-Teilung: zunächst belassen (T6), später überdenken (P3).
3. **R3** – OCR-Worker-Pflichtigkeit dokumentieren; prüfen, ob eine Warnung/Health-Metrik
   sinnvoll ist, wenn Jobs dauerhaft `queued` bleiben.
4. ~~R4~~ – **entfällt** durch Entscheidung T2 (Host-Konsistenz ist geregelt).
5. **R5** – `*-link`-Aliase und den irreführenden Kommentar im processor-Code später prüfen (T4).
6. **R6** – Ausschluss von `emotion-hate` durch den Debug-Processor als Design-Entscheidung
   bestätigen bzw. Lücke bewusst dokumentieren.
7. **R7** – Rotations-/Trennungskonzept für geteilte Secrets (`CALLBACK_AUTH_SECRET`,
   `API_KEY`, `HF_TOKEN`).
8. **R8** – Bei Multi-Host: Model-Caches pro Host statt global geteilt.
9. **R9** – Registry um Autarkie-Metadaten ergänzen (z. B. `requires: ["ocr-worker"]`).

---

## 9. Dokumentierte Architekturprobleme (für gesonderte spätere Entscheidung)

| # | Problem | Status |
|---|---|---|
| P1 | Debug-Zweck + hardcoded Docker-DNS-Adressen in `processor` | Lösungsweg entschieden: Umbenennung + URL-Klarheit (T1) |
| P2 | OCR-Async als Dateisystem-Queue zwischen zwei Containern → Host-Gebundenheit und stille Fehler ohne Worker | offen |
| P3 | Gemeinsamer Job-Statusordner über 4 ASR-Modellvarianten | zunächst akzeptiert (T6) |
| P4 | Routing aktuell über gemeinsamen Host + PathPrefix; Umstellung auf `<service-id>.api.mybister.com`, Entfernung beider ask-Hosts | Umstellungsmaßnahme (T2, Teil HTTPS-Umstellung) |
| P5 | Externes, default-benanntes `edge`-Netz, geteilt mit fremden Stacks | entschieden: bleibt (T3) |
| P6 | Gemeinsame Model-Caches und geteilte Secrets ohne Rotationskonzept | offen |
| P7 | `emotion-hate` wird nicht in die processor-Aggregation einbezogen | offen |
| P8 | Tote `*-link`-Aliase und dev-spezifischer docker.sock-Pfad | später (T4/T5) |
| P9 | Nicht deployte Komponenten mit vorhandenem Code (`gateway`, `food-recognition`, `statusbar`, `asr-service-vibevoice`) | dokumentiert |

---

## 10. Abgrenzung zur HTTPS-Umstellung

Die bereits beauftragte Umstellung auf dedizierte HTTPS-Service-URLs, Traefik-Routing und
API-Header-Authentifizierung bleibt Bestandteil der Implementierung und wird durch dieses
Audit **nicht** verändert:

- Bestehende Service-zu-Service-Aufrufe (allein `processor`) dürfen technisch auf die neuen
  **konfigurierbaren HTTPS-Endpunkte** umgestellt werden (R1).
- Die zugrunde liegenden **funktionalen Abhängigkeiten und Verarbeitungsabläufe bleiben
  unverändert** – es werden keine funktionalen Abhängigkeiten allein aufgrund dieser
  Autarkieprüfung entfernt.
- Die Host-Umstellung auf `<service-id>.api.mybister.com` inklusive Entfernung von
  `ask.mybister.com` und `ask.vereinstalk.de` ist Teil dieser Umstellung (T2).

Alle hier erkannten Architekturprobleme sind in Abschnitt 9 für eine **spätere gesonderte
Entscheidung** dokumentiert.
