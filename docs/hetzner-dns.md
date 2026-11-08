# Hetzner-DNS-Verwaltung (CNAME-Tool)

Eigenständiges Python-CLI-Hilfstool zur Verwaltung der CNAME-Records unserer
API-Microservices in der Hetzner-Cloud-DNS-Zone `mybister.com`.

Jeder eigenständig deploybare Service erhält einen CNAME der Form:

```
<service-id>.api.mybister.com   CNAME   <host>.mybister.com.
```

Beispiel: `asr-small.api.mybister.com → b03.mybister.com.`

> **Absolute CNAME-Werte:** Geschrieben wird der CNAME-Wert zwingend mit
> abschließendem Punkt (DNS-Wire-Format), also `b03.mybister.com.` – bei
> `CREATE`/`UPDATE` und in den Plan-Ausgaben (`target` in `set`/`bulk`).
> Eingaben (Umgebungsvariablen wie `DEFAULT_DOCKER_HOST`, `--host`,
> Service-Argumente) werden **ohne** Punkt angegeben; ein mitgeliebter
> Punkt wird akzeptiert und beim Einlesen entfernt.
>
> **Fehlender Punkt = Abweichung:** Ein gespeicherter Wert **ohne**
> Punkt (`b03.mybister.com`) gilt als abweichend und wird nicht als
> „identisch" übersprungen: `list`/`get` zeigen den Rohwert mit Status
> `DIFFERS`, `bulk` meldet dafür Konflikt und schreibt erst mit
> `--force` (bzw. `set ... --apply`) auf `b03.mybister.com.` um.
>
> **Zonenrelative RRset-Namen:** Die Hetzner-API hängt den Zonennamen an
> **jeden** RRset-Namen an, den sie empfängt. Das Tool sendet deshalb
> immer die zonenrelative Form (`asr-small.api` für
> `asr-small.api.mybister.com` in der Zone `mybister.com`, `@` = Apex) –
> ein FQDN auf dem Wire würde den verdoppelten Record
> `asr-small.api.mybister.com.mybister.com` erzeugen. Angezeigt werden
> wie gewohnt die vollständigen Namen; beim Lesen toleriert das Tool
> beide Namensformen.

Das Tool dient jetzt der initialen Bulk-Anlage und später dem automatischen
Abgleich durch den ressourcenbasierten Multi-Host-Deployment-Scheduler.

---

## 1. Voraussetzungen

- Python ≥ 3.11 und die vorhandene Projekt-`venv` (`.venv`)
- Ein Hetzner-Cloud-API-Token mit Zugriff auf die DNS-Zone `mybister.com`
- Netzzugang zu `https://api.hetzner.cloud/v1`
- Ausführung **aus dem Repository-Root** (das Modulpfad `tools.hetzner_dns`
  liegt im Repo)

Es sind keine zusätzlichen Python-Abhängigkeiten nötig – verwendet werden nur
`httpx` (bereits überall im Projekt im Einsatz) und die Standardbibliothek.
Keine Docker-Infrastruktur, keine Datenbank, keine Weboberfläche.

## 2. Hetzner-API-Token und Berechtigungen

1. Im Hetzner Cloud Console unter **Security → API Tokens** einen Token der
   Rolle **Read & Write** für das Projekt mit der DNS-Zone `mybister.com`
   anlegen.
2. Der Token braucht Lese- *und* Schreibrechte für die DNS-Zone
   (Read-only reicht nur für `list`/`get`, nicht für `--apply`).
3. Token **niemals** in Code, Konfigurationsdateien, Commits oder Logs
   eintragen – ausschließlich als Umgebungsvariable.

Das Tool verwendet ausschließlich die dokumentierten Endpunkte der
**aktuellen Hetzner Cloud API** (`https://api.hetzner.cloud/v1`):

| Endpunkt | Zweck |
|---|---|
| `GET /zones/{zone}` | Zone per Name/ID auflösen und prüfen |
| `GET /zones/{zone}/rrsets` | RRsets auflisten (paginiert) |
| `GET /zones/{zone}/rrsets/{name}/{type}` | Einzelnes RRset lesen (404 = fehlt) |
| `POST /zones/{zone}/rrsets` | RRset anlegen (asynchrone Aktion) |
| `POST .../rrsets/{name}/{type}/actions/set_records` | Records überschreiben (asynchrone Aktion) |
| `POST .../rrsets/{name}/{type}/actions/change_ttl` | TTL ändern (asynchrone Aktion) |
| `GET /actions/{id}` | Asynchrone Aktionen abfragen |

Die veraltete eigenständige API unter `dns.hetzner.com` wird **nicht**
verwendet. Aktuelle Referenz: <https://docs.hetzner.cloud/reference/cloud>
(„Zones“ und „Zone RR Sets“), Changelog: <https://docs.hetzner.cloud/changelog>.

## 3. Konfigurationsvariablen

```env
HETZNER_API_TOKEN=<secret>            # Pflicht: Hetzner-Cloud-API-Token (Read & Write)
HETZNER_DNS_ZONE=mybister.com         # DNS-Zone (Default: mybister.com)
HETZNER_DNS_TTL=60                    # TTL in Sekunden für neue/geänderte Records (Default: 60)
API_SERVICES_DOMAIN=api.mybister.com  # Service-Domain (Default: api.mybister.com)
DEFAULT_DOCKER_HOST=b03.mybister.com  # Erwarteter Ziel-Host für die Anzeige (Default: b03.mybister.com)
```

> Alle Ziel-Angaben (Config und CLI-Eingaben) werden **ohne**
> abschließenden Punkt geschrieben (`b03.mybister.com`); das Tool ergänzt
> den Punkt beim Schreiben des CNAME-Wertes selbst (`b03.mybister.com.`).
> Bereits gespeicherte Werte ohne Punkt werden als Abweichung erkannt
> und nur nach explizitem `--apply` (bzw. `bulk --force`) umgeschrieben.

Optionale Variablen:

| Variable | Bedeutung | Default |
|---|---|---|
| `HETZNER_DNS_ZONE_ID` | Zone direkt per ID ansprechen (überspringt Namens-Lookup) | – |
| `HETZNER_API_TIMEOUT` | HTTP-Timeout in Sekunden | `30` |
| `HETZNER_API_MAX_RETRIES` | Wiederholungen für Lesezugriffe | `3` |
| `API_SERVICES_REGISTRY` | Pfad zu einer alternativen Service-Registry (JSON) | `tools/service_registry.json` |

Das Token wird ausschließlich im Arbeitsspeicher gehalten, ist in `repr()`
und Logs ausgeblendet, wird nie abgelegt und nie in Fehlermeldungen
ausgegeben. `.env.example` enthält Platzhalter ohne echte Werte.

**Wichtig:** Beim Start/importieren anderer API-Services oder des Tools
werden **keine** produktiven DNS-Einträge geändert – nur explizite
`--apply`-Aufrufe schreiben.

## 4. Installation / Start

Kein Installationsschritt nötig. Aus dem Repository-Root:

```bash
# Konfiguration
export HETZNER_API_TOKEN="..."
export HETZNER_DNS_ZONE="mybister.com"
export DEFAULT_DOCKER_HOST="b03.mybister.com"

# Prüfen, dass das Tool läuft (ohne API-Zugriff)
.venv/bin/python -m tools.hetzner_dns --help
```

Für `python -m ...` ohne Präfix kann alternativ der Interpreter der `venv`
verwendet werden – alle Beispiele unten sind mit `python` formuliert; bei
Bedarf `python` durch `.venv/bin/python` ersetzen.

## 5. CLI-Befehle

```bash
python -m tools.hetzner_dns list                # alle Service-CNAMEs anzeigen
python -m tools.hetzner_dns list --sort host    # … nach Ziel-Host gruppiert
python -m tools.hetzner_dns list --json         # … als JSON
python -m tools.hetzner_dns get asr-small       # einen Service abfragen
python -m tools.hetzner_dns get asr-small --json
python -m tools.hetzner_dns set asr-small b02   # Änderung planen (Dry-Run)
python -m tools.hetzner_dns set asr-small b02 --apply
python -m tools.hetzner_dns bulk --host b03     # Bulk-Plan für alle Services
python -m tools.hetzner_dns bulk --host b03 --apply
python -m tools.hetzner_dns bulk --host b03 --apply --force
python -m tools.hetzner_dns sync                # Abgleich mit der Registry-Verteilung (Dry-Run)
python -m tools.hetzner_dns sync --apply
python -m tools.hetzner_dns services            # Service-Registry anzeigen
python -m tools.hetzner_dns services --json
```

Globale Flags je Unterbefehl: `--json` (maschinenlesbare Ausgabe),
`-v/--verbose` (Debug-Logging auf stderr, nie mit Token).

### 5.1 `list` – DNS-Einträge anzeigen

```bash
python -m tools.hetzner_dns list
```

Sortierung über `--sort`:

```bash
python -m tools.hetzner_dns list --sort service   # nach Service-Name (Standard)
python -m tools.hetzner_dns list --sort host      # nach Ziel-Host gruppiert (b01, b02, b03), je Host nach Name
```

Ausgabe (Auszug):

```text
SERVICE       DNS NAME                         TARGET             TTL  EXPECTED          STATUS
------------  -------------------------------  -----------------  ---  ----------------  --------
asr-tiny      asr-tiny.api.mybister.com        b03.mybister.com.  60   b03.mybister.com. PRESENT
asr-small     asr-small.api.mybister.com       -                  -    b03.mybister.com. MISSING
sentiment     sentiment.api.mybister.com       b01.mybister.com.  60   b01.mybister.com. PRESENT
ocr           ocr.api.mybister.com             b03.mybister.com   60   b03.mybister.com. DIFFERS
```

- `PRESENT` = CNAME existiert und zeigt auf den erwarteten Host (Registry-Feld
  `host`, Fallback `DEFAULT_DOCKER_HOST` – siehe Abschnitt 13)
- `DIFFERS` = CNAME existiert, zeigt aber woanders hin (nur Information –
  keine automatische Änderung). Auch ein Wert, dem nur der
  abschließende Punkt fehlt (`ocr` im Beispiel), liefert `DIFFERS` –
  angezeigt wird immer der **gespeicherte Rohwert**, damit die
  Abweichung sichtbar bleibt
- `MISSING` = kein CNAME angelegt
- `CONFLICT` = am selben Namen existiert ein Record anderen Typs (A/AAAA/…)

Mit `--json` liefert das Kommando ein JSON-Array der Struktur
`{service_id, fqdn, type, target, ttl, exists, conflict_types,
conflict_targets, expected_host, status}`. `target` ist der
**gespeicherte Rohwert** (ohne automatische Punkt-Ergänzung),
`expected_host` der erwartete absolute Wert.

### 5.2 `get` – einzelnen Service abfragen

```bash
python -m tools.hetzner_dns get asr-small
```

```text
Service: asr-small
DNS:     asr-small.api.mybister.com
Type:    CNAME
Target:  b03.mybister.com.
TTL:     60
```

Exit-Codes: `0` (vorhanden), `6` (noch nicht angelegt), `5` (Konflikt mit
fremdem Record-Typ).

```bash
python -m tools.hetzner_dns get asr-small --json
```

```json
{
  "service_id": "asr-small",
  "fqdn": "asr-small.api.mybister.com",
  "type": "CNAME",
  "target": "b03.mybister.com.",
  "ttl": 60,
  "exists": true,
  "conflict_types": [],
  "conflict_targets": [],
  "expected_host": "b03.mybister.com.",
  "status": "present"
}
```

### 5.3 `set` – einzelnen Service anlegen/aktualisieren

```bash
python -m tools.hetzner_dns set asr-small b02          # Dry-Run (Standard)
python -m tools.hetzner_dns set asr-small b02 --apply  # tatsächlich schreiben
python -m tools.hetzner_dns set asr-small b02.mybister.com --apply
```

Das Tool leitet daraus ab: `asr-small.api.mybister.com CNAME b02.mybister.com.`
Akzeptiert wird ein Kurz-Host (`b02` → `b02.mybister.com`) oder ein voller
FQDN innerhalb der Zone – geschrieben wird der Wert immer absolut mit
abschließendem Punkt (`b02.mybister.com.`).

Verhalten:

- fehlenden CNAME **anlegen**
- vorhandenen CNAME mit abweichendem Ziel **aktualisieren**
- vorhandenen CNAME, dem nur der abschließende Punkt fehlt
  (`b03.mybister.com` → `b03.mybister.com.`), **normalisieren**
- unveränderten CNAME **überspringen** (`SKIP`, kein Write)
- fremde Record-Typen (A/AAAA/TXT/…) am selben Namen als **Konflikt**
  behandeln und **nie** verändern/löschen
- ungültige Service-IDs und Zielhosts ablehnen (Exit-Code `2`)
- DNS-Namen außerhalb der Service-Domain bzw. Zone ablehnen

### 5.4 `bulk` – initialer Bulk-Vorgang

```bash
# Nur anzeigen, was sich ändern würde (kein Write, nur Lesezugriffe)
python -m tools.hetzner_dns bulk --host b03

# Fehlende CNAMEs tatsächlich erstellen
python -m tools.hetzner_dns bulk --host b03 --apply

# Abweichende CNAMEs bewusst überschreiben (nur mit --force)
python -m tools.hetzner_dns bulk --host b03 --apply --force
```

Beispielausgabe (Dry-Run):

```text
CREATE  asr-tiny.api.mybister.com    -> b03.mybister.com.
CREATE  asr-small.api.mybister.com   -> b03.mybister.com.
UPDATE  sentiment.api.mybister.com   -> b03.mybister.com.
SKIP    ocr.api.mybister.com         -> b03.mybister.com.

Summary:
Create: 2
Update: 1
Unchanged: 1
Conflicts: 0
Errors: 0

DRY RUN - No DNS changes applied.
```

Sicherheitsregeln des Bulk-Vorgangs:

| Fall | Verhalten |
|---|---|
| CNAME fehlt | wird mit `--apply` erstellt (`CREATE`) |
| CNAME identisch (inkl. Punkt) | wird übersprungen (`SKIP`) |
| CNAME nur ohne abschließenden Punkt (`…com` statt `…com.`) | **Konflikt** (`CONFLICT`), mit `--force` Normalisierung auf `…com.` (`UPDATE`) |
| CNAME zeigt auf anderen Host | **Konflikt** (`CONFLICT`), nicht überschrieben |
| … mit `--force` | wird umgestellt (`UPDATE`) |
| fremder Record-Typ vorhanden | **Konflikt**, auch `--force` greift hier nicht |
| nicht verwaltete Einträge | werden **nie** gelöscht (das Tool kennt keine Löschfunktion) |

> **`--force` überschreibt absichtlich vorhandene, abweichende
> Service-CNAMEs** (z. B. um einen versehentlich verteilten Stand wieder auf
> einen Host zurückzusetzen oder fehlende Punkte zu normalisieren). Immer
> erst den Dry-Run prüfen. Ohne `--force` meldet `bulk` solche Fälle als
> Konflikt und ändert nichts daran.

**Normalisierung nachvollziehen (Idempotenz prüfen):**

```bash
# 1) Dry-Run: dotlose/abweichende Werte erscheinen als CONFLICT
python -m tools.hetzner_dns bulk --host b03

# 2) Schreiben (nur die Abweichungen)
python -m tools.hetzner_dns bulk --host b03 --force --apply

# 3) Erneuter Dry-Run: jetzt müssen alle 21 Zeilen SKIP (Unchanged: 21)
#    erscheinen – andernfalls speichert Hetzner den Wert ohne Punkt,
#    bitte melden (dann bräuchte es eine andere Prüfstrategie).
python -m tools.hetzner_dns bulk --host b03
```

### 5.5 `sync` – auf die Registry-Verteilung abgleichen

```bash
python -m tools.hetzner_dns sync            # Dry-Run: was würde sich ändern?
python -m tools.hetzner_dns sync --apply    # tatsächlich schreiben
```

`sync` gleicht **jeden** Service gegen den in der Registry hinterlegten
Ziel-Host ab (`host`-Feld, siehe Abschnitt 13) – also gegen die geplante
Deployment-Verteilung (b01/b02/b03). Fehlende CNAMEs werden geplant
(`CREATE`), abweichende umgestellt (`UPDATE`), bereits korrekte
übersprungen (`SKIP`). Services **ohne** `host`-Eintrag erscheinen als
`ERROR`, bis die Zuordnung in der Registry ergänzt ist.

Anders als `bulk` (bewusst konservativ: abweichendes nur mit `--force`)
überschreibt `sync` abweichende CNAMEs bereits mit `--apply`, weil die
Registry die verbindliche Verteilungsplanung ist. Trotzdem gilt: zuerst den
Dry-Run prüfen. Damit eignet sich `sync --apply` auch als idempotenter
Abgleich für die spätere Deployment-Automatisierung.

### 5.6 `services` – Service-Registry

```bash
python -m tools.hetzner_dns services
```

Zeigt die zentrale, deklarative Service-Liste (`tools/service_registry.json`)
inkl. der bewusst ausgeschlossenen Komponenten – ganz ohne API-Zugriff und
ohne Token.

## 6. Geplante CNAMEs (Erstinitialisierung)

Alle 22 eigenständig deploybaren Services aus `tools/service_registry.json`
(Ziel-Host jeweils aus dem Registry-Feld `host`):

| Service-ID | DNS-Name | Typ | Ziel |
|---|---|---|---|
| `asr-tiny` | `asr-tiny.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `asr-small` | `asr-small.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `asr-medium` | `asr-medium.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `asr-large` | `asr-large.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `diarization` | `diarization.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `sentiment` | `sentiment.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `emotion` | `emotion.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `emotion-nuance` | `emotion-nuance.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `emotion-state` | `emotion-state.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `emotion-vibe` | `emotion-vibe.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `emotion-hate` | `emotion-hate.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `debug-processor` | `debug-processor.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `tts` | `tts.api.mybister.com` | CNAME | `b02.mybister.com.` |
| `convert` | `convert.api.mybister.com` | CNAME | `b02.mybister.com.` |
| `ocr` | `ocr.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `ocrpdf` | `ocrpdf.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `extraction` | `extraction.api.mybister.com` | CNAME | `b02.mybister.com.` |
| `ner` | `ner.api.mybister.com` | CNAME | `b01.mybister.com.` |
| `resume` | `resume.api.mybister.com` | CNAME | `b02.mybister.com.` |
| `metadata` | `metadata.api.mybister.com` | CNAME | `b03.mybister.com.` |
| `barcode` | `barcode.api.mybister.com` | CNAME | `b02.mybister.com.` |
| `qrcode` | `qrcode.api.mybister.com` | CNAME | `b02.mybister.com.` |

Skalierte Replikate (`--scale barcode=2`, `--scale metadata=2`) teilen sich
jeweils **eine** Service-ID und erhalten **einen** CNAME.

## 7. Beispielworkflow Erstinitialisierung

```bash
# Konfiguration
export HETZNER_API_TOKEN="..."
export HETZNER_DNS_ZONE="mybister.com"
export DEFAULT_DOCKER_HOST="b03.mybister.com"

# Bestehende DNS-Einträge anzeigen
python -m tools.hetzner_dns list

# Geplante Bulk-Änderungen anzeigen (Dry-Run)
python -m tools.hetzner_dns bulk --host b03

# Fehlende CNAMEs tatsächlich erstellen
python -m tools.hetzner_dns bulk --host b03 --apply

# Einen einzelnen Service später verschieben
python -m tools.hetzner_dns set asr-small b02 --apply

# Zuordnung kontrollieren
python -m tools.hetzner_dns get asr-small
```

## 8. Fehlerbehandlung und Exit-Codes

| Code | Bedeutung | Typische Ursache |
|---|---|---|
| `0` | Erfolg | Plan/Apply ohne Probleme |
| `2` | Validierungs-/Usage-Fehler | ungültige Service-ID, Zielhost außerhalb der Zone, falsche Argumente |
| `3` | Konfigurationsfehler | `HETZNER_API_TOKEN` fehlt, Zone nicht gefunden, ungültige Env-Werte |
| `4` | API-/Netzwerk-/Aktion-Fehler | HTTP-Fehler, Rate-Limit nach Retries, Timeout, fehlgeschlagene Aktion |
| `5` | Konflikt | fremder Record-Typ oder abweichender Bulk-CNAME ohne `--force` |
| `6` | Record nicht gefunden | `get` eines noch nicht angelegten Service-CNAMEs |

Verhalten bei Fehlern:

- **Rate-Limits** (`rate_limit_exceeded` / HTTP 429): für Lesezugriffe mit
  Backoff wiederholt (unter Berücksichtigung von `RateLimit-Reset`/
  `Retry-After`); schließlich als Fehler `4` gemeldet.
- **Temporäre Netzwerkfehler**: Lesezugriffe werden wiederholt; Schreibvorgänge
  nur, wenn der Request nachweislich **nicht** ausgeführt wurde (z. B.
  Verbindungsaufbau fehlgeschlagen). Mehrdeutige Fehler (5xx, Read-Timeout)
  werden **nicht** wiederholt, um keine unkontrollierten Schreiboperationen
  auszulösen – ein erneuter Lauf ist ohnehin idempotent.
- **Asynchrone Hetzner-Aktionen** werden nach jeder Schreiboperation
  abgefragt (`GET /actions/{id}`), bis `success` (oder Fehler/Timeout).
- **Fehlermeldung ohne Secret**: es werden nur API-Fehlercode, HTTP-Status
  und – falls vorhanden – die Correlation-ID ausgegeben, nie der Token.
- Einzelne Fehler im `bulk`-Lauf brechen den Gesamtlauf nicht ab; sie erscheinen
  als `ERROR` in der Zusammenfassung und führen zu Exit-Code `4`.

## 9. JSON-Ausgabe

Alle Befehle unterstützen `--json` für Automatisierung:

```bash
python -m tools.hetzner_dns get asr-small --json
python -m tools.hetzner_dns list --json
python -m tools.hetzner_dns bulk --host b03 --json
python -m tools.hetzner_dns set asr-small b02 --apply --json
```

Änderungs-Objekte (Plan und Ergebnis) haben immer einen dieser Statuswerte:

- `created` – CNAME wurde/wird erstellt
- `updated` – CNAME wurde/wird umgestellt
- `unchanged` – CNAME ist bereits korrekt (kein Write)
- `conflict` – fremder Record-Typ bzw. abweichendes Ziel ohne `--force`
- `failed` – Schreibfehler

zusätzlich `applied` (`false` im Dry-Run), `current_target`, `target`,
`ttl`, `message`, `policy`.

`bulk --json` liefert:

```json
{
  "dry_run": true,
  "force": false,
  "host": "b03",
  "summary": {"created": 2, "updated": 0, "unchanged": 1, "conflict": 0, "failed": 0},
  "results": [ ... ]
}
```

## 10. Verwendung aus Python (Deployment-Scheduler)

Die CLI ist nur ein dünner Wrapper; die eigentliche Logik ist als Bibliothek
importierbar – **ohne Subprozesse, ohne globale Seiteneffekte, ohne
versteckte DNS-Änderungen**:

```python
from tools.hetzner_dns import DnsConfig, HetznerDNSManager, ServiceRegistry

manager = HetznerDNSManager.from_env()      # liest nur die Umgebungsvariablen

records   = manager.list_service_records()  # alle Service-CNAMEs
record    = manager.get_service_record("asr-small")
plan      = manager.plan_service_record("asr-small", "b02")          # nur lesen
result    = manager.ensure_service_record("asr-small", "b02", apply=True)

bulk_plan = manager.plan_bulk("b03")                                # Dry-Run
results   = manager.apply_bulk(bulk_plan)                           # schreiben
counts    = {r.status: sum(1 for x in results if x.status == r.status) for r in results}
```

Eigenschaften:

- **Idempotenz**: `ensure_service_record`/`apply_bulk` mehrfach ausgeführt
  ergeben ab dem zweiten Mal `unchanged` ohne Writes.
- **Strukturierte Rückgaben**: `ServiceRecord`/`ChangeResult` mit `to_dict()`
  für JSON; Fehler als Typen (`ValidationError`, `ConfigError`,
  `MissingTokenError`, `APIError`, `RateLimitError`, `ActionFailedError`, …).
- **Konfigurierbare Timeouts/Retries** über `HETZNER_API_TIMEOUT`,
  `HETZNER_API_MAX_RETRIES` bzw. Konstruktorargumente.
- **Kein Side-Effekt beim Import** – Umgebungsvariablen werden erst bei
  `from_env()` gelesen, Requests erst bei Methodenaufrufen gemacht.

## 11. GitHub-Actions-Integration (später)

```yaml
jobs:
  dns-sync:
    runs-on: ubuntu-latest
    # Verhindert konkurrierende DNS-Änderungen mehrerer Jobs:
    concurrency:
      group: hetzner-dns-mybister-com
      cancel-in-progress: false
    env:
      HETZNER_API_TOKEN: ${{ secrets.HETZNER_API_TOKEN }}
      HETZNER_DNS_ZONE: mybister.com
      DEFAULT_DOCKER_HOST: b03.mybister.com
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install httpx
      - name: DNS-Plan prüfen (Dry-Run)
        run: python -m tools.hetzner_dns bulk --host b03 --json
      - name: DNS anwenden
        run: python -m tools.hetzner_dns bulk --host b03 --apply --json
```

Der Secret gehört in die **Repository-/Org-Secrets** (`HETZNER_API_TOKEN`),
nicht in die Konfigurationsdateien.

## 12. Sicherheitsmechanismen

- **Dry-Run als Standard**: `set` und `bulk` schreiben nur mit `--apply`.
- **`--force` nur explizit** für abweichende Bulk-CNAMEs.
- **Keine Löschfunktion** in dieser Phase; fremde/nicht verwaltete Records
  bleiben unangetastet.
- **Validierung**: Service-IDs müssen in der Registry stehen; Zielhosts
  müssen innerhalb der Zone und außerhalb der Service-Domain liegen;
  CNAME-Ziel darf nicht dem eigenen Namen entsprechen.
- **Konfliktbehandlung**: Ein CNAME kann nicht neben A/AAAA/TXT/… desselben
  Namens existieren – das Tool meldet den Konflikt und ändert nichts
  (der fremde Record wird nie automatisch gelöscht).
- **Schreibschutz vor Nebenläufigkeit**: Vor jedem Write wird der aktuelle
  Zustand des jeweiligen Namens erneut gelesen und neu bewertet
  (Re-Check unmittelbar vor dem Schreiben).
- **Kein Token in Logs/Errors/`repr`**, keine Speicherung des Tokens.
- **Keine automatischen DNS-Änderungen** durch Import oder Start anderer
  Services.

### Verbleibende Race Conditions

Die Hetzner Cloud API bietet **keine** konditionalen Writes (kein
Compare-and-Swap, kein `If-Match`): zwischen Re-Check und Write kann ein
paralleler Prozess denselben Namen ändern. Empfohlen:

1. DNS-ändernde Jobs immer über **eine** GitHub-Actions-`concurrency.group`
   laufen lassen (siehe oben, `cancel-in-progress: false`).
2. Bulk-Läufe bevorzugt einzeln und in Ruhe ausführen (Wartezeit zwischen
   den Writes ist minimal, da Aktionen ohnehin asynchron bestätigt werden).
3. Nach kritischen Läufen einen Kontroll-`list`-Lauf mit `--json` machen.

## 13. Service-Auswahl und Abweichungen zur Ausgangsliste

Grundlage ist die zentrale, deklarative Registry **`tools/service_registry.json`**
(einzige Quelle für das Tool; später auch für Deployment/Ressourcenplanung).
Alle 21 vorab genannten Service-IDs existieren als eigenständig deploybare
Services in `docker-compose.yml` (Traefik-Regel `Host(\`<service-id>.api.mybister.com\`)`,
hostname-basiertes HTTPS-Routing ohne PathPrefix)
und sind unverändert übernommen; inzwischen ergänzt: `ocrpdf`:

`asr-tiny`, `asr-small`, `asr-medium`, `asr-large`, `diarization`, `sentiment`,
`emotion`, `emotion-nuance`, `emotion-state`, `emotion-vibe`, `emotion-hate`,
`debug-processor`, `tts`, `convert`, `ocr`, `ocrpdf`, `extraction`, `ner`,
`resume`, `metadata`, `barcode`, `qrcode`

**Ziel-Host pro Service (Feld `host`):** Jeder Registry-Eintrag trägt den
Host, auf dem der Service in der Deployment-Verteilung läuft – damit ist die
Registry die Single Source of Truth für DNS **und** Verteilung:

| Host | Services |
|---|---|
| `b01` | `sentiment`, `ner`, `emotion`, `emotion-nuance`, `emotion-state`, `emotion-vibe`, `emotion-hate`, `debug-processor` |
| `b02` | `tts`, `convert`, `extraction`, `resume`, `barcode`, `qrcode` |
| `b03` | `asr-tiny`, `asr-small`, `asr-medium`, `asr-large`, `diarization`, `ocr`, `ocrpdf`, `metadata` |

Das Feld treibt die `EXPECTED`-Spalte in `list`/`get` (Fallback:
`DEFAULT_DOCKER_HOST`) und `sync` (Abschnitt 5.5) an. **Und es steuert das
Deployment:** Der Workflow „Deploy to self-hosted runners" startet auf jedem
Host genau die Services aus `services` + `components` mit passendem `host`;
die Liste liefert `tools/deploy_targets.py` (stdlib-only, läuft auf den
Runnern ohne venv):

```bash
python3 tools/deploy_targets.py b03          # Compose-Services für b03
python3 tools/deploy_targets.py b03 --others # alle NICHT für b03 (Prune-Kandidaten)
```

Erlaubt sind kurze Hostnamen (`b03`) oder FQDNs (`b03.mybister.com`).

**Rollen in der Registry:** Die Registry hat drei Listen mit klarer
Verantwortung – `services` (DNS **und** Deployment), `components` (nur
Deployment, kein eigener CNAME) und `excluded` (weder noch, nur
Dokumentation):

| Komponente | `host` | Rolle |
|---|---|---|
| `apikey-auth` | `*` (jeder Host) | Traefik-ForwardAuth-Middleware (Infrastruktur) |
| `ocr-worker` | `b03` | Async-Worker zur `ocr` (gemeinsame Dateisystem-Queue) |
| `callback-tester` | `b02` | Webhook-Testempfänger – CNAME manuell, kein Registry-DNS |

Bewusst **nicht** aufgenommen (unter `excluded` dokumentiert):

| Komponente | Grund |
|---|---|
| `gateway` | nginx-Gateway in `docker-compose.yml` auskommentiert |
| `food-recognition` | Code vorhanden, aber nicht in `docker-compose.yml` deploybar |
| `statusbar` | Code vorhanden, aber nicht in `docker-compose.yml` deploybar |
| `asr-vibevoice` | `asr-service-vibevoice` nicht in `docker-compose.yml` hinterlegt |

Modellvarianten (ASR tiny/small/medium/large, Emotion-Nuance/State/Vibe/Hate)
sind jeweils **eigenständig deploybare Compose-Services** mit eigenem
Traefik-Router und erhalten daher eigene CNAMEs. Replikate (`barcode`,
`metadata`) laufen unter einer gemeinsamen Service-ID.

Ein automatisierter Test (`test_registry_matches_docker_compose_services`)
prüft, dass Registry und `docker-compose.yml` synchron bleiben: neue
Compose-Services mit Host-Regel müssen in der Registry ergänzt werden (und
umgekehrt).

## 14. Tests

```bash
.venv/bin/python -m pytest tools/hetzner_dns/tests -q
```

Die Tests arbeiten ausschließlich mit einem gemockten Hetzner-API-Transport
(`httpx.MockTransport`) – **kein Netzwerkzugriff, keine produktiven
DNS-Änderungen**. Abgedeckt sind u. a.:

1. Erkennung und Anlage fehlender CNAMEs (`--apply`)
2. Überspringen identischer CNAMEs
3. Aktualisierung abweichender CNAMEs per `set`
4. Bulk-Dry-Run ohne Writes, Bulk-Anlage, Konflikte ohne/mit `--force`
5. Unveränderte A-/AAAA-Records
6. Ungültige Service-IDs und Zielhosts
7. Fehlender API-Token bei Schreiboperationen
8. API-Fehler, Rate-Limits, Netzwerkfehler (Retry-Policy)
9. Asynchrone Aktionen (Polling, Fehlerfälle)
10. Valide JSON-Ausgabe, Idempotenz, Geheimnis-Hygiene in Logs
11. Registry ohne Dubletten und Abgleich mit `docker-compose.yml`

## 15. Dateien

| Datei | Inhalt |
|---|---|
| `tools/service_registry.json` | Zentrale Service-Registry (deklarativ) |
| `tools/hetzner_dns/config.py` | Env-Konfiguration |
| `tools/hetzner_dns/registry.py` | Registry-Lader/-Validierung |
| `tools/hetzner_dns/client.py` | Hetzner-Cloud-API-Client (HTTP, Retry, Aktionen) |
| `tools/hetzner_dns/manager.py` | DNS-Logik (Plan/Ensure/Bulk) – Bibliotheksteil |
| `tools/hetzner_dns/validators.py` | DNS-Namen-/ID-Validierung |
| `tools/hetzner_dns/cli.py` | argparse-CLI |
| `tools/hetzner_dns/__main__.py` | `python -m tools.hetzner_dns` |
| `tools/hetzner_dns/tests/` | Unit-Tests mit gemockter API |
| `docs/hetzner-dns.md` | Diese Dokumentation |
