"""Orchestrierung: OCR-Text → strukturierte Resume-Daten.

Pipeline:
  1. Zeilen normalisieren (leere/Seitennummern raus)
  2. Sektions-Split (de/en Headings)
  3. Kontakt-Header → Profil-Stammdaten
  4. Sektions-Parser → experiences / educations / certifications / projects

Optionale HF-Verstärkung: Ist ``RESUME_NER_ENABLED=true``, bestätigt ein
NER-Modell (Standard: Davlan/xlm-roberta-base-ner-hrl, frei) ORG-Kandidaten
in Erfahrungs-Headings. Ohne Modell läuft alles deterministisch.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from . import contact as contact_parser
from . import dates
from . import sections

logger = logging.getLogger(__name__)

NER_MODEL_ID = os.getenv("RESUME_NER_MODEL_ID", "Davlan/xlm-roberta-base-ner-hrl")
NER_ENABLED = os.getenv("RESUME_NER_ENABLED", "false").lower() == "true"

# Organisations-Suffixe (de/en) für die Heuristik.
_ORG_SUFFIX_RE = re.compile(
    r"\b(AG|GmbH|GmbH & Co\.? KG|e\.V\.|Inc\.?|Ltd\.?|S\.?A\.?\.?|SRL|d\.o\.o\.|UG|SE|LLC|GmbH|Bank|Group)\b"
)

# Detailzeilen beginnen typisch so (werden zur Description).
_DETAIL_PREFIX_RE = re.compile(
    r"^\s*(?:[-•·*]|:|"
    r"Produktverantwortung|Business-Analyse|Priorisierung|Healthcare|Führung|Ergebnis|"
    r"Transformation|Plattform|Lifecycle|Stakeholder|Technik|Service Level|Regulierte|"
    r"Automatisierung|IT-Beratung|Eigene|Software Engineering)\b",
    re.IGNORECASE,
)

# "Zertifikate: ..." einleitende Präfixe entfernen.
_CERT_PREFIX_RE = re.compile(r"^\s*(?:Zertifikate|Certificates?|Certifications?)\s*:\s*", re.IGNORECASE)

# Klammerausdrücke am Ende ("(09/1995 - 08/2001)", "(2026)", "(gültig bis 02/2022)").
_TRAILING_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")

# Abschluss-Stichwörter — schützt vor Fehlklassifikation als Zertifikat.
_DEGREE_RE = re.compile(
    r"diplom|doktor|bachelor|masterarbeit|studium|hochschule|universität|fachhochschule",
    re.IGNORECASE,
)


def _looks_like_degree(name: str) -> bool:
    return bool(_DEGREE_RE.search(name))

# "Seite x/y"-Marker und reine Zähler aus OCR-Umbrüchen ignorieren.
# Absichtlich eng (beide Zahlen kurz), damit Datums-Zeilen wie "12/2021"
# nicht fälschlich als Seitenmarker verschwinden.
_PAGE_MARKER_RE = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")


def _clean_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or _PAGE_MARKER_RE.match(stripped):
            continue
        cleaned.append(stripped)
    return _merge_date_fragments(cleaned)


def _merge_date_fragments(lines: list[str]) -> list[str]:
    """Fügt aufgebrochene Datums-Ranges wieder zusammen.

    EasyOCR liefert Wortfluss statt Zeilenfluss: "12/2021" / "-" / "08/2026"
    können auf drei Zeilen stehen. Erkannte Fragmente werden zu
    "12/2021 - 08/2026" vereint, damit die Entry-Erkennung greift.
    """
    merged: list[str] = []
    index = 0
    while index < len(lines):
        frag = _date_fragment(lines[index])
        if frag is not None and index + 1 < len(lines):
            rest = _collect_range_tail(lines, index + 1)
            if rest is not None:
                merged.append(f"{frag} - {rest[0]}")
                index += 1 + rest[1]
                continue
        merged.append(lines[index])
        index += 1
    return merged


def _date_fragment(line: str) -> str | None:
    stripped = line.strip("() ")
    if dates.SINGLE_FRAG_RE.match(stripped):
        return stripped
    return None


def _collect_range_tail(lines: list[str], index: int) -> tuple[str, int] | None:
    """Sammelt den Rest einer Range ab ``index``: optionaler Trenner
    ("-", "bis") + Enddatum. Gibt (enddatum, verbrauchte_zeilen) zurück."""
    consumed = 0
    if index < len(lines) and dates.DASH_ONLY_RE.match(lines[index].strip()):
        consumed += 1
        index += 1
    if index < len(lines):
        end = _date_fragment(lines[index])
        if end is not None:
            return end, consumed + 1
    return None


@dataclass
class ResumeResult:
    profile: dict
    experiences: list[dict] = field(default_factory=list)
    educations: list[dict] = field(default_factory=list)
    certifications: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ResumeExtractor:
    """Deterministischer Resume-Parser mit optionalem HF-NER für Organisationen."""

    def __init__(self, ner_enabled: bool = NER_ENABLED, model_id: str = NER_MODEL_ID) -> None:
        self.ner_enabled = ner_enabled
        self.model_id = model_id
        self._ner_pipe = None

    @property
    def mode_name(self) -> str:
        return "resume"

    def preload(self) -> None:
        """Lädt das NER-Modell vor (nur wenn aktiviert) — Start-Verhalten
        analog zu den anderen Services."""
        if not self.ner_enabled or self._ner_pipe is not None:
            return
        from transformers import pipeline

        logger.info("Loading resume NER model: %s", self.model_id)
        self._ner_pipe = pipeline(
            "ner", model=self.model_id, device=-1, aggregation_strategy="simple"
        )

    def _org_entities(self, lines: list[str]) -> set[str]:
        """ORG-Entities per NER (leere Menge ohne Modell)."""
        if not self.ner_enabled:
            return set()
        if self._ner_pipe is None:
            self.preload()
        found: set[str] = set()
        try:
            for line in lines:
                for entity in self._ner_pipe(line[:500]):
                    if entity.get("entity_group") == "ORG" and float(entity.get("score", 0)) >= 0.7:
                        found.add(str(entity.get("word", "")).strip())
        except Exception as exc:
            logger.warning("NER-ORG-Erkennung fehlgeschlagen: %s", exc)
        return found

    # --------------------------------------------------
    # Eintrittspunkt
    # --------------------------------------------------

    def extract(self, text: str, lines: list[str] | None = None, locale: str = "de") -> ResumeResult:
        own_lines = _clean_lines(lines if lines else text.split("\n"))
        parts = sections.split(own_lines)
        result = ResumeResult(profile={})

        contact = contact_parser.parse(parts.header)
        profile_lines = [line for line in parts.profile if line]
        result.profile = {
            "display_name": contact.display_name,
            "email": contact.email,
            "phone": contact.phone,
            "location": contact.location,
            "summary": " ".join(profile_lines)[:2000] or None,
        }

        ner_orgs = self._org_entities(own_lines[:80]) if self.ner_enabled else set()

        experience_lines = parts.experience
        if not (
            parts.experience
            or parts.education
            or parts.certifications
            or parts.projects
            or parts.profile
            or parts.skills
            or parts.languages
        ):
            # Keine einzige Überschrift erkannt: Header ersatzweise als
            # Erfahrungs-Block parsen statt leer zurückzugeben.
            experience_lines = parts.header

        result.experiences = self._parse_experiences(experience_lines, ner_orgs, result.warnings)
        result.educations = self._parse_educations(parts.education, result.warnings)
        result.certifications = self._parse_certifications(
            parts.certifications + parts.education, result.warnings
        )
        result.projects = self._parse_projects(parts.projects, result.warnings)
        return result

    # --------------------------------------------------
    # Erfahrungen
    # --------------------------------------------------

    def _parse_experiences(
        self, lines: list[str], ner_orgs: set[str], warnings: list[str]
    ) -> list[dict]:
        entries: list[dict] = []
        current: dict | None = None
        description: list[str] = []
        last_free: str | None = None

        def flush() -> None:
            if current is not None:
                if description:
                    current["description"] = " ".join(description)[:2000]
                entries.append(current)

        index = 0
        while index < len(lines):
            line = lines[index]
            date_range = dates.parse_range(line)
            if date_range is not None:
                flush()
                description = []
                remainder = dates.strip_range(line)
                if not remainder and current is None and last_free:
                    # Datum steht allein (EasyOCR-Wortfluss): Vorzeile ist Header.
                    remainder = last_free
                last_free = None
                org, role = self._split_role_org(remainder, lines, index, ner_orgs)
                if role and org:
                    current = {
                        "organization": org,
                        "role": role,
                        "employment_type": None,
                        "location": None,
                        "started_on": date_range.started_on,
                        "ended_on": date_range.ended_on,
                        "description": None,
                    }
                    # Organisations-Folgezeile (z. B. "MEDI VERBUND AG") verbrauchen.
                    if self._is_org_continuation(lines, index, org):
                        index += 1
                else:
                    warnings.append(f"Erfahrungs-Zeile ohne Rolle/Org übersprungen: {line[:80]}")
                    current = None
            elif current is not None and line:
                description.append(line.lstrip("-•·* ").strip())
            elif current is None and self._looks_like_header(line):
                last_free = line
            index += 1
        flush()
        # Leere Platzhalter (reine Warnungs-Einträge) sind bereits verworfen.
        return [e for e in entries if e.get("organization") and e.get("role")]

    def _split_role_org(
        self, remainder: str, lines: list[str], index: int, ner_orgs: set[str]
    ) -> tuple[str | None, str | None]:
        """Teilt "Product Owner garrioCOM" in (org, role)."""
        remainder = remainder.strip(" ,;")
        if not remainder:
            return None, None
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        next_is_org = bool(next_line) and self._looks_like_org(next_line)

        tokens = remainder.split()
        tail = tokens[-1] if tokens else ""
        tail_is_fragment = bool(
            re.search(r"[a-z][A-Z]", tail)  # camelCase wie garrioCOM
            or (tail[:1].islower() and len(tail) > 2)
        )

        if next_is_org:
            if tail_is_fragment and len(tokens) > 1:
                return f"{tail} {next_line}".strip(), " ".join(tokens[:-1]).strip() or None
            return next_line, remainder or None

        if _ORG_SUFFIX_RE.search(remainder):
            return self._split_suffix_org(remainder)
        if ner_orgs:
            for candidate in sorted(ner_orgs, key=len, reverse=True):
                if candidate and candidate.lower() in remainder.lower():
                    role = remainder.lower().replace(candidate.lower(), "").strip(" ,;-/")
                    return candidate, role or remainder
        if tail_is_fragment and len(tokens) > 1:
            return tail, " ".join(tokens[:-1]).strip() or None
        if re.search(r"freiberuf|freelance|selbstständig|eigene produkte", remainder, re.IGNORECASE):
            return "Selbstständig", remainder
        return None, remainder or None

    @staticmethod
    def _split_suffix_org(remainder: str) -> tuple[str | None, str | None]:
        """Teilt "Software Engineer feuersee software GmbH" anhand des
        Suffixes: Org = Suffix + vorangehende kleingeschriebene Wortfolge
        (Fallback: letztes Wort), Rolle = Rest. Mehrwort-Org-Namen aus
        Großbuchstaben löst das optionale NER-Modell präziser auf."""
        last_match = None
        for last_match in _ORG_SUFFIX_RE.finditer(remainder):
            pass
        assert last_match is not None
        left = remainder[: last_match.start()].strip()
        suffix = remainder[last_match.start() :].strip()
        words = left.split()
        run: list[str] = []
        for word in reversed(words):
            # Kleingeschriebene Firmennamen ("feuersee software") komplett
            # einsammeln, sonst nur das letzte Wort ("Daimler" aus "Daimler AG").
            if word[:1].islower() or not run:
                run.append(word)
            if word[:1].islower():
                continue
            break
        org_start = " ".join(reversed(run))
        org = f"{org_start} {suffix}".strip()
        role = left[: len(left) - len(org_start)].strip(" ,;-/") or None
        return org or None, role

    def _is_org_continuation(self, lines: list[str], index: int, org: str) -> bool:
        if index + 1 >= len(lines):
            return False
        next_line = lines[index + 1].strip()
        return bool(next_line) and next_line in org and next_line != org

    @staticmethod
    def _looks_like_header(line: str) -> bool:
        """Kandidat für einen Entry-Header ohne eigenes Datum (Lookbehind bei
        alleinstehenden Datums-Zeilen). Keine Fragmente, keine Sätze."""
        if not (4 <= len(line) <= 100):
            return False
        if not re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            return False
        if dates.SINGLE_FRAG_RE.match(line.strip("() ")):
            return False
        return True

    @staticmethod
    def _looks_like_org(line: str) -> bool:
        if not line or dates.parse_range(line) is not None or len(line) > 80:
            return False
        if _ORG_SUFFIX_RE.search(line):
            return True
        letters = [c for c in line if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.6:
            return True
        return False

    # --------------------------------------------------
    # Bildung
    # --------------------------------------------------

    def _parse_educations(self, lines: list[str], warnings: list[str]) -> list[dict]:
        entries: list[dict] = []
        pending: list[str] = []
        for line in lines:
            if not line or _CERT_PREFIX_RE.match(line):
                continue
            date_range = dates.parse_range(line)
            if date_range is None:
                # Mögliche Fortsetzung eines umbrochenen Eintrags
                # ("... Hochschule für Technik und" + "Wirtschaft (09/1995 - ...)").
                pending.append(line)
                pending = pending[-2:]
                continue
            remainder = dates.strip_range(" ".join(pending + [line]))
            pending = []
            remainder = _TRAILING_PAREN_RE.sub("", remainder).strip(" ,;")
            # "Diplom-Informatiker (FH), Fachhochschule Furtwangen - Hochschule ..."
            chunks = [c.strip() for c in re.split(r",\s*", remainder, maxsplit=1)]
            qualification = chunks[0] if chunks else remainder
            institution = chunks[1] if len(chunks) > 1 else None
            if not qualification or not institution:
                warnings.append(f"Bildungs-Zeile unvollständig übersprungen: {line[:80]}")
                continue
            entries.append(
                {
                    "institution": institution,
                    "qualification": qualification,
                    "field": None,
                    "started_on": date_range.started_on,
                    "ended_on": date_range.ended_on,
                }
            )
        return entries

    # --------------------------------------------------
    # Zertifikate
    # --------------------------------------------------

    def _parse_certifications(self, lines: list[str], warnings: list[str]) -> list[dict]:
        entries: list[dict] = []
        for line in self._join_wrapped_parens(lines):
            text = _CERT_PREFIX_RE.sub("", line).strip()
            if not text:
                continue
            # Bildungs-Zeilen ("Diplom ..., FH ... (09/1995 - 08/2001)")
            # gehören nicht hierher.
            if dates.parse_range(text) is not None:
                continue
            for item in self._split_top_level_commas(text):
                item = item.strip()
                if not item or len(item) < 3:
                    continue
                paren = _TRAILING_PAREN_RE.search(item)
                issued_on = dates.parse_single(paren.group(1)) if paren else None
                name = _TRAILING_PAREN_RE.sub("", item).strip(" ,;") if paren else item
                # "gültig bis ..." beschreibt eher ein Ablaufdatum — als issued
                # wäre es falsch; solche Items nur mit Namen übernehmen.
                if paren and "gültig" in paren.group(1).lower():
                    issued_on = None
                # Abschluss-Zeilen ohne Jahr ("Diplom-Informatiker (FH)")
                # sind keine Zertifikate.
                if issued_on is None and paren and _looks_like_degree(name):
                    continue
                if paren is None and _looks_like_degree(item):
                    continue
                if not name:
                    warnings.append(f"Zertifikats-Item ohne Namen übersprungen: {item[:60]}")
                    continue
                entries.append({"name": name, "issuer": None, "issued_on": issued_on, "expires_on": None})
        # Duplikate (Bildung+Zertifikate teilen sich Zeilen) entfernen.
        seen: set[str] = set()
        unique: list[dict] = []
        for entry in entries:
            key = entry["name"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        return unique

    @staticmethod
    def _join_wrapped_parens(lines: list[str]) -> list[str]:
        """Hängt umbrochene Klammer-Reste ("FOSS Use" + "(2020), ...")
        wieder an die Vorzeile."""
        joined: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("(") and joined:
                joined[-1] = f"{joined[-1]} {stripped}"
            else:
                joined.append(line)
        return joined

    @staticmethod
    def _split_top_level_commas(text: str) -> list[str]:
        parts, depth, current = [], 0, []
        for char in text:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        parts.append("".join(current))
        return parts

    # --------------------------------------------------
    # Projekte
    # --------------------------------------------------

    def _parse_projects(self, lines: list[str], warnings: list[str]) -> list[dict]:
        entries: list[dict] = []
        current: dict | None = None
        description: list[str] = []
        last_free: str | None = None

        def flush() -> None:
            if current is not None:
                if description:
                    current["description"] = " ".join(description)[:2000]
                entries.append(current)

        for line in lines:
            date_range = dates.parse_range(line)
            if date_range is not None:
                flush()
                description = []
                name = dates.strip_range(line).strip(" ,;-/")
                if not name and current is None and last_free:
                    name = last_free
                last_free = None
                if not name:
                    warnings.append(f"Projekt-Zeile ohne Namen übersprungen: {line[:80]}")
                    current = None
                    continue
                current = {
                    "name": name,
                    "role": None,
                    "client_name": None,
                    "started_on": date_range.started_on,
                    "ended_on": date_range.ended_on,
                    "description": None,
                }
            elif current is not None and line:
                description.append(line.lstrip("-•·* ").strip())
            elif current is None and self._looks_like_header(line):
                last_free = line
        flush()
        return [e for e in entries if e.get("name")]
