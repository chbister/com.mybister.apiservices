"""Sektions-Split für Lebensläufe anhand von Überschriften (de/en).

Erkennt kanonische Sektionen:
  profile  – Kurzprofil / Summary
  experience – Berufliche Stationen / Experience
  education – Ausbildung / Bildung / Education
  certifications – Zertifikate / Certifications
  projects – Projekte / Freelance / Products
  skills – Kompetenzen / Skills
  languages – Sprachen / Languages
  contact – Kontakt-Header (wird separat behandelt)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dates import DATE_RANGE_RE

_HEADING_PATTERNS: dict[str, list[str]] = {
    "profile": [
        r"kurzprofil",
        r"profil",
        r"summary",
        r"professional summary",
        r"über mich",
        r"about me",
    ],
    "experience": [
        r"berufliche stationen",
        r"berufserfahrung",
        r"beruflicher werdegang",
        r"work experience",
        r"professional experience",
        r"employment history",
        r"experience",
    ],
    "education": [
        r"ausbildung",
        r"bildung",
        r"studium",
        r"education",
        r"academic background",
    ],
    "certifications": [
        r"zertifikate",
        r"certifications",
        r"certificates",
        r"lizenzen",
    ],
    "projects": [
        r"projekte",
        r"freiberufliche tätigkeit",
        r"eigene (digitale )?produkte",
        r"projects",
        r"freelance",
        r"selected projects",
    ],
    "skills": [
        r"kompetenzen",
        r"kenntnisse",
        r"fähigkeiten",
        r"skills",
        r"technologie",
        r"technologies",
    ],
    "languages": [
        r"sprachen",
        r"languages",
    ],
}

_COMPILED: dict[str, list[re.Pattern[str]]] = {
    section: [re.compile(rf"^\s*{p}\s*(?:&.*|/.*)?\s*$", re.IGNORECASE) for p in patterns]
    for section, patterns in _HEADING_PATTERNS.items()
}

# Manche Headings tragen Suffixe ("Ausbildung & Zertifikate"): beide Sektionen öffnen.
_COMBINED_SPLIT_RE = re.compile(r"\s*[&/]\s*")


@dataclass
class ResumeSections:
    header: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    experience: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


def _match_heading(line: str) -> list[str]:
    """Gibt alle Sektionen zurück, deren Heading-Muster auf ``line`` passt."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return []
    # Zeilen mit Datums-Range sind Einträge (z. B. "Product Owner garrioCOM
    # 12/2021 - 08/2026"), keine Überschriften.
    if DATE_RANGE_RE.search(stripped):
        return []
    # Kombinierte Headings ("Ausbildung & Zertifikate") an & / aufteilen.
    parts = _COMBINED_SPLIT_RE.split(stripped)
    matched: list[str] = []
    for part in parts:
        for section, patterns in _COMPILED.items():
            if any(p.match(part) for p in patterns):
                if section not in matched:
                    matched.append(section)
                break
    return matched


def split(lines: list[str]) -> ResumeSections:
    """Teilt OCR-Zeilen in kanonische Sektionen. Alles vor der ersten
    Überschrift landet in ``header`` (Kontakt-Block)."""
    sections = ResumeSections()
    current: list[str] = sections.header
    actives: list[str] = []
    seen_heading = False

    for line in lines:
        matched = _match_heading(line)
        if matched:
            seen_heading = True
            actives = matched
            current = getattr(sections, matched[0])
            continue
        if not seen_heading:
            sections.header.append(line)
            continue
        if len(actives) > 1:
            # Kombiniertes Heading: Zeile an alle beteiligten Sektionen hängen,
            # damit z. B. Zertifikats-Zeilen unter "Ausbildung & Zertifikate"
            # in beiden Sektionen sichtbar sind.
            for section in actives:
                getattr(sections, section).append(line)
        else:
            current.append(line)
    return sections
