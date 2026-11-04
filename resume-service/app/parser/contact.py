"""Kontakt-Header-Parsing (Name, E-Mail, Telefon, Ort).

Der Header sind die Zeilen vor der ersten Sektions-Überschrift, typisch:
  LEBENSLAUF
  Christian Bister
  Stuttgart | +49 173 4671730 | cb@abwx.de
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# EasyOCR-Artefakt: Punkt vor der TLD fehlt ("cb@abwx de" statt "cb@abwx.de").
_BROKEN_EMAIL_RE = re.compile(
    r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+?)\s+([A-Za-z]{2,4})\b"
)

# Fallback, falls phonenumbers kein Ergebnis liefert.
_PHONE_FALLBACK_RE = re.compile(r"\+?[\d][\d\s/().-]{5,}\d")

_SKIP_HEADER_WORDS = {
    "lebenslauf",
    "curriculum vitae",
    "resume",
    "cv",
}

_NAME_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:[-\s]+[A-ZÄÖÜ][a-zäöüß]+)+$")


@dataclass
class Contact:
    display_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None


def _find_phone(text: str) -> str | None:
    try:
        import phonenumbers

        for match in phonenumbers.PhoneNumberMatcher(text, "DE"):
            return phonenumbers.format_number(
                match.number, phonenumbers.PhoneNumberFormat.E164
            )
    except Exception as exc:  # Bibliothek fehlt oder Text unparsebar.
        logger.debug("phonenumbers-Matching übersprungen: %s", exc)
    fallback = _PHONE_FALLBACK_RE.search(text)
    return fallback.group(0).strip() if fallback else None


def parse(header_lines: list[str]) -> Contact:
    contact = Contact()
    candidates: list[str] = []

    for line in header_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in _SKIP_HEADER_WORDS:
            continue

        if contact.email is None:
            email = _EMAIL_RE.search(stripped)
            if email:
                contact.email = email.group(0)

        if contact.phone is None:
            phone = _find_phone(stripped)
            if phone:
                contact.phone = phone

        # "Stuttgart | +49 ... | cb@abwx.de": Segmente ohne Mail/Telefon sind Orte.
        if "|" in stripped:
            for segment in (s.strip() for s in stripped.split("|")):
                if not segment or _EMAIL_RE.search(segment) or _find_phone(segment):
                    continue
                if contact.location is None and len(segment) < 60:
                    contact.location = segment
        candidates.append(stripped)

    if contact.display_name is None:
        for candidate in candidates:
            short = " ".join(candidate.split("|")[0].split())
            if (
                _EMAIL_RE.search(short)
                or _find_phone(short)
                or len(short) > 60
            ):
                continue
            if _NAME_RE.match(short):
                contact.display_name = short
                break

    return contact
