"""Erkennung und Normalisierung deutscher/englischer Datums-Ranges in CV-Texten.

Versteht u. a.:
  12/2021 - 08/2026, 10/2003 - heute, 05/2010 - 09/2014
  09/1995 - 08/2001, 03/2022, 2015 - 2019, 2026
  mit Bindestrich, En-Dash und Em-Dash als Trenner.

Ausgabeformat: ISO-Datum (YYYY-MM-DD). ``heute``/``present``/``now``/``current``
bedeuten "laufend" und ergeben ``ended_on = None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PRESENT_WORDS = r"(?:heute|present|now|current|lfd\.?|laufend)"

_MONTH_YEAR = r"(0?[1-9]|1[0-2])\s*[./]\s*(\d{4})"
_YEAR_ONLY = r"((?:19|20)\d{2})"

_DASH = r"[-\u2013\u2014]"

DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_MONTH_YEAR}|{_YEAR_ONLY})\s*{_DASH}\s*(?P<end>{_MONTH_YEAR}|{_YEAR_ONLY}|{_PRESENT_WORDS})",
    re.IGNORECASE,
)

_SINGLE_DATE_RE = re.compile(rf"(?P<date>{_MONTH_YEAR}|{_YEAR_ONLY})")

_PRESENT_RE = re.compile(rf"^{_PRESENT_WORDS}$", re.IGNORECASE)

# Einzelne Datums-Fragmente auf eigener Zeile (EasyOCR-Wortfluss).
SINGLE_FRAG_RE = re.compile(rf"^(?:{_MONTH_YEAR}|{_YEAR_ONLY}|{_PRESENT_WORDS})$", re.IGNORECASE)
DASH_ONLY_RE = re.compile(rf"^{_DASH}$|^(?:bis|to)$", re.IGNORECASE)


@dataclass
class DateRange:
    started_on: str | None
    ended_on: str | None


def _normalize_month_year(month: str, year: str) -> str:
    return f"{int(year):04d}-{int(month):02d}-01"


def _normalize_single(value: str) -> str | None:
    value = value.strip()
    if _PRESENT_RE.match(value):
        return None
    match = re.fullmatch(rf"\s*(?:{_MONTH_YEAR}|{_YEAR_ONLY})\s*", value)
    if not match:
        return None
    month, year, year_only = match.group(1), match.group(2), match.group(3)
    if year_only is not None:
        return f"{int(year_only):04d}-01-01"
    return _normalize_month_year(month, year)


def parse_range(text: str) -> DateRange | None:
    """Sucht die erste Datums-Range in ``text`` und normalisiert sie."""
    match = DATE_RANGE_RE.search(text)
    if match is None:
        return None
    start = _normalize_single(match.group("start"))
    end_raw = match.group("end").strip()
    end = None if _PRESENT_RE.match(end_raw) else _normalize_single(end_raw)
    if start is None:
        return None
    return DateRange(started_on=start, ended_on=end)


def parse_single(text: str) -> str | None:
    """Normalisiert ein einzelnes Datum (z. B. ``03/2022`` oder ``2026``)."""
    match = _SINGLE_DATE_RE.search(text)
    if match is None:
        return None
    return _normalize_single(match.group("date"))


def strip_range(text: str) -> str:
    """Entfernt die erste Datums-Range aus ``text`` (für Rollen-/Org-Parsing)."""
    return DATE_RANGE_RE.sub("", text, count=1).strip(" \t-–—:,;")
