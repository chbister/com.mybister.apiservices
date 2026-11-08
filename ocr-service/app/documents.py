"""Dokument-Erkennung und PDF-Seiten-Rendering für den OCR-Worker.

Der Dienst entscheidet selbst anhand der Magic Bytes, was ankommt —
der Caller muss keinen Typ wählen (Extension/MIME sind unzuverlässig):

- ``image``: PNG, JPEG, WEBP, GIF, BMP, TIFF → direkt an die OCR-Engine.
- ``pdf``: Seiten werden gerendert und einzeln durch die OCR-Engine
  geschickt, danach mit Form-Feed getrennt konkateniert.
- ``unsupported``: alles andere (z. B. DOCX, ZIP) → sauberer Job-Fehler.

pypdfium2 wird bewusst lazy importiert (keine System-Deps im Slim-Image,
kein Import beim reinen Bild-Betrieb oder in Unit-Tests).
"""

from __future__ import annotations

import logging
from typing import List, Literal

logger = logging.getLogger(__name__)

DocumentKind = Literal["image", "pdf", "unsupported"]

# Seiten-Rendering: 200 DPI-Äquivalent (Skalierung bei 72pt-Basis).
PDF_RENDER_SCALE = 200 / 72


def detect_kind(data: bytes) -> DocumentKind:
    """Bestimmt den Dokumenttyp anhand der Magic Bytes."""
    if not data:
        return "unsupported"
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if data.startswith(b"\xff\xd8\xff"):
        return "image"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image"
    if data.startswith(b"BM"):
        return "image"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image"
    return "unsupported"


def render_pdf_pages(data: bytes, max_pages: int) -> List:
    """Rendert PDF-Seiten zu PIL-Bildern (max. ``max_pages``).

    Gibt Tupel (images, total_pages) zurück, damit der Caller eine
    Kürzung in den Metadata vermerken kann.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF-Verarbeitung benötigt pypdfium2 (siehe requirements.txt)."
        ) from exc

    pdf = pdfium.PdfDocument(data)
    try:
        total_pages = len(pdf)
        images = []
        for index in range(min(total_pages, max_pages)):
            page = pdf[index]
            try:
                images.append(page.render(scale=PDF_RENDER_SCALE).to_pil())
            finally:
                page.close()
        return images, total_pages
    finally:
        pdf.close()


def join_page_texts(texts: List[str]) -> str:
    """Konkateniert Seitentexte mit Form-Feed (bewahrt Seitengrenzen)."""
    return "\f".join(text for text in texts if text.strip())
