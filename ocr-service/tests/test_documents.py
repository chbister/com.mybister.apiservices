"""Tests für Dokument-Erkennung und Seiten-Konkatenierung.

Absichtlich ohne schwere Deps (kein pypdfium2, keine OCR-Engines):
`render_pdf_pages` wird dort gemockt, wo PDFs im Spiel sind.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from documents import detect_kind, join_page_texts


def test_detect_pdf_by_magic_bytes():
    assert detect_kind(b"%PDF-1.4 fake content") == "pdf"


def test_detect_images_by_magic_bytes():
    assert detect_kind(b"\x89PNG\r\n\x1a\nrest") == "image"
    assert detect_kind(b"\xff\xd8\xffrest") == "image"
    assert detect_kind(b"RIFF\x00\x00\x00\x00WEBP rest") == "image"
    assert detect_kind(b"GIF89arest") == "image"
    assert detect_kind(b"BMrest") == "image"


def test_detect_unsupported():
    assert detect_kind(b"PK\x03\x04 zip content") == "unsupported"
    assert detect_kind(b"") == "unsupported"
    assert detect_kind(b"plain text") == "unsupported"


def test_join_page_texts_skips_blank_pages():
    assert join_page_texts(["Seite eins", "  ", "Seite zwei"]) == "Seite eins\fSeite zwei"
    assert join_page_texts([]) == ""
