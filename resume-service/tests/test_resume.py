"""Tests für den Resume-Parser (deterministisch, ohne Modell-Downloads)."""

from fastapi.testclient import TestClient

from app.main import app
from app.parser import dates
from app.parser.resume import ResumeExtractor

CV_LINES = [
    "LEBENSLAUF",
    "Max Mustermann",
    "Stuttgart | +49 170 1234567 | max@example.de",
    "Kurzprofil",
    "Product Owner mit langjähriger Erfahrung in digitaler Produktentwicklung.",
    "Berufliche Stationen",
    "Product Owner myAPP 12/2021 - 08/2026",
    "MUSTER AG",
    "Produktverantwortung: Strategie, Roadmap und Backlog.",
    "Service Manager Middleware 10/2014 - 07/2021",
    "Beispiel GmbH",
    "Verantwortung für Betrieb und Modernisierung globaler Services.",
    "Freiberufliche Tätigkeit / Produktentwicklung 10/2003 - heute",
    "Kundenspezifische Integrationen und eigene Apps.",
    "Ausbildung & Zertifikate",
    "Diplom-Informatiker (FH), Hochschule Musterstadt (09/1995 - 08/2001)",
    "Zertifikate: Example Cloud Practitioner (2024), Beispiel Scrum Master (gültig bis 02/2022)",
]

client = TestClient(app)


def test_date_range_german_present():
    result = dates.parse_range("Freiberufliche Tätigkeit 10/2003 - heute")
    assert result is not None
    assert result.started_on == "2003-10-01"
    assert result.ended_on is None


def test_date_range_year_only():
    result = dates.parse_range("Studium 2015 - 2019")
    assert result is not None
    assert (result.started_on, result.ended_on) == ("2015-01-01", "2019-01-01")


def test_extract_full_cv():
    extractor = ResumeExtractor(ner_enabled=False)
    result = extractor.extract("\n".join(CV_LINES), CV_LINES, "de")

    assert result.profile["display_name"] == "Max Mustermann"
    assert result.profile["email"] == "max@example.de"
    assert result.profile["phone"] == "+491701234567"
    assert result.profile["location"] == "Stuttgart"

    assert len(result.experiences) == 3
    first, second, freelance = result.experiences
    assert first["organization"] == "myAPP MUSTER AG"
    assert first["role"] == "Product Owner"
    assert (first["started_on"], first["ended_on"]) == ("2021-12-01", "2026-08-01")
    assert "Produktverantwortung" in (first["description"] or "")
    assert second["organization"] == "Beispiel GmbH"
    assert second["role"] == "Service Manager Middleware"
    assert freelance["organization"] == "Selbstständig"
    assert freelance["ended_on"] is None

    assert len(result.educations) == 1
    education = result.educations[0]
    assert education["institution"] == "Hochschule Musterstadt"
    assert education["qualification"] == "Diplom-Informatiker (FH)"
    assert (education["started_on"], education["ended_on"]) == ("1995-09-01", "2001-08-01")

    names = [c["name"] for c in result.certifications]
    assert "Example Cloud Practitioner" in names
    assert "Beispiel Scrum Master" in names
    assert not any("Diplom" in name for name in names)


def test_extract_endpoint_contract():
    response = client.post(
        "/v1/resume/extract",
        json={"text": "\n".join(CV_LINES), "lines": CV_LINES, "locale": "de"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["profile"]["display_name"] == "Max Mustermann"
    assert data["count"] == (
        len(data["experiences"])
        + len(data["educations"])
        + len(data["certifications"])
        + len(data["projects"])
    )
    assert data["count"] > 0
    assert data["metadata"]["locale"] == "de"


def test_extract_without_headings_falls_back_to_experience():
    lines = ["Max Mustermann", "IT Consultant 01/2020 - 05/2023", "Muster GmbH"]
    result = ResumeExtractor(ner_enabled=False).extract("\n".join(lines), lines, "de")
    assert len(result.experiences) == 1
    assert result.experiences[0]["organization"] == "Muster GmbH"
    assert result.experiences[0]["role"] == "IT Consultant"


def test_extract_rejects_empty_text():
    response = client.post("/v1/resume/extract", json={"text": "   ", "lines": []})
    assert response.status_code == 400


def test_health_contract():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "resume"
    assert "supported_locales" in data["config"]
