import pytest
import httpx
import base64
import time
import os

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost")

SERVICES = [
    "tts",
    "ocr",
    "extraction",
    "resume",
    "asr-tiny",
    "sentiment",
    "emotion-nuance",
    "emotion-vibe",
    "emotion-state",
    "debug-processor",
    "barcode",
    "callback-tester",
    "ocrpdf",
    "ics",
    "media"
]

@pytest.mark.parametrize("service", SERVICES)
def test_service_health(service):
    """Verify that every service exposing a health endpoint returns HTTP 200."""
    url = f"{BASE_URL}/{service}/health"
    response = httpx.get(url, timeout=10.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_ocr_scan():
    """E2E test for barcode service."""
    url = f"{BASE_URL}/ocr/v1/ocr"
    # Using a known QR code image URL or local file
    # For CI, it's better to use a local file if possible, or a very reliable URL.
    # Let's use a simple data URL or a mock if we don't have a file yet.
    # Actually, let's just use the requirement to have tests/data/

    image_path = "tests/data/ocr_test.png"
    # If image doesn't exist, we skip or fail.
    # I will create a small QR code image in the next step.
    if not os.path.exists(image_path):
        pytest.skip("Test image not found")

    with open(image_path, "rb") as f:
        files = {"file": ("ocr_test.png", f, "image/png")}
        response = httpx.post(url, files=files, timeout=30.0)

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data

def test_sentiment_analysis():
    """E2E test for sentiment service."""
    url = f"{BASE_URL}/sentiment/v1/sentiment"
    payload = {"text": "I love this project, it is amazing!"}
    response = httpx.post(url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data

def test_barcode_scan():
    """E2E test for barcode service."""
    url = f"{BASE_URL}/barcode/v1/barcode"
    # Using a known QR code image URL or local file
    # For CI, it's better to use a local file if possible, or a very reliable URL.
    # Let's use a simple data URL or a mock if we don't have a file yet.
    # Actually, let's just use the requirement to have tests/data/
    
    image_path = "tests/data/qr_test.png"
    # If image doesn't exist, we skip or fail. 
    # I will create a small QR code image in the next step.
    if not os.path.exists(image_path):
        pytest.skip("Test image not found")

    with open(image_path, "rb") as f:
        files = {"file": ("qr_test.png", f, "image/png")}
        response = httpx.post(url, files=files, timeout=30.0)
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["count"] >= 0

def test_debug_processor_e2e():
    """E2E test for debug-processor service, which aggregates other services."""
    url = f"{BASE_URL}/debug-processor/v1/process"
    payload = {
        "text": "Hello world. I am happy.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello world."},
            {"start": 1.0, "end": 2.0, "text": "I am happy."}
        ],
        "metadata": {
            "model_size": "tiny",
            "compute_type": "int8",
            "device": "cpu",
            "processing_time": 0.1,
            "hostname": "test-host"
        }
    }
    response = httpx.post(url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "segments" in data
    # Check if at least some analysis was performed
    assert "sentiment" in data["analysis"]


def test_media_resolve():
    """E2E test for media resolve service (link -> hosted audio URL).

    Opt-in via MEDIA_RESOLVE_E2E=1 because it downloads from an external
    platform (YouTube) and therefore needs network access.
    """
    if os.getenv("MEDIA_RESOLVE_E2E") != "1":
        pytest.skip("Set MEDIA_RESOLVE_E2E=1 to run (downloads an external media URL)")

    url = f"{BASE_URL}/media/v1/media/resolve"
    payload = {"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", "ttl_hours": 1}
    response = httpx.post(url, json=payload, timeout=30.0)
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_url = f"{BASE_URL}/media/v1/media/status/{job_id}"
    for _ in range(60):
        data = httpx.get(status_url, timeout=10.0).json()
        if data["status"] != "processing":
            break
        time.sleep(2)
    assert data["status"] == "completed", data.get("error")
    assert "result" in data
    assert "file_url" in data["result"]

    file_response = httpx.get(data["result"]["file_url"], timeout=30.0)
    assert file_response.status_code == 200
    assert len(file_response.content) > 0


def test_generate_ics():
    """E2E test for the ICS generator (event calendar)."""
    url = f"{BASE_URL}/ics/v1/generate-ics"
    payload = {
        "timezone": "Europe/Berlin",
        "events": [
            {
                "start": "2026-05-01T10:00:00",
                "end": "2026-05-01T11:00:00",
                "summary": "Team Meeting",
                "location": "Berlin",
                "description": "Quarterly planning",
            }
        ],
    }
    response = httpx.post(url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert data["fileName"] == "schedule.ics"
    assert data["mimeType"] == "text/calendar"

    ics = base64.b64decode(data["data"]).decode("utf-8")
    assert "BEGIN:VCALENDAR" in ics
    assert "BEGIN:VEVENT" in ics
    assert "Team Meeting" in ics


def test_generate_ics_invalid_payload():
    """Validation errors from the ICS generator come back as JSON 400."""
    url = f"{BASE_URL}/ics/v1/generate-ics"
    response = httpx.post(
        url, json={"events": [{"start": "not-a-date", "end": "x", "summary": "A"}]}, timeout=30.0
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_generate_birthday_cal():
    """E2E test for the birthday calendar generator."""
    url = f"{BASE_URL}/ics/v1/generate-birthday-cal"
    payload = {"birthdays": [{"name": "Anna", "date": "1990-07-04"}]}
    response = httpx.post(url, json=payload, timeout=30.0)
    assert response.status_code == 200
    data = response.json()
    assert data["fileName"] == "birthdays.ics"

    ics = base64.b64decode(data["data"]).decode("utf-8")
    assert "BEGIN:VCALENDAR" in ics
    assert "RRULE:FREQ=YEARLY" in ics
    assert "Birthday: Anna" in ics
