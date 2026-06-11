import pytest
import httpx
import time
import os

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:80")

SERVICES = [
    "asr-tiny",
    "sentiment",
    "emotion-nuance",
    "emotion-vibe",
    "emotion-state",
    "processor",
    "barcode",
    "callback-tester"
]

@pytest.mark.parametrize("service", SERVICES)
def test_service_health(service):
    """Verify that every service exposing a health endpoint returns HTTP 200."""
    url = f"{BASE_URL}/{service}/health"
    # The development Traefik might need a few seconds to register services
    max_retries = 5
    for i in range(max_retries):
        try:
            response = httpx.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "ok"
                return
        except Exception:
            pass
        time.sleep(2)
    
    response = httpx.get(url, timeout=10.0)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

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

def test_processor_e2e():
    """E2E test for processor service, which aggregates other services."""
    url = f"{BASE_URL}/processor/v1/process"
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
