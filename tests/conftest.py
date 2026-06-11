import pytest
import httpx
import time
import os
import sys

# Configuration constants with defaults
DEFAULT_TIMEOUT = 1200  # 20 minutes
DEFAULT_POLL_INTERVAL = 5
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost")

# List of services to check for readiness
SERVICES = [
    "ocr",
    "extraction",
    "asr-tiny",
    "sentiment",
    "emotion-nuance",
    "emotion-vibe",
    "emotion-state",
    "processor",
    "barcode",
    "callback-tester"
]

@pytest.fixture(scope="session", autouse=True)
def wait_for_all_services():
    """
    Session-level fixture that waits for all required services to become healthy.
    Blocking until all services are ready or timeout is reached.
    """
    timeout = int(os.getenv("SERVICE_STARTUP_TIMEOUT", DEFAULT_TIMEOUT))
    poll_interval = int(os.getenv("SERVICE_POLL_INTERVAL", DEFAULT_POLL_INTERVAL))
    
    print(f"\n[Readiness] Waiting for {len(SERVICES)} services to become healthy (timeout: {timeout}s)...")
    
    start_time = time.time()
    pending_services = list(SERVICES)
    
    while pending_services:
        if time.time() - start_time > timeout:
            failed_services = []
            for service in pending_services:
                url = f"{BASE_URL}/{service}/health"
                failed_services.append(f"Service '{service}' did not become healthy within {timeout} seconds.\nHealth endpoint: {url}")
            
            error_msg = "\n\n".join(failed_services)
            pytest.exit(f"\n[Readiness] FAILED:\n{error_msg}")

        still_pending = []
        for service in pending_services:
            url = f"{BASE_URL}/{service}/health"
            try:
                # We use a short timeout for the health check itself
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok":
                        elapsed = int(time.time() - start_time)
                        print(f"[Readiness] {service} is healthy after {elapsed} seconds")
                        continue
            except Exception:
                # Ignore connection errors while waiting
                pass
            
            still_pending.append(service)
        
        pending_services = still_pending
        
        if pending_services:
            # Wait before next round of polling
            time.sleep(poll_interval)
    
    total_elapsed = int(time.time() - start_time)
    print(f"[Readiness] All services are healthy after {total_elapsed} seconds.\n")
