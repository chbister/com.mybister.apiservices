from __future__ import annotations

import os
import json
import logging
import socket
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "Callback Tester Service"
STORAGE_DIR = os.getenv("CALLBACK_STORAGE_DIR", "/srv/callbacks")

os.makedirs(STORAGE_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "callback-tester",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.post("/listen")
async def listen(request: Request):
    """
    Catch-all endpoint for any POST request.
    Stores the payload and metadata into a file.
    """
    try:
        body = await request.json()
    except:
        body = await request.body()
        body = {"raw_body": str(body)}

    # Use job_id if present, else timestamp
    job_id = body.get("job_id", f"raw-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}")
    filename = f"{job_id}.json"
    filepath = os.path.join(STORAGE_DIR, filename)

    data_to_store = {
        "received_at": datetime.utcnow().isoformat(),
        "headers": dict(request.headers),
        "payload": body,
        "hostname": socket.gethostname()
    }

    with open(filepath, "w") as f:
        json.dump(data_to_store, f, indent=2)

    logger.info("Callback received and stored: %s", filename)
    
    return {"status": "received", "file": filename}

@app.get("/list")
def list_callbacks():
    """Returns a list of received callback files."""
    return {"files": os.listdir(STORAGE_DIR)}
