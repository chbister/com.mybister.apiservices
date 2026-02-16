#!/bin/sh
set -e

# Optional sanity checks (won't fail container if missing, but helpful in logs)
command -v exiftool >/dev/null 2>&1 && echo "exiftool: ok" || echo "exiftool: missing"
command -v ffprobe >/dev/null 2>&1 && echo "ffprobe: ok" || echo "ffprobe: missing"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
