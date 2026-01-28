import json
import os
import re
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

app = FastAPI(title="asr-vibevoice", version="1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/VibeVoice-ASR")

def _extract_json_from_text(s: str) -> dict:
    # greedy: nimm das erste {...} als JSON (funktioniert oft bei CLI tools)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in stdout/stderr.")
    return json.loads(m.group(0))

@app.get("/health")
def health():
    return {"ok": True, "model_path": MODEL_PATH}

@app.post("/v1/asr")
async def asr(
    file: UploadFile = File(...),
    # kompatibel zu deinem Whisper-Service:
    language: str = Form("auto"),
    timestamps: str = Form("true"),
):
    # VibeVoice braucht i.d.R. kein language setting; wir akzeptieren es nur API-kompatibel
    # timestamps wird ebenfalls i.d.R. im Output vorhanden sein

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, f"input{suffix}")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        cmd = [
            "python",
            "/srv/VibeVoice/demo/vibevoice_asr_inference_from_file.py",
            "--model_path", MODEL_PATH,
            "--audio_files", in_path,
        ]

        # ggf. stdout/stderr zusammen, damit wir JSON nicht verpassen
        p = subprocess.run(cmd, capture_output=True, text=True)

        if p.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "vibevoice inference failed",
                    "stderr": p.stderr[-2000:],
                    "stdout": p.stdout[-2000:],
                },
            )

        # 1) Versuch: JSON aus stdout
        raw = None
        try:
            raw = _extract_json_from_text(p.stdout)
        except Exception:
            # 2) Versuch: JSON aus stderr (manche tools loggen nach stderr)
            try:
                raw = _extract_json_from_text(p.stderr)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "could not parse vibevoice output as JSON",
                        "stdout_tail": p.stdout[-2000:],
                        "stderr_tail": p.stderr[-2000:],
                        "hint": "Check what the demo script outputs; we can adapt parsing to file output if needed.",
                    },
                )

        # ---- Normalisierung (je nachdem wie VibeVoice JSON aussieht) ----
        # Hier mache ich ein defensives Mapping. Wir behalten das Rohresultat zusätzlich.
        text = raw.get("text") or raw.get("transcript") or ""
        segments = raw.get("segments") or raw.get("chunks") or []

        return {
            "backend": "vibevoice",
            "language": language,
            "text": text,
            "segments": segments,
            "raw": raw,
        }
