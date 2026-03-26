from __future__ import annotations

import socket
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import qrcode
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "QR-Code Generation Service"
app = FastAPI(title=APP_TITLE)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "qrcode",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.get("/v1/qrcode")
async def generate_qrcode(
    data: str = Query(..., description="The string or URL to encode in the QR code"),
    box_size: int = Query(10, description="Size of each box in pixels"),
    border: int = Query(4, description="Border thickness in boxes"),
    error_correction: str = Query("L", description="Error correction level (L, M, Q, H)")
):
    logger.info("QR-Code generation request received for: %s", data)
    
    ec_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H
    }
    
    ec_level = ec_levels.get(error_correction.upper(), qrcode.constants.ERROR_CORRECT_L)
    
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=ec_level,
            box_size=box_size,
            border=border,
        )
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to buffer
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        logger.error("QR-Code generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
