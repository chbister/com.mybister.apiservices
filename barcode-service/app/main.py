from __future__ import annotations

import os
import socket
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pyzbar import pyzbar
from PIL import Image, ImageDraw
import io
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "Barcode/QR-Code Recognition Service"
app = FastAPI(title=APP_TITLE)

async def get_image_bytes(file: Optional[UploadFile], file_url: Optional[str]) -> bytes:
    if file:
        return await file.read()
    elif file_url:
        logger.info("Downloading image from %s", file_url)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(file_url, timeout=30.0)
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {response.status_code}")
                return response.content
            except Exception as e:
                logger.error("Download failed: %s", str(e))
                raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'file_url' must be provided.")

class BarcodeResult(BaseModel):
    data: str
    type: str
    rect: dict

class BarcodeResponse(BaseModel):
    results: List[BarcodeResult]
    count: int
    hostname: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "barcode",
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }

@app.post("/v1/barcode", response_model=BarcodeResponse)
async def scan_barcode(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None)
):
    logger.info("Barcode scan request received file=%s, file_url=%s", 
                file.filename if file else "None", file_url)
    
    try:
        contents = await get_image_bytes(file, file_url)
        image = Image.open(io.BytesIO(contents))
        
        # Decode barcodes/QR codes
        decoded_objects = pyzbar.decode(image)
        
        results = []
        for obj in decoded_objects:
            # Filter out false positives: width/height must be > 0
            if obj.rect.width <= 0 or obj.rect.height <= 0:
                logger.info("Skipping false positive barcode (dim=0): %s", obj.data)
                continue
                
            results.append(BarcodeResult(
                data=obj.data.decode("utf-8"),
                type=obj.type,
                rect={
                    "left": obj.rect.left,
                    "top": obj.rect.top,
                    "width": obj.rect.width,
                    "height": obj.rect.height
                }
            ))
            
        return BarcodeResponse(
            results=results,
            count=len(results),
            hostname=socket.gethostname()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Barcode scanning failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Scanning failed: {str(e)}")

@app.post("/v1/barcode/highlight")
async def highlight_barcode(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None)
):
    logger.info("Barcode highlight request received file=%s, file_url=%s", 
                file.filename if file else "None", file_url)
    
    try:
        contents = await get_image_bytes(file, file_url)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Decode barcodes/QR codes
        decoded_objects = pyzbar.decode(image)
        
        # Filter out false positives: width/height must be > 0
        decoded_objects = [obj for obj in decoded_objects if obj.rect.width > 0 and obj.rect.height > 0]
        
        if not decoded_objects:
             # Just return the original if nothing found
             pass
        else:
            draw = ImageDraw.Draw(image)
            
            for obj in decoded_objects:
                # Automatic color adjustment for maximum contrast
                r = obj.rect
                try:
                    crop = image.crop((r.left, r.top, r.left + r.width, r.top + r.height)).convert("L")
                    data = list(crop.getdata())
                    avg_brightness = sum(data) / len(data) if data else 0
                    if avg_brightness < 128:
                        highlight_color = (0, 255, 255) # Cyan
                    else:
                        highlight_color = (0, 0, 255)   # Deep Blue
                except:
                    highlight_color = (0, 191, 255) # Fallback to DeepSkyBlue

                # pyzbar provides 'polygon' which is more accurate for rotated codes
                if obj.polygon and len(obj.polygon) >= 2:
                    points = [(p.x, p.y) for p in obj.polygon]
                    if len(points) == 2:
                        draw.line(points, fill=highlight_color, width=5)
                    else:
                        draw.polygon(points, outline=highlight_color, width=5)
                else:
                    # Fallback to rect
                    draw.rectangle(
                        [r.left, r.top, r.left + r.width, r.top + r.height],
                        outline=highlight_color,
                        width=5
                    )
        
        # Save to buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Barcode highlighting failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Highlighting failed: {str(e)}")
