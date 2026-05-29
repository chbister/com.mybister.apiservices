import os
import numpy as np
import logging
from PIL import Image
from .base import OCREngine

logger = logging.getLogger(__name__)

class PaddleOCREngine(OCREngine):
    def __init__(self, lang: str = 'latin', config: str = None):
        super().__init__(lang=lang or 'latin', config=config)
        self.reader = None
        # Initialize reader immediately for warm-up support
        self._get_reader()

    def _get_reader(self):
        try:
            from paddleocr import PaddleOCR
            import paddleocr
        except ImportError:
            raise RuntimeError("paddleocr is not installed")
        
        if self.reader is None:
            # Default lightweight profile for screenshots
            profile = os.getenv("OCR_PROFILE", "screenshot").lower()
            
            # Base parameters
            params = {
                "lang": self.lang,
                "use_gpu": os.getenv("OCR_ENABLE_GPU", "false").lower() == "true",
                "show_log": False
            }

            if profile == "screenshot":
                # Lightweight settings optimized for screenshots
                # We avoid heavy document analysis models
                params.update({
                    "use_textline_orientation": os.getenv("OCR_USE_TEXTLINE_ORIENTATION", "false").lower() == "true",
                    "det_model_dir": None, # Use default mobile detection model
                    "rec_model_dir": None, # Use default mobile recognition model
                    "cls_model_dir": None,
                    "enable_mkldnn": True,
                    "cpu_threads": 2,
                    "rec_char_type": "ch", # Default to chinese/latin mixed
                })
                
                # If lang is german/latin, ensure we don't use server models
                if self.lang in ["german", "latin", "en"]:
                     # These are usually defaults for mobile anyway
                     pass
            
            # Migration from deprecated use_angle_cls if present in environment
            env_angle_cls = os.getenv("OCR_USE_ANGLE_CLASSIFIER")
            if env_angle_cls is not None:
                params["use_textline_orientation"] = env_angle_cls.lower() == "true"
                # Ensure we don't carry over the old key if it was somehow set
                params.pop("use_angle_cls", None)

            if self.config:
                # Basic parsing of config string like "use_textline_orientation=True,det_db_thresh=0.3"
                try:
                    for item in self.config.split(','):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            k = k.strip()
                            v = v.strip()
                            # Try to convert to bool or float if applicable
                            if v.lower() == 'true': v = True
                            elif v.lower() == 'false': v = False
                            else:
                                try: v = float(v)
                                except ValueError: pass
                            
                            # Handle deprecated key in config string
                            if k == "use_angle_cls":
                                k = "use_textline_orientation"
                                
                            params[k] = v
                except Exception:
                    pass
            
            # Final safeguard: use_angle_cls and use_textline_orientation are mutually exclusive in 3.5.0
            if "use_textline_orientation" in params:
                params.pop("use_angle_cls", None)
            
            paddle_version = getattr(paddleocr, "__version__", "unknown")
            logger.info("Initializing PaddleOCR (version: %s) with profile: %s, params: %s", 
                        paddle_version, profile, params)
            self.reader = PaddleOCR(**params)
        return self.reader

    def extract_text(self, image: Image.Image) -> str:
        reader = self._get_reader()
        
        img_array = np.array(image.convert('RGB'))
        
        # Use predict instead of ocr if available (preferred in newer versions)
        # However, PaddleOCR's ocr() method is still the high-level API most used.
        # The 'predict' mentioned in the warning might refer to internal routing.
        # Let's try to use ocr() but without deprecated runtime params.
        
        try:
            # result = self.reader.ocr(img_array)
            # Some versions might prefer reader.predict
            if hasattr(self.reader, 'ocr'):
                result = self.reader.ocr(img_array)
            else:
                result = self.reader.predict(img_array)
        except Exception as e:
            logger.error("PaddleOCR extraction failed: %s", str(e))
            raise
        
        # PaddleOCR returns a list of lists of [box, (text, confidence)]
        texts = []
        if result:
            # Handle different return formats between versions if necessary
            # Standard: [[ [box], (text, conf) ], ...]
            # Sometimes it's wrapped in another list for batching: [[[ [box], (text, conf) ]]]
            
            if len(result) > 0 and isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
                # Wrapped in extra list (batch)
                for line in result:
                    if line:
                        for res in line:
                            if isinstance(res, list) and len(res) > 1 and isinstance(res[1], tuple):
                                texts.append(res[1][0])
            else:
                # Single list
                for res in result:
                    if isinstance(res, list) and len(res) > 1 and isinstance(res[1], tuple):
                        texts.append(res[1][0])
        
        return "\n".join(texts)
