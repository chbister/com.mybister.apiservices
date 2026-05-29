try:
    import pytesseract
except ImportError:
    pytesseract = None
from PIL import Image
from .base import OCREngine

class TesseractEngine(OCREngine):
    def extract_text(self, image: Image.Image) -> str:
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed")
        
        kwargs = {}
        if self.lang:
            kwargs['lang'] = self.lang
        if self.config:
            kwargs['config'] = self.config
            
        return pytesseract.image_to_string(image, **kwargs)
