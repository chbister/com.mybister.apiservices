try:
    import easyocr
except ImportError:
    easyocr = None
import numpy as np
from PIL import Image
from .base import OCREngine

class EasyOCREngine(OCREngine):
    def __init__(self, lang: str = 'de,en', config: str = None):
        super().__init__(lang=lang, config=config)
        self.reader = None
        # Initialize reader immediately for warm-up support
        self._get_reader()

    def _get_reader(self):
        if easyocr is None:
            raise RuntimeError("easyocr is not installed")
        
        if self.reader is None:
            # EasyOCR expects a list of languages
            langs = [l.strip() for l in self.lang.split(',')] if self.lang else ['de', 'en']
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Initializing EasyOCR with languages: %s", langs)
            self.reader = easyocr.Reader(langs)
        return self.reader

    def extract_text(self, image: Image.Image) -> str:
        reader = self._get_reader()
        
        # EasyOCR expects numpy array or file path
        img_array = np.array(image.convert('RGB'))
        results = reader.readtext(img_array, detail=0)
        return "\n".join(results)
