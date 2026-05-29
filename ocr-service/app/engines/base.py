from abc import ABC, abstractmethod
from PIL import Image

class OCREngine(ABC):
    def __init__(self, lang: str = None, config: str = None):
        self.lang = lang
        self.config = config

    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        pass
