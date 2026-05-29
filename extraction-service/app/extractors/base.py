from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class ExtractionResult(BaseModel):
    data: str
    type: str
    confidence: float

class BaseExtractor(ABC):
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """The identifier for this extractor mode (e.g., 'name', 'phonenumber')."""
        pass

    @abstractmethod
    async def extract(self, text: str, lines: List[str]) -> List[ExtractionResult]:
        """Extract entities from the given text and lines."""
        pass

    def preload(self):
        """Preload models or resources if needed."""
        pass
