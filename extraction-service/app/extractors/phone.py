import logging
from typing import List
import phonenumbers
from .base import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)

class PhoneExtractor(BaseExtractor):
    @property
    def mode_name(self) -> str:
        return "phonenumber"

    async def extract(self, text: str, lines: List[str]) -> List[ExtractionResult]:
        results = []
        
        # We can search in the whole text as phonenumbers usually spans single lines or specific patterns
        # Using phonenumbers.PhoneNumberMatcher for finding numbers in text
        try:
            # We don't have a default region, but we can try to guess or just use None for international format
            # Typical OCR results for contacts might have DE numbers (+49)
            for match in phonenumbers.PhoneNumberMatcher(text, "DE"):
                formatted_number = phonenumbers.format_number(
                    match.number, phonenumbers.PhoneNumberFormat.E164
                )
                results.append(ExtractionResult(
                    data=formatted_number,
                    type="PHONENUMBER",
                    confidence=1.0 # phonenumbers is deterministic
                ))
        except Exception as e:
            logger.error(f"Phone extraction failed: {e}")
            
        return results
