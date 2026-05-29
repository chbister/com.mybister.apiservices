import logging
import re
from typing import List, Optional
from .base import BaseExtractor, ExtractionResult
from transformers import pipeline
import os

logger = logging.getLogger(__name__)

class NameExtractor(BaseExtractor):
    def __init__(self):
        self.model_id = os.getenv("NAME_MODEL_ID", "Davlan/xlm-roberta-base-ner-hrl")
        self.ner_pipe = None
        # UI/Noise patterns to reject
        self.noise_patterns = [
            "admin",
            "alle anzeigen",
            "zu favoriten",
            "mitgliedslabel hinzufügen",
            "online",
            "zuletzt online",
            "nachricht",
            "anruf",
            "videoanruf",
            "profil",
            "info"
        ]
        # Common name patterns (simplified)
        self.name_pattern = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$")

    @property
    def mode_name(self) -> str:
        return "name"

    def preload(self):
        if self.ner_pipe is None:
            logger.info(f"Loading Name NER model: {self.model_id}")
            self.ner_pipe = pipeline("ner", model=self.model_id, device=-1, aggregation_strategy="simple")

    def _is_valid_name_structure(self, text: str) -> bool:
        """Checks if the text already looks like a valid name."""
        return bool(self.name_pattern.match(text))

    def _clean_candidate(self, name: str, original_line: str) -> str:
        """
        Non-destructive cleanup of name candidates.
        Ensures name matches original line if it was mutated by NER.
        """
        logger.debug(f"candidate_raw={name}")
        
        # 1. Basic cleanup of NER artifacts
        cleaned = name.replace(" ", " ").strip()
        logger.debug(f"candidate_after_basic_cleanup={cleaned}")

        # 2. Try to find the exact case-sensitive match in the original line
        # This prevents "min Scheffel" if the original line had "Armin Scheffel"
        if cleaned in original_line:
            # If it's a substring, check if it's a partial word mutation
            # e.g., line="Armin", cleaned="min"
            start_idx = original_line.find(cleaned)
            # If it looks like a suffix of a word in the original, we might want to expand it
            # But safer is to check if the original line itself is a valid name
            if self._is_valid_name_structure(original_line):
                logger.debug(f"candidate_match_original_structure=true line={original_line}")
                return original_line

        # 3. Trim punctuation only
        cleaned = cleaned.strip(".,!?- ")
        logger.debug(f"candidate_after_punctuation_trim={cleaned}")
        
        return cleaned

    async def extract(self, text: str, lines: List[str]) -> List[ExtractionResult]:
        if self.ner_pipe is None:
            self.preload()
        
        results = []
        # Process line by line as it's often better for OCR results
        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            # Simple noise check
            if line.lower() in self.noise_patterns:
                continue

            # Check if line itself is already a very strong candidate
            if self._is_valid_name_structure(line):
                logger.debug(f"candidate_fast_track={line} accepted=true")
                results.append(ExtractionResult(
                    data=line,
                    type="NAME",
                    confidence=1.0 # High confidence for fast-track
                ))
                continue

            entities = self.ner_pipe(line)
            for entity in entities:
                # PER = Person
                if entity["entity_group"] == "PER":
                    raw_name = entity["word"]
                    final_name = self._clean_candidate(raw_name, line)
                    
                    logger.debug(f"candidate_after_ner={raw_name} candidate_final={final_name} accepted=true")
                    
                    if len(final_name) > 2:
                        results.append(ExtractionResult(
                            data=final_name,
                            type="NAME",
                            confidence=round(float(entity["score"]), 2)
                        ))
        
        return results
