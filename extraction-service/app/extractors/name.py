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
        # UI/Noise patterns to reject (generic labels)
        self.noise_patterns = {
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
            "info",
            "suchen",
            "personen"
        }
        # Common name patterns (simplified)
        self.name_pattern = re.compile(r"^[A-Z][a-zà-ÿ]+(?:\s[A-Z][a-zà-ÿ]+)*$", re.UNICODE)

    @property
    def mode_name(self) -> str:
        return "name"

    def preload(self):
        if self.ner_pipe is None:
            logger.info(f"Loading Name NER model: {self.model_id}")
            self.ner_pipe = pipeline("ner", model=self.model_id, device=-1, aggregation_strategy=None)

    def _aggregate_entities(self, entities: List[dict], original_text: str) -> List[dict]:
        """
        Aggregates raw NER tokens into contiguous entity spans.
        Handles subwords and adjacent tokens belonging to the same entity type.
        """
        if not entities:
            return []

        aggregated = []
        current_group = None

        for entity in entities:
            tag = entity["entity"]
            # We care about PERSON entities (B-PER, I-PER)
            if not tag.endswith("-PER"):
                if current_group:
                    aggregated.append(current_group)
                    current_group = None
                continue

            # Check if this token is contiguous with the previous one
            is_contiguous = False
            if current_group:
                # Same entity type (PER)
                # AND adjacent or very close (e.g., whitespace in between)
                # xlm-roberta offsets are usually good.
                prev_end = current_group["end"]
                curr_start = entity["start"]
                
                # If they are adjacent or separated only by whitespace/punctuation
                # we group them. The model often mis-labels B-PER/I-PER.
                gap = original_text[prev_end:curr_start]
                if curr_start >= prev_end and (curr_start - prev_end <= 1 or not gap.strip()):
                    is_contiguous = True

            if is_contiguous:
                current_group["end"] = entity["end"]
                current_group["scores"].append(entity["score"])
            else:
                if current_group:
                    aggregated.append(current_group)
                current_group = {
                    "start": entity["start"],
                    "end": entity["end"],
                    "scores": [entity["score"]]
                }
        
        if current_group:
            aggregated.append(current_group)

        # Finalize groups
        results = []
        for group in aggregated:
            group["score"] = sum(group["scores"]) / len(group["scores"])
            group["word"] = original_text[group["start"]:group["end"]]
            results.append(group)
            
        return results

    def _is_valid_name_structure(self, text: str) -> bool:
        """Checks if the text already looks like a valid name."""
        return bool(self.name_pattern.match(text))

    def _clean_candidate(self, name: str, original_line: str) -> str:
        """
        Non-destructive cleanup of name candidates.
        Ensures name matches original line if it was mutated by NER.
        """
        logger.debug(f"candidate_raw='{name}' line='{original_line}'")
        
        # 1. Basic cleanup of NER artifacts and OCR noise at boundaries
        # Strip common OCR noise characters but keep letters (including Unicode)
        cleaned = name.strip(" ~!@#$%^&*()_+={}|[]\\:\";'<>?,./")
        
        # 2. If the cleaned name is part of a word in the original line,
        # we try to expand it to the full word if it's mostly covered.
        # e.g. "Oli4" -> NER might say "Oli" is PER.
        words = original_line.split()
        for word in words:
            # Strip noise from word for comparison
            word_clean = word.strip("~!@#$%^&*()_+={}|[]\\:\";'<>?,./")
            if cleaned in word_clean and len(cleaned) >= 2:
                # If we have something like "Oli" from "Oli4", and "Oli" is a significant part
                # or if the word starts with the cleaned name.
                if word_clean.startswith(cleaned):
                    # Check if the remaining part is just noise (like '4' in 'Oli4')
                    suffix = word_clean[len(cleaned):]
                    if not suffix or not any(c.isalpha() for c in suffix):
                        logger.debug(f"expanding '{cleaned}' to word '{word_clean}' from '{word}'")
                        cleaned = cleaned # Keep cleaned, but we could return word_clean if we wanted "Oli"
                        # Actually for Oli4 -> Oli is expected.
                        break

        # 3. Final check: if the original line (after noise stripping) matches the pattern, prefer it.
        line_clean = original_line.strip(" ~!@#$%^&*()_+={}|[]\\:\";'<>?,./")
        
        # Also try stripping digits for the fast-track check if it helps
        line_no_digits = re.sub(r"\d+", "", line_clean).strip()
        if self._is_valid_name_structure(line_clean):
            logger.debug(f"candidate_match_original_structure=true line_clean='{line_clean}'")
            return line_clean
        elif self._is_valid_name_structure(line_no_digits) and len(line_no_digits) >= 3:
            logger.debug(f"candidate_match_original_structure_no_digits=true line_no_digits='{line_no_digits}'")
            return line_no_digits
        
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
                logger.debug(f"skipping_noise_line='{line}'")
                continue

            # Check if line itself is already a very strong candidate
            # Strip noise for the fast-track check
            line_clean = line.strip(" ~!@#$%^&*()_+={}|[]\\:\";'<>?,./")
            line_no_digits = re.sub(r"\d+", "", line_clean).strip()
            if self._is_valid_name_structure(line_clean):
                logger.debug(f"candidate_fast_track='{line_clean}' accepted=true")
                results.append(ExtractionResult(
                    data=line_clean,
                    type="NAME",
                    confidence=1.0 # High confidence for fast-track
                ))
                continue
            elif self._is_valid_name_structure(line_no_digits) and len(line_no_digits) >= 3:
                logger.debug(f"candidate_fast_track_no_digits='{line_no_digits}' accepted=true")
                results.append(ExtractionResult(
                    data=line_no_digits,
                    type="NAME",
                    confidence=0.9 # Slightly lower confidence as we stripped digits
                ))
                continue

            raw_entities = self.ner_pipe(line)
            logger.debug(f"line='{line}' raw_entities={raw_entities}")
            
            entities = self._aggregate_entities(raw_entities, line)
            logger.debug(f"line='{line}' aggregated_entities={entities}")
            
            for entity in entities:
                raw_name = entity["word"]
                final_name = self._clean_candidate(raw_name, line)
                
                logger.debug(f"candidate_after_ner='{raw_name}' final='{final_name}' score={entity['score']}")
                
                if len(final_name) >= 2:
                    results.append(ExtractionResult(
                        data=final_name,
                        type="NAME",
                        confidence=round(float(entity["score"]), 2)
                    ))
        
        return results
