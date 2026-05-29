import pytest
import asyncio
from app.extractors.name import NameExtractor
from app.extractors.phone import PhoneExtractor
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_phone_extractor():
    extractor = PhoneExtractor()
    text = "Call me at +49 176 29524296 or 0176 29524296"
    lines = ["Call me at +49 176 29524296", "or 0176 29524296"]
    
    results = await extractor.extract(text, lines)
    
    assert len(results) >= 1
    assert results[0].type == "PHONENUMBER"
    assert results[0].data == "+4917629524296"

@pytest.mark.asyncio
async def test_name_extractor_noise():
    # We mock the pipeline to avoid downloading models during tests
    with patch("app.extractors.name.pipeline") as mock_pipeline:
        mock_ner = MagicMock()
        mock_pipeline.return_value = mock_ner
        
        # Admin should be rejected by noise filter before calling NER
        extractor = NameExtractor()
        results = await extractor.extract("Admin", ["Admin"])
        
        assert len(results) == 0
        mock_ner.assert_not_called()

@pytest.mark.asyncio
async def test_name_extractor_success():
    with patch("app.extractors.name.pipeline") as mock_pipeline:
        mock_ner = MagicMock()
        mock_ner.return_value = [{"entity_group": "PER", "word": "Andre Ritter", "score": 0.98}]
        mock_pipeline.return_value = mock_ner
        
        extractor = NameExtractor()
        results = await extractor.extract("Andre Ritter", ["Andre Ritter"])
        
        assert len(results) == 1
        assert results[0].data == "Andre Ritter"
        assert results[0].type == "NAME"
        assert results[0].confidence == 1.0 # Fast-tracked

@pytest.mark.asyncio
async def test_name_extractor_mutation_protection():
    with patch("app.extractors.name.pipeline") as mock_pipeline:
        mock_ner = MagicMock()
        # Mocking a mutation that should be prevented
        mock_ner.return_value = [{"entity_group": "PER", "word": "min Scheffel", "score": 0.98}]
        mock_pipeline.return_value = mock_ner
        
        extractor = NameExtractor()
        # Even if NER returns "min Scheffel", fast-track or cleanup should preserve "Armin Scheffel"
        results = await extractor.extract("Armin Scheffel", ["Armin Scheffel"])
        
        assert len(results) == 1
        assert results[0].data == "Armin Scheffel"
        assert results[0].type == "NAME"

@pytest.mark.asyncio
async def test_name_extractor_konrad():
    extractor = NameExtractor()
    results = await extractor.extract("Konrad", ["Konrad"])
    assert len(results) == 1
    assert results[0].data == "Konrad"
