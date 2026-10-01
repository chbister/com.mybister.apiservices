import pytest
from fastapi.testclient import TestClient
import io
import unittest.mock as mock
from app.main import app, tts_manager

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_tts():
    with mock.patch("app.main.TTS") as mocked_tts:
        # Setup mocked model instance
        mock_instance = mocked_tts.return_value
        mock_instance.to.return_value = mock_instance
        mock_instance.is_multi_speaker = True
        mock_instance.tts_to_file.return_value = None
        
        # Mock tts_manager initialization
        with mock.patch("app.main.tts_manager.load_model", side_effect=None):
            tts_manager.model = mock_instance
            yield mocked_tts

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "tts-service"

def test_ac1_voice_cloning():
    # AC1 – Voice Cloning: document + referenceAudio
    # Mock synthesize to return fake audio
    with mock.patch("app.main.tts_manager.synthesize", return_value=b"FAKE_AUDIO"):
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>"),
            "referenceAudio": ("ref.wav", b"fake audio content")
        }
        response = client.post("/v1/tts", files=files)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == b"FAKE_AUDIO"

def test_ac2_default_male_voice():
    # AC2 – Default Male Voice: document + gender=male
    with mock.patch("app.main.tts_manager.synthesize", return_value=b"FAKE_AUDIO") as mock_synth:
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>")
        }
        data = {"gender": "male"}
        response = client.post("/v1/tts", files=files, data=data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        # Verify it called synthesize with the correct speaker name (built-in)
        mock_synth.assert_called_once()
        args, kwargs = mock_synth.call_args
        assert kwargs["speaker"] == tts_manager.default_male_voice
        assert kwargs["speaker_wav"] is None

def test_ac3_default_female_voice():
    # AC3 – Default Female Voice: document + gender=female
    with mock.patch("app.main.tts_manager.synthesize", return_value=b"FAKE_AUDIO") as mock_synth:
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>")
        }
        data = {"gender": "female"}
        response = client.post("/v1/tts", files=files, data=data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        # Verify it called synthesize with the correct speaker name (built-in)
        mock_synth.assert_called_once()
        args, kwargs = mock_synth.call_args
        assert kwargs["speaker"] == tts_manager.default_female_voice
        assert kwargs["speaker_wav"] is None

def test_ac4_unsupported_format():
    # Unsupported document format
    files = {
        "document": ("doc.txt", b"Hello world")
    }
    data = {"gender": "male"}
    response = client.post("/v1/tts", files=files, data=data)
    
    assert response.status_code == 400
    assert "Unsupported document format" in response.json()["detail"]

def test_ac5_invalid_ssml():
    files = {
        "document": ("doc.ssml", b"Invalid SSML")
    }
    data = {"gender": "male"}
    response = client.post("/v1/tts", files=files, data=data)
    
    assert response.status_code == 400
    assert "Invalid SSML format" in response.json()["detail"]

def test_ac6_mp3_output():
    with mock.patch("app.main.tts_manager.synthesize", return_value=b"FAKE_AUDIO"):
        files = {
            "document": ("doc.ssml", b"<speak>Hello</speak>")
        }
        data = {"gender": "male", "outputFormat": "mp3"}
        response = client.post("/v1/tts", files=files, data=data)
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"

def test_ac7_missing_voice_info():
    files = {
        "document": ("doc.ssml", b"<speak>Hello</speak>")
    }
    response = client.post("/v1/tts", files=files)
    assert response.status_code == 400
    assert "Either referenceAudio or gender must be provided" in response.json()["detail"]

def test_ac8_pdf_processing():
    from PyPDF2 import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_buf = io.BytesIO()
    writer.write(pdf_buf)
    pdf_content = pdf_buf.getvalue()
    
    files = {
        "document": ("document.pdf", pdf_content)
    }
    data = {"gender": "male"}
    response = client.post("/v1/tts", files=files, data=data)
    # Blank PDF -> content is empty -> 400
    assert response.status_code == 400
    assert "Document content is empty" in response.json()["detail"]
