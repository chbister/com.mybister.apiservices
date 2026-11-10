import pytest
from fastapi.testclient import TestClient
import io
import sys
import types
import unittest.mock as mock

# Stub the heavy ML dependencies when they are not installed (local test run);
# inside the service image the real packages are used.
try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

try:
    from TTS.api import TTS  # noqa: F401
    from TTS.utils.manage import ModelManager  # noqa: F401
except Exception:
    tts_stub = types.ModuleType("TTS")
    tts_api = types.ModuleType("TTS.api")
    tts_api.TTS = mock.MagicMock()
    tts_utils = types.ModuleType("TTS.utils")
    tts_manage = types.ModuleType("TTS.utils.manage")
    tts_manage.ModelManager = mock.MagicMock()
    sys.modules.setdefault("TTS", tts_stub)
    sys.modules.setdefault("TTS.api", tts_api)
    sys.modules.setdefault("TTS.utils", tts_utils)
    sys.modules.setdefault("TTS.utils.manage", tts_manage)

import app.main as main  # noqa: E402

# The singleton is created in the app lifespan, so it must be read from the
# module at test time instead of importing the initial None value.
app = main.app


@pytest.fixture
def client():
    """Run the lifespan, install a mocked multi-speaker model, expose client."""
    with mock.patch.object(main.TTSManager, "load_model", return_value=None):
        with TestClient(app) as test_client:
            manager = main.tts_manager
            original = {
                "model": manager.model,
                "default_male_voice": manager.default_male_voice,
                "default_female_voice": manager.default_female_voice,
                "has_gpu": manager.has_gpu,
            }
            model = mock.MagicMock()
            model.is_multi_speaker = True
            model.speakers = ["de_male_01", "de_female_01"]
            model.tts_to_file.return_value = None
            manager.model = model
            manager.default_male_voice = "de_male_01"
            manager.default_female_voice = "de_female_01"
            manager.has_gpu = False  # prod runs CPU-only: no voice cloning
            try:
                yield test_client
            finally:
                for key, value in original.items():
                    setattr(manager, key, value)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "tts-service"


def test_ac1_voice_cloning(client):
    # AC1 – Voice Cloning: document + referenceAudio
    with mock.patch.object(main.tts_manager, "synthesize", return_value=b"FAKE_AUDIO"):
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>"),
            "referenceAudio": ("ref.wav", b"fake audio content")
        }
        response = client.post("/v1/tts", files=files)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == b"FAKE_AUDIO"


def test_ac2_default_male_voice(client):
    # AC2 – Default Male Voice: document + gender=male
    with mock.patch.object(main.tts_manager, "synthesize", return_value=b"FAKE_AUDIO") as mock_synth:
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>")
        }
        data = {"gender": "male"}
        response = client.post("/v1/tts", files=files, data=data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        mock_synth.assert_called_once()
        args, kwargs = mock_synth.call_args
        assert kwargs["speaker"] == main.tts_manager.default_male_voice
        assert kwargs["speaker_wav"] is None


def test_ac3_default_female_voice(client):
    # AC3 – Default Female Voice: document + gender=female
    with mock.patch.object(main.tts_manager, "synthesize", return_value=b"FAKE_AUDIO") as mock_synth:
        files = {
            "document": ("doc.ssml", b"<speak>Hello world</speak>")
        }
        data = {"gender": "female"}
        response = client.post("/v1/tts", files=files, data=data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        mock_synth.assert_called_once()
        args, kwargs = mock_synth.call_args
        assert kwargs["speaker"] == main.tts_manager.default_female_voice
        assert kwargs["speaker_wav"] is None


def test_ac4_unsupported_format(client):
    # Unsupported document format
    files = {
        "document": ("doc.txt", b"Hello world")
    }
    data = {"gender": "male"}
    response = client.post("/v1/tts", files=files, data=data)

    assert response.status_code == 400
    assert "Unsupported document format" in response.json()["detail"]


def test_ac5_invalid_ssml(client):
    files = {
        "document": ("doc.ssml", b"Invalid SSML")
    }
    data = {"gender": "male"}
    response = client.post("/v1/tts", files=files, data=data)

    assert response.status_code == 400
    assert "Invalid SSML format" in response.json()["detail"]


def test_ac6_mp3_output(client):
    with mock.patch.object(main.tts_manager, "synthesize", return_value=b"FAKE_AUDIO"):
        files = {
            "document": ("doc.ssml", b"<speak>Hello</speak>")
        }
        data = {"gender": "male", "outputFormat": "mp3"}
        response = client.post("/v1/tts", files=files, data=data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"


def test_ac7_missing_voice_info(client):
    files = {
        "document": ("doc.ssml", b"<speak>Hello</speak>")
    }
    response = client.post("/v1/tts", files=files)
    assert response.status_code == 400
    assert "Either referenceAudio or gender must be provided" in response.json()["detail"]


def test_ac8_pdf_processing(client):
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
