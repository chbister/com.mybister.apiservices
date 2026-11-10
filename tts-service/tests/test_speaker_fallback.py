"""
Regression tests for the multi-speaker fallback on CPU-only hosts.

Prod runs a CPU-only torch build (see Dockerfile), so `vits_mode` is always True
and voice cloning via `speaker_wav` is disabled by design
(see DUAL_MODEL_IMPLEMENTATION.md). The loaded model is nevertheless XTTS v2,
which is multi-speaker - a synthesis call without any speaker used to raise
"Model is multi-speaker but no `speaker` is provided." and surfaced as HTTP 500
for every request (GlitchTip issues 15/16/17).
"""
import io
import sys
import types
import unittest.mock as mock

import pytest

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

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402

# The singleton is created in the app lifespan, so it must be read from the
# module at test time instead of importing the initial None value.
app = main.app
tts_manager = main.tts_manager


def make_model(multi_speaker=True, speakers=("spk_a", "spk_b")):
    model = mock.MagicMock()
    model.is_multi_speaker = multi_speaker
    model.speakers = list(speakers)
    model.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model.tts_to_file.return_value = None
    return model


@pytest.fixture
def client():
    """Run the app lifespan with a mocked model (CPU-only, no GPU)."""
    with mock.patch.object(main.TTSManager, "load_model", return_value=None):
        with TestClient(app) as test_client:
            manager = main.tts_manager
            original = {
                "model": manager.model,
                "has_gpu": manager.has_gpu,
                "device": manager.device,
                "default_voice": manager.default_voice,
                "default_male_voice": manager.default_male_voice,
                "default_female_voice": manager.default_female_voice,
            }
            manager.model = make_model()
            manager.has_gpu = False  # vits_mode -> no voice cloning
            manager.device = "cpu"
            manager.default_voice = None
            manager.default_male_voice = None
            manager.default_female_voice = None
            try:
                yield test_client
            finally:
                for key, value in original.items():
                    setattr(manager, key, value)


def test_gender_request_without_gpu_returns_200(client):
    # GlitchTip #15: gender=female, CPU-only -> 500 before the fallback existed
    files = {"document": ("doc.ssml", b"<speak>Hallo Welt</speak>")}
    response = client.post("/v1/tts", files=files, data={"gender": "female"})

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert kwargs["speaker"] == "spk_a"
    assert "speaker_wav" not in kwargs


def test_reference_audio_without_gpu_returns_200(client):
    # GlitchTip #16: referenceAudio, cloning disabled on CPU -> 500 before
    files = {
        "document": ("doc.ssml", b"<speak>Hallo Welt</speak>"),
        "referenceAudio": ("ref.wav", b"RIFFfake"),
    }
    response = client.post("/v1/tts", files=files)

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert kwargs["speaker"] == "spk_a"
    assert "speaker_wav" not in kwargs


def test_named_default_voice_is_passed_as_speaker(client):
    # Default voices may be built-in speaker names instead of file paths
    main.tts_manager.default_female_voice = "de_female_01"
    main.tts_manager.model.speakers = ["de_female_01", "de_male_01"]

    files = {"document": ("doc.ssml", b"<speak>Hallo</speak>")}
    response = client.post("/v1/tts", files=files, data={"gender": "female"})

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert kwargs["speaker"] == "de_female_01"
    assert "speaker_wav" not in kwargs


def test_voice_cloning_still_used_when_gpu_available(client):
    main.tts_manager.has_gpu = True
    main.tts_manager.device = "cuda"

    files = {
        "document": ("doc.ssml", b"<speak>Hallo</speak>"),
        "referenceAudio": ("ref.wav", b"RIFFfake"),
    }
    response = client.post("/v1/tts", files=files)

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert kwargs["speaker_wav"].endswith(".wav")


def test_model_not_loaded_returns_503(client):
    main.tts_manager.model = None
    files = {"document": ("doc.ssml", b"<speak>Hallo</speak>")}
    response = client.post("/v1/tts", files=files, data={"gender": "male"})

    assert response.status_code == 503


def test_document_content_is_used(client):
    # The former debug override hardcoded the text and ignored the document
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)

    files = {"document": ("document.pdf", buffer.getvalue())}
    response = client.post("/v1/tts", files=files, data={"gender": "male"})

    # Empty PDF -> empty text -> 400 (was 200 with the hardcoded test text)
    assert response.status_code == 400


def test_single_speaker_model_is_called_without_speaker(client):
    main.tts_manager.model = make_model(multi_speaker=False, speakers=())
    files = {"document": ("doc.ssml", b"<speak>Hallo</speak>")}
    response = client.post("/v1/tts", files=files, data={"gender": "male"})

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert "speaker" not in kwargs
    assert "speaker_wav" not in kwargs


def test_broken_tts_speakers_property_still_falls_back(client):
    """TTS 0.22.0: the `TTS.speakers` property raises AttributeError for XTTS v2."""

    class BrokenSpeakersModel:
        is_multi_speaker = True
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        def __init__(self, names):
            self.tts_to_file = mock.MagicMock(return_value=None)
            manager = mock.MagicMock()
            manager.name_to_id = names  # speaker_manager works, property does not
            self.synthesizer = mock.MagicMock()
            self.synthesizer.tts_model.speaker_manager = manager

        @property
        def speakers(self):
            raise AttributeError("'TTS' object has no attribute 'speakers'")

    main.tts_manager.model = BrokenSpeakersModel(["Alpha Voice", "Beta Voice"])

    files = {"document": ("doc.ssml", b"<speak>Hallo</speak>")}
    response = client.post("/v1/tts", files=files, data={"gender": "female"})

    assert response.status_code == 200
    kwargs = main.tts_manager.model.tts_to_file.call_args.kwargs
    assert kwargs["speaker"] == "Alpha Voice"  # first built-in speaker
