import os
import shutil
import time
import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies that are not available in the test environment
sys.modules["TTS"] = MagicMock()
sys.modules["TTS.api"] = MagicMock()
sys.modules["TTS.utils.manage"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["ulid"] = MagicMock()

# PREVENT EARLY INITIALIZATION BY MOCKING ENV VARS
os.environ["TTS_HOME"] = os.path.join(os.getcwd(), "test_tts_cache_unit")
os.environ["TTS_MODEL_NAME"] = "tts_models/multilingual/multi-dataset/xtts_v2"

# Mock FastAPI multipart check
import fastapi.dependencies.utils
fastapi.dependencies.utils.ensure_multipart_is_installed = lambda: None

# Add app to path
sys.path.append(os.path.join(os.getcwd(), "tts-service"))
from app.main import TTSManager

@pytest.fixture
def test_cache_dir(tmp_path):
    cache_dir = tmp_path / "tts_cache"
    cache_dir.mkdir()
    return str(cache_dir)

@pytest.fixture
def tts_manager(test_cache_dir):
    with patch.dict(os.environ, {
        "HF_HOME": test_cache_dir,
        # TTS_HOME must match the tests' expected layout: ModelManager and the
        # TTS API both download to <TTS_HOME>/tts/<model_full_name>.
        "TTS_HOME": test_cache_dir,
        "TTS_MODEL_NAME": "tts_models/multilingual/multi-dataset/xtts_v2",
        "TTS_LOAD_MAX_RETRIES": "1",
        "TTS_LOAD_INITIAL_DELAY": "0.01"
    }):
        # Mock ModelManager to return predictable paths
        with patch("app.main.ModelManager") as MockMM:
            mm_instance = MockMM.return_value
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
            model_path = os.path.join(model_dir, "model.pth")
            config_path = os.path.join(model_dir, "config.json")
            mm_instance.get_local_model_paths.return_value = (model_path, config_path, {})

            manager = TTSManager()
            # Also mock the instance's model_manager just in case
            manager.model_manager = mm_instance
            # Keep the patches active for the whole test: load_model() creates a
            # fresh ModelManager and re-reads TTS_HOME/HF_HOME at runtime.
            # Never hit the network in unit tests: the sample-voice warm-up
            # download is stubbed; tests that care patch _download_file themselves.
            with patch.object(manager, "_download_file"):
                yield manager

def test_load_model_success_warm_cache(tts_manager, test_cache_dir):
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w") as f: f.write("{}")
    with open(os.path.join(model_dir, "model.pth"), "w") as f: f.write("data")
    with open(os.path.join(model_dir, "vocab.json"), "w") as f: f.write("{}")
    with open(os.path.join(model_dir, "speakers_xtts.pth"), "w") as f: f.write("data")

    with patch("app.main.TTS") as MockTTS:
        tts_manager.load_model()
        assert tts_manager.model is not None
        assert MockTTS.called
        # Verify no cleanup was called
        assert os.path.exists(os.path.join(model_dir, "config.json"))

def test_load_model_recovery_corrupted_cache(tts_manager, test_cache_dir):
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir)
    # Missing essential files, but has a .part file
    with open(os.path.join(model_dir, "model.pth.part"), "w") as f: f.write("partial")
    
    with patch("app.main.TTS") as MockTTS:
        # Define side effect to simulate successful download on retry
        def side_effect(*args, **kwargs):
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "config.json"), "w") as f: f.write("{}")
            with open(os.path.join(model_dir, "model.pth"), "w") as f: f.write("data")
            with open(os.path.join(model_dir, "vocab.json"), "w") as f: f.write("{}")
            with open(os.path.join(model_dir, "speakers_xtts.pth"), "w") as f: f.write("data")
            return MagicMock()
        
        MockTTS.side_effect = side_effect
        
        tts_manager.load_model()
        
        assert tts_manager.model is not None
        assert not os.path.exists(os.path.join(model_dir, "model.pth.part"))
        assert os.path.exists(os.path.join(model_dir, "model.pth"))

def test_load_model_retry_on_network_error(tts_manager, test_cache_dir):
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    
    with patch("app.main.TTS") as MockTTS:
        mock_model = MagicMock()
        
        # Simulate network error then success
        # Note: we need to mock the .to(device) call too
        MockTTS.return_value.to.side_effect = [
            Exception("Connection reset by peer"),
            mock_model
        ]
        
        # We need to simulate file creation for _is_cache_valid to pass
        def side_effect(*args, **kwargs):
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "config.json"), "w") as f: f.write("{}")
            with open(os.path.join(model_dir, "model.pth"), "w") as f: f.write("data")
            with open(os.path.join(model_dir, "vocab.json"), "w") as f: f.write("{}")
            with open(os.path.join(model_dir, "speakers_xtts.pth"), "w") as f: f.write("data")
            return MockTTS.return_value
            
        MockTTS.side_effect = side_effect
        
        tts_manager.load_model()
        
        assert tts_manager.model == mock_model
        assert MockTTS.call_count == 2

def test_load_model_fail_fast_after_max_retries(tts_manager):
    with patch("app.main.TTS") as MockTTS:
        MockTTS.side_effect = Exception("Persistent error")
        
        with pytest.raises(RuntimeError) as excinfo:
            tts_manager.load_model()
        
        # Max retries is 1 in the fixture, so it tries once + 1 retry = 2 attempts
        assert "failure after 2 retries" in str(excinfo.value)
        assert MockTTS.call_count == 2

def test_load_local_model_success(tts_manager, test_cache_dir):
    snapshot_path = test_cache_dir
    model_path = os.path.join(snapshot_path, "model.pth")
    config_path = os.path.join(snapshot_path, "config.json")
    with open(model_path, "w") as f: f.write("data")
    with open(config_path, "w") as f: f.write("{}")
    # Also need these for _is_cache_valid
    with open(os.path.join(snapshot_path, "vocab.json"), "w") as f: f.write("{}")
    with open(os.path.join(snapshot_path, "speakers_xtts.pth"), "w") as f: f.write("data")
    
    # Simulate being in local mode by setting model_root (though we'll call load_model which checks it)
    tts_manager.model_root = "/tmp/fake_root" 
    
    with patch("app.main.TTS") as MockTTS:
        mock_instance = MockTTS.return_value
        mock_instance.to.return_value = mock_instance
        
        with patch.object(tts_manager, "_resolve_snapshot", return_value=snapshot_path):
            tts_manager.load_model()
        
        assert tts_manager.model == mock_instance
        # _load_local_model passes the snapshot directory (the folder holding
        # model.pth), not the checkpoint file itself - TTS 0.22 expects a
        # directory for XTTS (Xtts.load_checkpoint joins model.pth inside it).
        MockTTS.assert_called_once_with(model_path=snapshot_path, config_path=config_path)

def test_load_local_model_missing_file(tts_manager, test_cache_dir):
    snapshot_path = test_cache_dir
    # Missing model.pth
    config_path = os.path.join(snapshot_path, "config.json")
    with open(config_path, "w") as f: f.write("{}")
    
    tts_manager.model_root = "/tmp/fake_root"
    
    with patch.object(tts_manager, "_resolve_snapshot", return_value=snapshot_path):
        with pytest.raises(RuntimeError) as excinfo:
            tts_manager.load_model()
    
    assert "Local XTTS model files are missing or invalid" in str(excinfo.value)

def test_load_local_model_initialization_failure(tts_manager, test_cache_dir):
    snapshot_path = test_cache_dir
    model_path = os.path.join(snapshot_path, "model.pth")
    config_path = os.path.join(snapshot_path, "config.json")
    with open(model_path, "w") as f: f.write("data")
    with open(config_path, "w") as f: f.write("{}")
    # Also need these for _is_cache_valid
    with open(os.path.join(snapshot_path, "vocab.json"), "w") as f: f.write("{}")
    with open(os.path.join(snapshot_path, "speakers_xtts.pth"), "w") as f: f.write("data")
    
    tts_manager.model_root = "/tmp/fake_root"
    
    with patch("app.main.TTS") as MockTTS:
        MockTTS.side_effect = Exception("Init error")
        
        with patch.object(tts_manager, "_resolve_snapshot", return_value=snapshot_path):
            with pytest.raises(RuntimeError) as excinfo:
                tts_manager.load_model()
        
        assert "Local XTTS model loading failure" in str(excinfo.value)

def test_resolve_snapshot_via_refs(tts_manager, test_cache_dir):
    model_root = os.path.join(test_cache_dir, "models--coqui--XTTS-v2")
    snapshots_dir = os.path.join(model_root, "snapshots")
    snapshot_hash = "hash123"
    snapshot_path = os.path.join(snapshots_dir, snapshot_hash)
    os.makedirs(snapshot_path)
    os.makedirs(os.path.join(model_root, "refs"))
    
    with open(os.path.join(model_root, "refs", "main"), "w") as f:
        f.write(snapshot_hash)
        
    tts_manager.model_root = model_root
    resolved = tts_manager._resolve_snapshot()
    assert resolved == snapshot_path

def test_resolve_snapshot_via_mtime(tts_manager, test_cache_dir):
    model_root = os.path.join(test_cache_dir, "models--coqui--XTTS-v2")
    snapshots_dir = os.path.join(model_root, "snapshots")
    
    # Create two snapshots
    s1 = os.path.join(snapshots_dir, "s1")
    s2 = os.path.join(snapshots_dir, "s2")
    os.makedirs(s1)
    os.makedirs(s2)
    
    # Set mtime so s2 is newer
    os.utime(s1, (time.time() - 100, time.time() - 100))
    os.utime(s2, (time.time(), time.time()))
    
    tts_manager.model_root = model_root
    resolved = tts_manager._resolve_snapshot()
    assert resolved == s2

def test_load_model_with_root_and_relative_voices(tts_manager, test_cache_dir):
    model_root = os.path.join(test_cache_dir, "models--coqui--XTTS-v2")
    snapshot_path = os.path.join(model_root, "snapshots", "hash1")
    os.makedirs(snapshot_path)
    
    with open(os.path.join(snapshot_path, "model.pth"), "w") as f: f.write("data")
    with open(os.path.join(snapshot_path, "config.json"), "w") as f: f.write("{}")
    # Also needed for _is_cache_valid
    with open(os.path.join(snapshot_path, "vocab.json"), "w") as f: f.write("{}")
    with open(os.path.join(snapshot_path, "speakers_xtts.pth"), "w") as f: f.write("data")
    
    # Voice file
    voice_dir = os.path.join(snapshot_path, "samples")
    os.makedirs(voice_dir)
    voice_path = os.path.join(voice_dir, "voice.wav")
    with open(voice_path, "w") as f: f.write("audio")
    
    tts_manager.model_root = model_root
    tts_manager.default_male_voice = "samples/voice.wav"
    tts_manager.default_female_voice = "samples/voice.wav"
    
    with patch("app.main.TTS") as MockTTS:
        mock_instance = MockTTS.return_value
        mock_instance.to.return_value = mock_instance
        
        tts_manager.load_model()
        
        expected_config_path = os.path.join(snapshot_path, "config.json")
        
        # XTTS is loaded from the snapshot directory (TTS 0.22 joins model.pth
        # inside the given directory), not from the checkpoint file itself.
        MockTTS.assert_called_once_with(
            model_path=snapshot_path,
            config_path=expected_config_path
        )
        assert tts_manager.default_male_voice == voice_path
        assert tts_manager.default_female_voice == voice_path

def test_resolve_snapshot_missing_root(tts_manager):
    # A missing TTS_MODEL_ROOT is not fatal: _resolve_snapshot signals the
    # fallback to the repository download strategy by returning None.
    tts_manager.model_root = "/nonexistent/path"
    assert tts_manager._resolve_snapshot() is None

def test_resolve_snapshot_missing_snapshots_dir(tts_manager, test_cache_dir):
    # Root exists but is unusable (no HF layout) -> still a hard error,
    # because that indicates a corrupted/partial cache, not a missing download.
    model_root = os.path.join(test_cache_dir, "models--coqui--XTTS-v2")
    os.makedirs(model_root)
    tts_manager.model_root = model_root
    with pytest.raises(RuntimeError) as excinfo:
        tts_manager._resolve_snapshot()
    assert "snapshots directory missing" in str(excinfo.value)

def test_load_model_falls_back_when_model_root_missing(tts_manager, test_cache_dir):
    # TTS_MODEL_ROOT points to a directory that does not exist (fresh host /
    # hf_cache volume without the prefetched model): load_model() must fall
    # back to the repository download strategy instead of raising.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    for fname in ("model.pth", "config.json", "vocab.json", "speakers_xtts.pth"):
        with open(os.path.join(model_dir, fname), "w") as f: f.write("data")

    tts_manager.model_root = "/nonexistent/tts/model/root"

    def fake_download(url, target, timeout=60):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(b"RIFF")

    with patch("app.main.TTS") as MockTTS, \
            patch.object(tts_manager, "_download_file", side_effect=fake_download):
        mock_instance = MockTTS.return_value
        mock_instance.to.return_value = mock_instance

        tts_manager.load_model()

    assert tts_manager.model is not None
    MockTTS.assert_called_once_with(model_name=model_name)

def test_fallback_warms_up_sample_voices(tts_manager, test_cache_dir):
    # Repository fallback must download all 8 XTTS sample wavs into
    # <model_dir>/samples/ and resolve relative voice config against it.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    for fname in ("model.pth", "config.json", "vocab.json", "speakers_xtts.pth"):
        with open(os.path.join(model_dir, fname), "w") as f: f.write("data")

    tts_manager.model_root = "/nonexistent/tts/model/root"
    tts_manager.default_voice = "samples/de_sample.wav"

    downloaded = []

    def fake_download(url, target, timeout=60):
        downloaded.append(url)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(b"RIFF")

    with patch("app.main.TTS") as MockTTS, \
            patch.object(tts_manager, "_download_file", side_effect=fake_download):
        tts_manager.load_model()

    assert tts_manager.model is not None
    # All 8 language samples requested
    assert len(downloaded) == 8
    for name in ("de_sample.wav", "en_sample.wav", "es_sample.wav", "fr_sample.wav",
                 "ja-sample.wav", "pt_sample.wav", "tr_sample.wav", "zh-cn-sample.wav"):
        assert os.path.exists(os.path.join(model_dir, "samples", name)), name

    # Relative TTS_DEFAULT_VOICE resolves against the repository model dir
    expected_voice = os.path.join(model_dir, "samples", "de_sample.wav")
    assert tts_manager.default_male_voice == expected_voice
    assert tts_manager.default_female_voice == expected_voice

def test_fallback_sample_download_failure_is_non_fatal(tts_manager, test_cache_dir):
    # A failing samples download (offline, HF hiccup) must not prevent the
    # service from loading: voices fall back to speaker names.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    for fname in ("model.pth", "config.json", "vocab.json", "speakers_xtts.pth"):
        with open(os.path.join(model_dir, fname), "w") as f: f.write("data")

    tts_manager.model_root = "/nonexistent/tts/model/root"
    tts_manager.default_voice = "samples/de_sample.wav"

    with patch("app.main.TTS") as MockTTS, \
            patch.object(tts_manager, "_download_file", side_effect=OSError("network down")):
        tts_manager.load_model()

    assert tts_manager.model is not None
    # Voice could not resolve to a file -> falls back to built-in speaker name
    assert tts_manager.default_male_voice == "de_male_01"
    assert tts_manager.default_female_voice == "de_female_01"
    # No empty/leftover sample files
    samples_dir = os.path.join(model_dir, "samples")
    if os.path.isdir(samples_dir):
        assert not [f for f in os.listdir(samples_dir) if f.endswith(".part")]

def test_fallback_keeps_bare_speaker_name_voice(tts_manager, test_cache_dir):
    # A bare speaker name (no path separator) must survive voice validation
    # with snapshot_path set - it is not a file to resolve.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    model_dir = os.path.join(test_cache_dir, "tts", model_name.replace("/", "--"))
    os.makedirs(model_dir, exist_ok=True)
    for fname in ("model.pth", "config.json", "vocab.json", "speakers_xtts.pth"):
        with open(os.path.join(model_dir, fname), "w") as f: f.write("data")

    tts_manager.model_root = "/nonexistent/tts/model/root"
    tts_manager.default_voice = "my_custom_speaker"

    with patch("app.main.TTS") as MockTTS, \
            patch.object(tts_manager, "_download_file"):
        tts_manager.load_model()

    assert tts_manager.model is not None
    assert tts_manager.default_male_voice == "my_custom_speaker"
    assert tts_manager.default_female_voice == "my_custom_speaker"
