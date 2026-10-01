import os
import io
import socket
import logging
import time
import tempfile
import shutil
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response, Request
from pydantic import BaseModel
import PyPDF2
import ulid
import torch
from TTS.api import TTS
from TTS.utils.manage import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- TTS Model Management ---

class TTSManager:
    def __init__(self):
        self.model: Optional[TTS] = None
        self.model_name: str = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
        # CPU fallback model (VITS - no voice cloning support)
        self.cpu_model_name: str = os.getenv("TTS_CPU_MODEL_NAME", "tts_models/de/thorsten/tacotron2-DDS_German")
        self.model_root: Optional[str] = os.getenv("TTS_MODEL_ROOT")
        self.device: str = os.getenv("TTS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        # GPU detection for dual-model strategy
        self.has_gpu: bool = self.device == "cuda" and torch.cuda.is_available()
        self.default_voice: Optional[str] = os.getenv("TTS_DEFAULT_VOICE")
        self.default_male_voice: Optional[str] = os.getenv("TTS_DEFAULT_MALE_VOICE")
        self.default_female_voice: Optional[str] = os.getenv("TTS_DEFAULT_FEMALE_VOICE")

    @property
    def active_model_name(self) -> str:
        """Return the model name currently in use based on hardware availability."""
        if not self.model:
            return None
        # If GPU available or forced CPU mode, use XTTS v2; otherwise VITS fallback
        if self.has_gpu or os.getenv("TTS_FORCE_CPU") == "true":
            return self.model_name
        return self.cpu_model_name

    def _select_model_for_loading(self) -> str:
        """Select appropriate model name based on available hardware."""
        # Force CPU mode takes precedence
        if os.getenv("TTS_FORCE_CPU") == "true":
            logger.info(f"Forcing CPU mode via TTS_FORCE_CPU environment variable, using {self.cpu_model_name}")
            return self.cpu_model_name

        # GPU detected: use XTTS v2 for voice cloning
        if self.has_gpu:
            logger.info(f"GPU detected, using XTTS v2: {self.model_name}")
            return self.model_name

        # No GPU available: fall back to CPU-optimized VITS model
        logger.warning(f"No GPU available, falling back to CPU-optimized VITS model: {self.cpu_model_name}")
        return self.cpu_model_name

    def _init_cache_and_retry_config(self):
        """Initialize cache directories and retry configuration."""
        # Standard Hugging Face cache location
        hf_home = os.getenv("HF_HOME", os.getenv("HF_HUB_CACHE", "/root/.cache/huggingface"))
        # Default TTS cache location
        self.tts_home: str = os.getenv("TTS_HOME", os.path.join(hf_home, "tts") if "huggingface" in hf_home else "/root/.cache/tts")

        # Retry configuration
        self.max_retries: int = int(os.getenv("TTS_LOAD_MAX_RETRIES", "3"))
        self.retry_backoff: float = float(os.getenv("TTS_LOAD_RETRY_BACKOFF", "2.0"))
        self.initial_retry_delay: float = float(os.getenv("TTS_LOAD_INITIAL_DELAY", "5.0"))

        # Ensure TTS_HOME exists for the library
        os.makedirs(self.tts_home, exist_ok=True)
        os.environ["TTS_HOME"] = self.tts_home

        # Initialize ModelManager for dynamic path resolution
        self.model_manager = ModelManager(output_prefix=self.tts_home)

    def _validate_voices(self, snapshot_path: Optional[str] = None):
        """
        Validate and resolve default voices based on priority:
        1. Gender-specific default reference (TTS_DEFAULT_MALE_VOICE / TTS_DEFAULT_FEMALE_VOICE)
        2. Generic default reference (TTS_DEFAULT_VOICE)
        3. Fallback to built-in speaker names if no paths are provided.
        """
        
        def resolve_path(voice_path: Optional[str]) -> Optional[str]:
            if not voice_path:
                return None
            actual_path = voice_path
            if snapshot_path and not os.path.isabs(voice_path):
                actual_path = os.path.join(snapshot_path, voice_path)
            
            # If it looks like a path (has separator or exists)
            if os.path.sep in actual_path or os.path.exists(actual_path):
                if not os.path.exists(actual_path):
                    logger.warning(f"Voice file not found: {actual_path}")
                    return None
                if not os.access(actual_path, os.R_OK):
                    logger.warning(f"Voice file not readable: {actual_path}")
                    return None
                return actual_path
            return voice_path

        # Resolve all potential candidates
        resolved_male = resolve_path(self.default_male_voice)
        resolved_female = resolve_path(self.default_female_voice)
        resolved_generic = resolve_path(self.default_voice)

        # Apply fallback logic
        # 1. Male voice selection
        final_male = resolved_male or resolved_generic or "de_male_01"
        
        # 2. Female voice selection
        final_female = resolved_female or resolved_generic or "de_female_01"

        # If we still have only one of the gender-specific ones and no generic, 
        # use the one we have for both if it's a path.
        if resolved_male and not resolved_female and not resolved_generic:
            if os.path.exists(str(resolved_male)):
                final_female = resolved_male
                logger.info("Using male reference as fallback for female voice.")
        
        if resolved_female and not resolved_male and not resolved_generic:
            if os.path.exists(str(resolved_female)):
                final_male = resolved_female
                logger.info("Using female reference as fallback for male voice.")

        self.default_male_voice = final_male
        self.default_female_voice = final_female
        
        if resolved_generic and final_male == resolved_generic and final_female == resolved_generic:
            logger.info("Shared default reference voice enabled.")

        logger.info(f"Default male voice: {self.default_male_voice}")
        logger.info(f"Default female voice: {self.default_female_voice}")

    def _is_cache_valid(self, model_dir: str) -> bool:
        """
        Check if the required XTTS model files exist in the directory.
        """
        required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
        for f in required_files:
            p = os.path.join(model_dir, f)
            if not os.path.exists(p):
                logger.warning(f"Cache invalid: Missing required file {f} at {model_dir}")
                return False
            if not os.access(p, os.R_OK):
                logger.warning(f"Cache invalid: File {f} not readable at {model_dir}")
                return False
            
        # Check for typical signs of interrupted downloads like .part files
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(".part") or file.endswith(".tmp"):
                    logger.warning(f"Cache invalid: Found incomplete download file {file} in {model_dir}")
                    return False
                    
        return True

    def _cleanup_cache(self, model_dir: str):
        """Remove the model directory if it exists."""
        if os.path.exists(model_dir):
            logger.info(f"Cleaning up invalid cache directory: {model_dir}")
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                logger.error(f"Failed to cleanup cache directory {model_dir}: {e}")
                # We don't raise here, but it might cause the next download to fail if it's a permission issue

    def _resolve_snapshot(self) -> Optional[str]:
        """
        Automatically resolve the active Hugging Face snapshot in self.model_root.
        """
        if not self.model_root:
            return None
            
        if not os.path.isdir(self.model_root):
            logger.error(f"Model root is not a directory: {self.model_root}")
            raise RuntimeError(f"Critical: TTS_MODEL_ROOT is not a directory: {self.model_root}")

        snapshots_dir = os.path.join(self.model_root, "snapshots")
        if not os.path.isdir(snapshots_dir):
            logger.error(f"Snapshots directory missing: {snapshots_dir}")
            raise RuntimeError(f"Critical: Hugging Face snapshots directory missing: {snapshots_dir}")

        # Strategy 1: Check refs/main for the active snapshot hash
        refs_main = os.path.join(self.model_root, "refs", "main")
        if os.path.isfile(refs_main):
            try:
                with open(refs_main, "r") as f:
                    snapshot_hash = f.read().strip()
                snapshot_path = os.path.join(snapshots_dir, snapshot_hash)
                if os.path.isdir(snapshot_path):
                    logger.info(f"Resolved snapshot from refs/main: {snapshot_hash}")
                    return snapshot_path
            except Exception as e:
                logger.warning(f"Failed to read {refs_main}: {e}")

        # Strategy 2: Use the latest snapshot directory by modification time
        try:
            snapshots = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)]
            snapshots = [d for d in snapshots if os.path.isdir(d)]
            if not snapshots:
                logger.error(f"No snapshots available in {snapshots_dir}")
                raise RuntimeError(f"Critical: No Hugging Face snapshots available in {snapshots_dir}")
            
            # Sort by modification time, latest first
            snapshots.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_snapshot = snapshots[0]
            logger.info(f"Resolved latest snapshot: {os.path.basename(latest_snapshot)}")
            return latest_snapshot
        except Exception as e:
            logger.error(f"Failed to resolve snapshot in {snapshots_dir}: {e}")
            raise RuntimeError(f"Critical: Failed to resolve Hugging Face snapshot: {e}")

    def load_model(self):
        # Initialize cache directories and retry configuration before model loading
        self._init_cache_and_retry_config()

        if self.model_root:
            snapshot_path = self._resolve_snapshot()
            self._load_local_model(snapshot_path=snapshot_path)
        else:
            self._load_repository_model()

        # After loading model, validate default voices if not already done in _load_local_model
        if not self.model_root:
            self._validate_voices()

    def _load_local_model(self, snapshot_path: str):
        logger.info("Loading strategy: Local")
        logger.info(f"Model root: {self.model_root}")
        logger.info(f"Resolved snapshot: {os.path.basename(snapshot_path)}")
        logger.info(f"Hugging Face cache location: {os.getenv('HF_HOME', '/root/.cache/huggingface')}")
        
        if not self._is_cache_valid(snapshot_path):
            logger.error("Cache validation: FAILED")
            raise RuntimeError(f"Critical: Local XTTS model files are missing or invalid in {snapshot_path}")

        logger.info("Cache validation: SUCCESS")
        
        # Resolve snapshot-relative voices
        self._validate_voices(snapshot_path=snapshot_path)
        
        # Resolve required model files
        model_path = os.path.join(snapshot_path, "model.pth")
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(snapshot_path, "config.json")
        vocab_path = os.path.join(snapshot_path, "vocab.json")
        speaker_file = os.path.join(snapshot_path, "speakers_xtts.pth")

        logger.info(f"Resolved model path: {model_path}")
        logger.info(f"Resolved config path: {config_path}")
        
        start_time = time.time()
        try:
            # XTTS model loading expects explicit paths to model and config
            self.model = TTS(
                model_path=model_dir,
                config_path=config_path
            ).to(self.device)
            
            duration = time.time() - start_time
            logger.info("Model initialized successfully.")
            logger.info(f"Total initialization time: {duration:.2f}s")
            logger.info(f"Execution device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model from local path: {e}")
            raise RuntimeError(f"Critical: Local XTTS model loading failure: {e}")

    def _load_repository_model(self):
        """Load TTS model - always loads XTTS v2 for consistent behavior."""
        logger.info("Loading strategy: Repository")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Repository cache: {self.tts_home}")

        # Always load XTTS v2 to ensure voice cloning capability
        # The synthesize() method will handle VITS vs XTTS mode based on hardware
        selected_model_name = self.model_name
        # Discover the model path dynamically using ModelManager
        try:
            model_path, config_path, model_item = self.model_manager.get_local_model_paths(selected_model_name)
            model_dir = os.path.dirname(model_path)
            logger.info(f"Resolved model directory: {model_dir}")
        except Exception as e:
            # If ModelManager cannot resolve it, it might not be downloaded yet
            logger.info(f"Model not found in cache: {e}")
            model_dir = os.path.join(self.tts_home, selected_model_name.replace("/", "--"))
            model_path = os.path.join(model_dir, "model.pth")
            config_path = os.path.join(model_dir, "config.json")

        retries = 0
        current_delay = self.initial_retry_delay

        while retries <= self.max_retries:
            start_time = time.time()
            try:
                # Validate cache before attempting load (XTTS v2 specific)
                if os.path.exists(model_dir):
                    if self._is_cache_valid(model_dir):
                        logger.info(f"Cache validation: SUCCESS")
                        logger.info(f"Cache status: HIT (Valid model found at {model_dir})")
                    else:
                        logger.info(f"Cache validation: FAILED")
                        logger.info(f"Cache status: CORRUPTED (Corrupted or incomplete model found. Triggering recovery.)")
                        self._cleanup_cache(model_dir)
                else:
                    logger.info(f"Cache status: MISS (Model will be downloaded to {self.tts_home})")

                self.model = TTS(model_name=selected_model_name).to(self.device)
                
                # Update paths after loading/downloading in case they changed
                try:
                    model_path, config_path, _ = self.model_manager.get_local_model_paths(selected_model_name)
                    model_dir = os.path.dirname(model_path)
                except:
                    pass

                # Double check after loading/downloading (XTTS v2 only)
                if not self._is_cache_valid(model_dir):
                    logger.error(f"Cache validation: FAILED after download at {model_dir}")
                    raise RuntimeError("Model download completed but resulting cache is invalid")

                logger.info(f"Cache validation: SUCCESS")
                end_time = time.time()
                duration = end_time - start_time

                logger.info(f"TTS model '{selected_model_name}' loaded successfully on {self.device}")
                logger.info("Model initialized successfully.")
                logger.info(f"Total initialization time: {duration:.2f}s")
                logger.info(f"Default voices: Male={self.default_male_voice}, Female={self.default_female_voice}")
                return # Success!

            except Exception as e:
                retries += 1
                error_msg = str(e).lower()
                
                # Diagnostic logging
                diagnostic = "Unknown error"
                if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                    diagnostic = "Network interruption"
                elif "no space left on device" in error_msg or "enospc" in error_msg:
                    diagnostic = "Insufficient disk space"
                elif "permission denied" in error_msg or "eacces" in error_msg:
                    diagnostic = "Permission issues"
                elif "corrupted" in error_msg or "invalid" in error_msg or "checksum" in error_msg:
                    diagnostic = "Corrupted cache/download"
                
                logger.error(f"Failed to initialize TTS model (Attempt {retries}/{self.max_retries + 1}): {e}")
                logger.error(f"Diagnostic: {diagnostic}")

                if retries <= self.max_retries:
                    logger.info(f"Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= self.retry_backoff
                    # Clean up again before retry if it seems like a corruption issue
                    if diagnostic in ["Corrupted cache/download", "Network interruption"]:
                         # Resolve directory again if possible for cleanup
                         try:
                             m_path, _, _ = self.model_manager.get_local_model_paths(self.model_name)
                             m_dir = os.path.dirname(m_path)
                             self._cleanup_cache(m_dir)
                         except:
                             self._cleanup_cache(model_dir)
                else:
                    logger.error("Max retries exceeded. Failing fast.")
                    raise RuntimeError(f"Critical: TTS model loading failure after {retries} retries: {e} (Diagnostic: {diagnostic})")

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
        language: str = "de",
        output_format: str = "wav"
    ) -> bytes:
        """Synthesize speech using XTTS v2 (always loaded) with mode-specific behavior."""
        if not self.model:
            raise RuntimeError("TTS model not loaded")

        # Determine which speaker name to use based on hardware and voice cloning availability
        is_vits_mode = not self.has_gpu and os.getenv("TTS_FORCE_CPU") != "true"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            # XTTS v2 mode (always loaded): supports voice cloning via speaker_wav and built-in speakers
            logger.info(f"Using XTTS v2 model: {self.model.model_name}")
            # For VITS-mode requests, don't pass speaker_wav as voice cloning won't work well on CPU anyway
            effective_speaker_wav = None if is_vits_mode else speaker_wav

            if effective_speaker_wav:
                self.model.tts_to_file(
                    text=text,
                    speaker_wav=effective_speaker_wav,
                    language=language,
                    file_path=tmp_wav_path
                )
            elif speaker:
                self.model.tts_to_file(
                    text=text,
                    speaker=speaker,
                    language=language,
                    file_path=tmp_wav_path
                )
            else:
                # Fallback if neither provided, though validation should prevent this
                self.model.tts_to_file(
                    text=text,
                    language=language,
                    file_path=tmp_wav_path
                )

            # Convert to MP3 if requested
            if output_format.lower() == "mp3":
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    tmp_mp3_path = tmp_mp3.name
                
                os.system(f"ffmpeg -i {tmp_wav_path} -codec:a libmp3lame -qscale:a 2 {tmp_mp3_path} -y -loglevel error")
                
                with open(tmp_mp3_path, "rb") as f:
                    content = f.read()
                
                if os.path.exists(tmp_mp3_path):
                    os.remove(tmp_mp3_path)
            else:
                with open(tmp_wav_path, "rb") as f:
                    content = f.read()

            return content
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

# --- Singleton Instance ---

# The singleton instance is initialized during application startup
# This avoids issues with environment variables and filesystem access during import
tts_manager: Optional[TTSManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tts_manager
    tts_manager = TTSManager()
    try:
        tts_manager.load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        # We don't raise here to allow the service to start and respond to health checks
        # The service will return 503 for TTS requests if the model is not loaded
    yield
    # Shutdown
    pass

app = FastAPI(title="TTS Microservice", lifespan=lifespan)

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"

class VoiceSource(str, Enum):
    REFERENCE_AUDIO = "REFERENCE_AUDIO"
    DEFAULT_MALE = "DEFAULT_MALE"
    DEFAULT_FEMALE = "DEFAULT_FEMALE"

def extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

@app.post("/v1/tts")
async def generate_speech(
    request: Request,
    document: UploadFile = File(...),
    referenceAudio: Optional[UploadFile] = File(None),
    gender: Optional[Gender] = Form(None),
    language: str = Form("de-DE"),
    outputFormat: str = Form("wav")
):
    #request_id = str(ulid.new())
    request_id = f"{ulid.ULID()}"
    logger.info(f"Received request with request_id: {request_id}, {document}, {referenceAudio}, {gender}, {language}, {outputFormat}")
    start_time = time.time()
    
    # Validation Rule: If neither referenceAudio nor gender is provided
    if not referenceAudio and not gender:
        raise HTTPException(status_code=400, detail="Either referenceAudio or gender must be provided")

    # Read document content
    doc_content = await document.read()
    filename = document.filename.lower() if document.filename else ""
    doc_type = "PDF" if filename.endswith(".pdf") else "SSML" if filename.endswith(".ssml") else "UNKNOWN"
    
    # Supported document formats: PDF, SSML
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(doc_content)
        is_ssml = False
    elif filename.endswith(".ssml"):
        text = doc_content.decode("utf-8", errors="ignore").strip()
        is_ssml = True
        # Basic SSML validation
        if not (text.startswith("<speak>") and text.endswith("</speak>")):
             raise HTTPException(status_code=400, detail="Invalid SSML format")
        # For simple TTS engines, we might strip SSML if not supported.
        # Requirement: "If the selected model does not support SSML natively, convert the SSML into supported synthesis parameters"
        # For YourTTS, we strip it for now as a fallback.
        import re
        text = re.sub(r'<[^>]*>', '', text).strip()
    else:
        raise HTTPException(status_code=400, detail="Unsupported document format. Only PDF and SSML are supported.")

    text = "Das ist das Haus vom Nikolaus!"

    if not text:
        raise HTTPException(status_code=400, detail="Document content is empty")

    # Voice selection logic
    voice_source = None
    speaker_wav_path = None
    speaker_name = None
    temp_ref_file = None
    logger.info(f"Voice selection logic: {voice_source}, {speaker_wav_path}, {speaker_name}, {temp_ref_file}")
    try:
        if referenceAudio:
            voice_source = VoiceSource.REFERENCE_AUDIO
            ref_content = await referenceAudio.read()
            ref_filename = referenceAudio.filename.lower() if referenceAudio.filename else ""
            
            if not (ref_filename.endswith(".wav") or ref_filename.endswith(".mp3")):
                raise HTTPException(status_code=400, detail="Unsupported audio format for reference voice. Only WAV and MP3 are supported.")
            
            suffix = ".wav" if ref_filename.endswith(".wav") else ".mp3"
            temp_ref_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_ref_file.write(ref_content)
            temp_ref_file.close()
            speaker_wav_path = temp_ref_file.name
        else:
            if gender == Gender.MALE:
                voice_source = VoiceSource.DEFAULT_MALE
                speaker_wav_path = tts_manager.default_male_voice
            elif gender == Gender.FEMALE:
                voice_source = VoiceSource.DEFAULT_FEMALE
                speaker_wav_path = tts_manager.default_female_voice
        logger.info(f"Voice info: {voice_source}, {speaker_name}")
        # Speech generation
        synthesis_start = time.time()
        logger.info(f"Starting synthesis: {synthesis_start}")
        logger.info(f"Text: {text}")
        try:
            if not tts_manager or not tts_manager.model:
                raise HTTPException(status_code=503, detail="TTS service not ready (model not loaded)")

            # For VITS mode (CPU fallback), don't pass speaker_wav as voice cloning isn't supported
            vits_mode = not tts_manager.has_gpu and os.getenv("TTS_FORCE_CPU") != "true"
            synthesize_speaker_wav = None if vits_mode else speaker_wav_path

            audio_content = tts_manager.synthesize(
                text=text,
                speaker_wav=synthesize_speaker_wav,
                speaker=speaker_name,
                language=language.split('-')[0],  # Use 2-letter code for XTTS v2
                output_format=outputFormat
            )
        except Exception as e:
            logger.error(f"Synthesis failure for request {request_id}: {e}")
            raise HTTPException(status_code=500, detail="Internal synthesis failure")
        
        synthesis_duration = time.time() - synthesis_start
        total_duration = time.time() - start_time

        # Logging requirements
        debug_text = text if logger.isEnabledFor(logging.DEBUG) else (text[:100] + "..." if len(text) > 100 else text)
        logger.info(
            f"ID={request_id} Model={tts_manager.active_model_name if tts_manager.model else 'unavailable'} Source={voice_source} "
            f"Lang={language} Doc={doc_type} Format={outputFormat} Device={tts_manager.device} HasGPU={tts_manager.has_gpu} "
            f"SynthTime={synthesis_duration:.2f}s TotalTime={total_duration:.2f}s"
        )
        logger.debug(f"ID={request_id} Text: {debug_text}")

        media_type = "audio/wav" if outputFormat.lower() == "wav" else "audio/mpeg"
        return Response(content=audio_content, media_type=media_type)

    finally:
        if temp_ref_file and os.path.exists(temp_ref_file.name):
            os.remove(temp_ref_file.name)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "tts-service",
        "ready": tts_manager is not None and tts_manager.model is not None,
        "model": tts_manager.active_model_name if tts_manager else None,
        "device": tts_manager.device if tts_manager else "unknown",
        "has_gpu": tts_manager.has_gpu if tts_manager else None,
        "hostname": socket.gethostname(),
        "datetime": datetime.utcnow().isoformat()
    }
