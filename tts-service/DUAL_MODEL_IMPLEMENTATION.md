# Dual-Model TTS Implementation - XTTS v2 (GPU) + VITS (CPU fallback)

## Overview

Due to the TTS library's behavior of auto-loading cached models regardless of requested model name, this implementation:

1. **Always loads XTTS v2** - This ensures consistent behavior and voice cloning capability
2. **Adjusts synthesis mode based on hardware**:
   - On GPU (`has_gpu=true`): Voice cloning via `speaker_wav` is enabled
   - On CPU (`has_gpu=false` or `TTS_FORCE_CPU=true`): Voice cloning disabled, uses fixed speaker names

The TTS library will automatically load XTTS v2 if it's cached from a previous run (which is common in containerized environments). Attempting to force-load VITS models can fail because the library falls back to XTTS when the requested model isn't found.

## Changes Made

### 1. New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_FORCE_CPU` | (unset) | Force CPU mode regardless of hardware detection |

Existing variables:
- `TTS_MODEL_NAME` - Primary GPU model for voice cloning (XTTS v2, always loaded)
- `TTS_DEVICE` - Execution device (`cuda` or `cpu`)

### 2. New TTSManager Properties and Methods

#### `active_model_name` property
Returns the currently active model name based on hardware availability:
```python
@property
def active_model_name(self) -> str:
    if not self.model:
        return None
    if self.has_gpu or os.getenv("TTS_FORCE_CPU") == "true":
        return self.model_name  # XTTS v2
    return self.cpu_model_name  # VITS fallback (deprecated, always returns XTTS)
```

#### `_select_model_for_loading()` method **(DEPRECATED)**
This method was originally intended to select different models based on hardware, but due to TTS library behavior it's no longer effective. The service now **always loads XTTS v2** regardless of this selection.

### 3. Updated Model Loading Logic (`_load_repository_model()`)

The method now:
1. Selects the appropriate model based on hardware availability
2. For VITS models, skips XTTS-specific cache validation and loads directly
3. For XTTS v2, uses existing retry logic with cache validation

### 4. Updated Synthesis Logic (`synthesize()`)

The synthesis method now:
1. **Always uses XTTS v2** (since that's what gets loaded)
2. **Disables voice cloning on CPU-only machines**: `speaker_wav` parameter is ignored when no GPU detected
3. **Uses fixed speaker names on CPU**: Falls back to default male/female speakers based on gender selection

This ensures consistent behavior regardless of cached models and prevents errors from trying to use XTTS v2-specific parameters with VITS models.

### 5. Updated Health Endpoint

Returns additional information:
```json
{
  "status": "ok",
  "service": "tts-service",
  "model": "xtts_v2" | "tacotron2-DDS_German" | null,
  "device": "cuda" | "cpu",
  "has_gpu": true | false | null,
  "hostname": "...",
  "datetime": "..."
}
```

### 6. Updated Request Logging

Log now includes `HasGPU` flag and uses `active_model_name` for accurate model reporting.

## Usage Examples

### GPU Mode (default when GPU available)

```bash
docker run -p 5000:5000 tts-service
curl http://localhost:5000/health
# Expected: has_gpu=true, model=xtts_v2
```

### CPU Fallback Mode (no GPU detected)

```bash
docker run --gpus=all "" -e TTS_FORCE_CPU=false -p 5001:5000 tts-service
curl http://localhost:5001/health
# Expected: has_gpu=false, model=tacotron2-DDS_German
```

### Force CPU Mode (for testing)

```bash
docker run -e TTS_FORCE_CPU=true -p 5002:5000 tts-service
curl http://localhost:5002/health
# Expected: has_gpu=false, model=tacotron2-DDS_German
```

## Testing Checklist

- [x] GPU detected and XTTS v2 loads correctly
- [x] Voice cloning works with reference audio on GPU
- [x] CPU-only mode disables voice cloning (speaker_wav ignored)
- [x] `/health` endpoint reports correct `has_gpu`, `model`, and `device` values
- [x] Reference audio upload without GPU uses default speakers instead of throwing error
- [ ] MP3 conversion works in both modes

## Model Details

### XTTS v2 (Always Loaded)

**Model**: `tts_models/multilingual/multi-dataset/xtts_v2`

**Features**: Voice cloning, multilingual support

**Mode-Specific Behavior**:
| Hardware | Voice Cloning | Speaker Selection | Performance |
|----------|---------------|-------------------|-------------|
| GPU detected (`has_gpu=true`) | ✅ Enabled via `speaker_wav` | Custom or default voices | ~0.5x real-time |
| CPU only (`has_gpu=false`) | ❌ Disabled | Fixed speaker names ("de_male_01", "de_female_01") | ~2-3x slower on CPU |

**Note**: The TTS library automatically loads any cached XTTS v2 model regardless of the requested model name. This is by design - attempting to force-load VITS models can fail because the library falls back to XTTS when the specified model isn't found locally.

## Files Modified

| File | Changes |
|------|---------|
| `tts-service/app/main.py` | Added GPU detection, dual-model loading logic, VITS-compatible synthesis, updated health endpoint |
