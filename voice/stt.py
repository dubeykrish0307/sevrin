"""
Transcription. Two backends:

  whisper.cpp (voice/stt_whispercpp.py) — Metal/Core ML accelerated on
  Apple Silicon, makes a genuinely heavy model (large-v3-turbo by default)
  practical at real-time speed. Used automatically when WHISPER_CPP_DIR is
  set in .env and the server starts successfully.

  faster-whisper (below) — CPU-only on macOS (no Metal backend there), so
  kept to smaller models. This is the automatic fallback if whisper.cpp
  isn't configured, fails to start, or a request to it fails mid-session —
  a broken or missing whisper.cpp setup degrades the assistant, it doesn't
  take it down.

get_model()/transcribe() keep the same signatures either way, so nothing
else in the codebase needs to know or care which backend is active.
"""
import numpy as np
from faster_whisper import WhisperModel

import config

_model = None
_backend = None   # 'whispercpp' | 'fasterwhisper', set on first get_model() call


def get_model():
    """Eager-loads whichever backend is configured. Call once at startup —
    for whisper.cpp this pays the model-load cost up front, same reasoning
    as why faster-whisper was already eager-loaded rather than lazy."""
    global _model, _backend
    if _backend is not None:
        return _model

    if getattr(config, "WHISPER_CPP_DIR", None):
        try:
            from voice import stt_whispercpp
            stt_whispercpp.start_server()
            _backend = "whispercpp"
            print(f"[stt] backend: whisper.cpp "
                  f"({getattr(config, 'WHISPER_CPP_MODEL', 'large-v3-turbo')})")
            return None
        except Exception as e:
            print(f"[stt] whisper.cpp unavailable ({e})")
            print("[stt] falling back to faster-whisper")

    _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    _backend = "fasterwhisper"
    print(f"[stt] backend: faster-whisper ({config.WHISPER_MODEL}, CPU)")
    return _model


def transcribe(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        return ""
    get_model()   # ensures _backend is set even if the caller skipped warm-up

    if _backend == "whispercpp":
        from voice import stt_whispercpp
        try:
            return stt_whispercpp.transcribe(pcm_bytes)
        except stt_whispercpp.WhisperCppUnavailable as e:
            print(f"[stt] whisper.cpp request failed ({e}) — this utterance falls back "
                  f"to faster-whisper. If this keeps happening, the server itself needs "
                  f"attention, not just this one call.")
            return _fasterwhisper_transcribe(pcm_bytes)

    return _fasterwhisper_transcribe(pcm_bytes)


def _fasterwhisper_transcribe(pcm_bytes: bytes) -> str:
    global _model
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # No vad_filter here — listener.py already trims to just the speech
    # before this ever runs. beam_size=1 (greedy) trades a small amount of
    # accuracy for speed.
    segments, _ = _model.transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
