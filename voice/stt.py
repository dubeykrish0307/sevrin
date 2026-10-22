"""Speech to text.

Uses the whisper.cpp server (voice/stt_whispercpp.py) when WHISPER_CPP_DIR is
set and it starts. Otherwise, or whenever a request to it fails,
faster-whisper on the CPU. Callers don't need to know which one ran.
"""
import numpy as np
from faster_whisper import WhisperModel

import config

_model = None
_backend = None     # "whispercpp" or "fasterwhisper", decided on the first get_model()


def _faster_whisper():
    global _model
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def get_model():
    """Load the STT backend. Call once at startup so the first command doesn't wait for it."""
    global _backend
    if _backend is not None:
        return _model

    if config.WHISPER_CPP_DIR:
        try:
            from voice import stt_whispercpp
            stt_whispercpp.start_server()
            _backend = "whispercpp"
            print(f"[stt] backend: whisper.cpp ({config.WHISPER_CPP_MODEL})")
            return None
        except Exception as e:
            print(f"[stt] whisper.cpp unavailable ({e})")
            print("[stt] falling back to faster-whisper")

    _faster_whisper()
    _backend = "fasterwhisper"
    print(f"[stt] backend: faster-whisper ({config.WHISPER_MODEL}, CPU)")
    return _model


def transcribe(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        return ""
    get_model()

    if _backend == "whispercpp":
        from voice import stt_whispercpp
        try:
            return stt_whispercpp.transcribe(pcm_bytes)
        except stt_whispercpp.WhisperCppUnavailable as e:
            print(f"[stt] whisper.cpp request failed ({e}), using faster-whisper for this one")

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # the listener already trimmed to speech, so no vad_filter; greedy decoding for speed
    segments, _ = _faster_whisper().transcribe(audio, language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
