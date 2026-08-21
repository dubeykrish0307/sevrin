"""
Transcription runs fully local via faster-whisper.
"""
import numpy as np
from faster_whisper import WhisperModel

import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def transcribe(pcm_bytes: bytes) -> str:
    if not pcm_bytes:
        return ""
    model = get_model()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # No vad_filter here — listener.py already trims to just the speech
    # (with a small pre-roll) using webrtcvad before this ever runs.
    # Running faster-whisper's own internal VAD (a heavier Silero net) on
    # top of that was pure redundant compute on every single call, and is
    # the most likely reason transcription time stayed ~flat regardless of
    # how much you actually said. beam_size=1 (greedy) trades a small
    # amount of accuracy for speed — bump back to 2+ if you notice more
    # misheard words.
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
