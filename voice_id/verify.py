"""
Compares a live audio clip against your enrolled voiceprint. wake_word.py
calls voice_similarity() after a phrase match to get the actual match
score — both the words AND the voice have to match for the wake to fire.
"""
import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

import config

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "profile", "voiceprint.npy")

_encoder = None
_voiceprint = None


def voiceprint_available() -> bool:
    """Cheap check callers can use before calling is_my_voice(), so a
    missing enrollment degrades to 'skip the check' instead of crashing."""
    return os.path.exists(PROFILE_PATH)


def _load():
    global _encoder, _voiceprint
    if _encoder is None:
        _encoder = VoiceEncoder()
    if _voiceprint is None:
        if not os.path.exists(PROFILE_PATH):
            raise FileNotFoundError(
                "No voiceprint found. Run voice_id/record.py then voice_id/enroll.py first."
            )
        _voiceprint = np.load(PROFILE_PATH)


def voice_similarity(pcm_bytes: bytes, sample_rate: int = 16000):
    """Returns the raw cosine similarity (0-1) against your enrolled
    voiceprint, or None if the clip was too short to embed reliably.
    Exposed separately from is_my_voice() so callers can log the actual
    number — useful for tuning VOICE_MATCH_THRESHOLD against real data
    instead of guessing."""
    _load()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav = preprocess_wav(audio, source_sr=sample_rate)
    if len(wav) < sample_rate * 0.5:
        return None  # too short for a reliable embedding
    embed = _encoder.embed_utterance(wav)
    embed = embed / np.linalg.norm(embed)
    return float(np.dot(embed, _voiceprint))


def is_my_voice(pcm_bytes: bytes, sample_rate: int = 16000) -> bool:
    similarity = voice_similarity(pcm_bytes, sample_rate)
    return similarity is not None and similarity >= config.VOICE_MATCH_THRESHOLD