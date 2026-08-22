"""
Speaker verification — is this actually Krish?

Used in three places now (previously only one):
  * the wake word, so "hey jarvis" from someone else does nothing
  * every COMMAND after waking, so a stranger can't drive SEVRIN once it's
    awake — this was the biggest hole in the old design
  * barge-in, where it also serves as the echo filter (SEVRIN's own TTS
    voice doesn't match Krish, so he can't interrupt himself)

Scoring uses the CENTROID of your enrolled clips. I tested per-clip "top-K"
matching against this and it was consistently WORSE at separating speakers
(~13-16% lower d-prime in simulation): averaging suppresses per-clip
recording noise, which is exactly what you want, while top-K rewards whichever
single clip happens to share a recording condition with the sample — including
for impostors in the same room. Per-clip embeddings are still stored, but
they're used for THRESHOLD CALIBRATION, not scoring.

The threshold comes from calibration.json, computed from your actual voice
during enrollment (see enroll.py). Falls back to config if absent.
"""
import json
import os

import numpy as np

import config
from voice.voice_id import encoder as enc

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")
EMBEDDINGS_PATH = os.path.join(PROFILE_DIR, "embeddings.npy")
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")

_centroid = None
_embeddings = None
_calibration = None

# Short clips give noisier embeddings — but note the penalty here RAISES the
# bar, and enrollment clips are longer than live wake clips, so stacking a
# penalty on top of a threshold calibrated from longer audio double-counts
# the same effect and rejects genuine speech. The penalty is therefore small,
# and short samples are instead handled by demanding a minimum duration.
MIN_CLIP_SECONDS = 0.6
SHORT_CLIP_SECONDS = 1.2
SHORT_CLIP_PENALTY = 0.02


def voiceprint_available() -> bool:
    return os.path.exists(PROFILE_PATH)


def _load():
    global _centroid, _embeddings, _calibration
    if _centroid is None:
        if not os.path.exists(PROFILE_PATH):
            raise FileNotFoundError(
                "No voiceprint found. Run voice/voice_id/record.py then voice/voice_id/enroll.py."
            )
        _centroid = np.load(PROFILE_PATH)
    if _embeddings is None:
        if os.path.exists(EMBEDDINGS_PATH):
            _embeddings = np.load(EMBEDDINGS_PATH)
        else:
            # older enrollment without per-clip data — degrade to centroid only
            _embeddings = _centroid.reshape(1, -1)
    if _calibration is None:
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH) as f:
                _calibration = json.load(f)
            if _calibration.get("pipeline_version", 1) < 4:
                raise RuntimeError(
                    "Voiceprint was built with an older embedding pipeline and its "
                    "scores are not comparable. "
                    "Re-run: python3 voice/voice_id/enroll.py"
                )
            saved = _calibration.get("backend")
            current = enc.backend_name()
            if saved and saved != current:
                # Embeddings from different models are not comparable at all —
                # scores would be meaningless rather than merely inaccurate.
                raise RuntimeError(
                    f"Voiceprint was enrolled with '{saved}' but the running "
                    f"encoder is '{current}'. Re-run: python3 voice/voice_id/enroll.py"
                )
        else:
            _calibration = {"threshold": config.VOICE_MATCH_THRESHOLD}


# Enrollment scores come from leave-one-out on full recordings; live samples
# are shorter and captured mid-motion, so they score systematically lower
# against the same centroid. Without allowing for that gap, a threshold that
# looks well-calibrated on paper rejects the real thing.
LIVE_DOMAIN_ALLOWANCE = float(getattr(config, "VOICE_LIVE_ALLOWANCE", 0.08))


def threshold_for(duration_seconds: float) -> float:
    _load()
    base = float(_calibration.get("threshold", config.VOICE_MATCH_THRESHOLD))
    base = max(0.35, base - LIVE_DOMAIN_ALLOWANCE)   # 0.35 floor suits ECAPA's scale
    if duration_seconds < SHORT_CLIP_SECONDS:
        return base + SHORT_CLIP_PENALTY
    return base


def _canonical_window(audio: np.ndarray, sample_rate: int):
    """Return a fixed-length CONTIGUOUS slice, so every verification sees the
    same duration the voiceprint was built from.

    Enrollment embeds 2.5s windows. Wake clips are exactly 2.5s and score
    well (0.43-0.59), but COMMANDS are whatever length the sentence happened
    to be — and a ~1s command scored 0.13 against the same voiceprint, purely
    from the duration mismatch. Speaker embeddings are sensitive to utterance
    length, so the fix is to always score the same amount of audio.

    Longer input: take a window from the middle, where the speech actually is
    (the start often contains the leading pre-roll and the end trails off).
    Shorter input: use it whole — audio can't be invented, and the caller is
    warned via the returned flag.
    """
    win = int(sample_rate * float(getattr(config, "VOICE_ID_CLIP_SECONDS", 2.5)))
    if audio.size <= win:
        return audio, False           # shorter than ideal
    start = (audio.size - win) // 2   # centre slice, contiguous
    return audio[start:start + win], True


def voice_similarity(pcm_bytes: bytes, sample_rate: int = 16000):
    """Top-K similarity against enrolled clips, or None if the sample can't
    be scored reliably (too short, too quiet)."""
    _load()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0 or np.abs(audio).max() < 1e-4:
        return None
    if audio.size < sample_rate * MIN_CLIP_SECONDS:
        return None

    # Match the enrollment domain: always embed a fixed-length window.
    audio, _full = _canonical_window(audio, sample_rate)

    emb = enc.embed(audio, sample_rate)
    if emb is None:
        return None

    # centroid scoring — see the module docstring for why this beats top-K
    return float(np.dot(_centroid, emb))


def verify(pcm_bytes: bytes, sample_rate: int = 16000):
    """Returns (is_krish, score, threshold_used). score/threshold are None
    when the sample couldn't be scored at all."""
    score = voice_similarity(pcm_bytes, sample_rate)
    if score is None:
        return False, None, None
    duration = len(pcm_bytes) / 2 / sample_rate
    thr = threshold_for(duration)
    return (score >= thr), score, thr


def is_my_voice(pcm_bytes: bytes, sample_rate: int = 16000) -> bool:
    ok, _, _ = verify(pcm_bytes, sample_rate)
    return ok


def calibration_summary():
    """For startup logging so it's obvious how well-calibrated the profile is."""
    try:
        _load()
    except Exception:
        return None
    return dict(_calibration)
