"""Is this Krish? Used on the wake word, on every command and on barge-in
(where it also filters out SEVRIN's own voice).

A sample is scored against the centroid of the enrolment embeddings. I tried
matching against the best individual clips instead and it separated speakers
worse: averaging cancels per-clip recording noise.

With an impostor cohort (cohort.py) the decision uses an AS-norm z-score,
which holds up when the mic or room changes. Without one it falls back to
the raw cosine against the threshold enroll.py calibrated.
"""
import json
import os

import numpy as np

import config
from voice.voice_id import cohort as cohort_mod
from voice.voice_id import encoder

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")
PIPELINE_VERSION = 5    # must match enroll.py

# AS-norm compares against the closest 20% of the cohort (at least 5 voices).
# Using half of it, as I first did, was too broad and let friends score close.
COHORT_TOP_FRACTION = 0.20
COHORT_TOP_N_MIN = 5

# Shorter clips give noisier embeddings. Below MIN we don't score at all;
# below SHORT the raw-cosine threshold gets a small bump.
MIN_CLIP_SECONDS = 0.6
SHORT_CLIP_SECONDS = 1.2
SHORT_CLIP_PENALTY = 0.02

_centroid = None
_calibration = None
_cohort = None
_cohort_top_n = None
_cohort_enroll_stats = None     # (mean, std) of the centroid's top-N cohort scores


def voiceprint_available() -> bool:
    return os.path.exists(PROFILE_PATH)


def _load():
    global _centroid, _calibration, _cohort, _cohort_top_n, _cohort_enroll_stats
    if _centroid is None:
        if not os.path.exists(PROFILE_PATH):
            raise FileNotFoundError(
                "No voiceprint found. Run voice/voice_id/record.py then voice/voice_id/enroll.py.")
        _centroid = np.load(PROFILE_PATH)

    if _cohort is None:
        _cohort = cohort_mod.load()
        if _cohort is not None:
            _cohort_top_n = max(COHORT_TOP_N_MIN, int(len(_cohort) * COHORT_TOP_FRACTION))
            sims = np.sort(_cohort @ _centroid)[-_cohort_top_n:]
            _cohort_enroll_stats = (float(sims.mean()), float(sims.std()) or 1e-6)

    if _calibration is None:
        if not os.path.exists(CALIBRATION_PATH):
            _calibration = {"threshold": config.VOICE_MATCH_THRESHOLD}
            return
        with open(CALIBRATION_PATH) as f:
            _calibration = json.load(f)
        # scores from a different pipeline or model aren't comparable at all
        if _calibration.get("pipeline_version", 1) < PIPELINE_VERSION:
            raise RuntimeError("Voiceprint was built with an older embedding pipeline and its "
                               "scores are not comparable. Re-run: python3 voice/voice_id/enroll.py")
        saved, current = _calibration.get("backend"), encoder.backend_name()
        if saved and saved != current:
            raise RuntimeError(f"Voiceprint was enrolled with '{saved}' but the running "
                               f"encoder is '{current}'. Re-run: python3 voice/voice_id/enroll.py")


def threshold_for(duration_seconds: float) -> float:
    """Raw-cosine threshold for live audio (only used without a cohort).

    Enrolment scores come from leave-one-out on clean recordings; live clips
    are shorter and noisier and score lower against the same centroid.
    """
    _load()
    base = max(0.35, float(_calibration.get("threshold", config.VOICE_MATCH_THRESHOLD))
               - config.VOICE_LIVE_ALLOWANCE)   # 0.35 floor suits ECAPA's scale
    return base + SHORT_CLIP_PENALTY if duration_seconds < SHORT_CLIP_SECONDS else base


def _centre_window(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """A contiguous VOICE_ID_CLIP_SECONDS slice from the middle, where the speech is.

    Embeddings depend on clip length: a 1 s command scored 0.13 against a
    voiceprint built from 2.5 s windows. Shorter audio is used as-is.
    """
    win = int(sample_rate * config.VOICE_ID_CLIP_SECONDS)
    if audio.size <= win:
        return audio
    start = (audio.size - win) // 2
    return audio[start:start + win]


def voice_similarity(pcm_bytes: bytes, sample_rate: int = 16000):
    """{"raw", "z", "normalized"} for a PCM16 sample, or None if it can't be
    scored reliably (too short or silent)."""
    _load()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0 or np.abs(audio).max() < 1e-4:
        return None
    if audio.size < sample_rate * MIN_CLIP_SECONDS:
        return None

    emb = encoder.embed(_centre_window(audio, sample_rate), sample_rate)
    if emb is None:
        return None
    raw = float(np.dot(_centroid, emb))

    if _cohort is None:
        return {"raw": raw, "z": None, "normalized": False}

    # AS-norm: how far above the closest impostors this score sits, measured
    # from both the test sample's side and the voiceprint's side
    test_sims = np.sort(_cohort @ emb)[-_cohort_top_n:]
    t_mean, t_std = float(test_sims.mean()), float(test_sims.std()) or 1e-6
    e_mean, e_std = _cohort_enroll_stats
    z = 0.5 * (((raw - t_mean) / t_std) + ((raw - e_mean) / e_std))
    return {"raw": raw, "z": z, "normalized": True}


def verify(pcm_bytes: bytes, sample_rate: int = 16000):
    """Returns (is_krish, scores, threshold). scores is None if unscoreable."""
    result = voice_similarity(pcm_bytes, sample_rate)
    if result is None:
        return False, None, None
    if result["normalized"]:
        thr = config.VOICE_ZNORM_THRESHOLD
        return result["z"] >= thr, result, thr
    thr = threshold_for(len(pcm_bytes) / 2 / sample_rate)
    return result["raw"] >= thr, result, thr


def score_value(result):
    """The number the decision is made on: z if normalised, else raw cosine."""
    if result is None:
        return None
    return result["z"] if result["normalized"] else result["raw"]


def score_text(result) -> str:
    if result is None:
        return "unscoreable"
    if result["normalized"]:
        return f"z={result['z']:.2f} (raw {result['raw']:.2f})"
    return f"{result['raw']:.2f}"


def calibration_summary():
    """The calibration dict, for startup logging. None if there's no voiceprint."""
    try:
        _load()
    except Exception:
        return None
    return dict(_calibration)
