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
from voice.voice_id import cohort as cohort_mod

PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")
EMBEDDINGS_PATH = os.path.join(PROFILE_DIR, "embeddings.npy")
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")

_centroid = None
_embeddings = None
_calibration = None
_cohort = None
_cohort_enroll_stats = None   # (mean, std) of enrollment-vs-cohort scores

# AS-norm compares against the top-N most similar impostors rather than all
# of them; using the closest ones makes the normalisation sharper.
# Fraction of the cohort to compare against, not a fixed count. With the
# earlier fixed value of 12 against a 24-voice cohort, ANY test sample was
# being compared against fully half the cohort — too broad a slice for a
# sharp reference, and part of why real friends were scoring close enough
# to pass. A smaller proportional slice compares against genuinely the
# closest-sounding impostors, which is what AS-norm is supposed to do.
COHORT_TOP_FRACTION = 0.20
COHORT_TOP_N_MIN = 5

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
    global _cohort, _cohort_enroll_stats
    if _cohort is None:
        _cohort = cohort_mod.load()
        if _cohort is not None and _centroid is not None:
            n = max(COHORT_TOP_N_MIN, int(len(_cohort) * COHORT_TOP_FRACTION))
            sims = np.sort(_cohort @ _centroid)[-n:]
            _cohort_enroll_stats = (float(sims.mean()), float(sims.std()) or 1e-6)

    if _calibration is None:
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH) as f:
                _calibration = json.load(f)
            if _calibration.get("pipeline_version", 1) < 5:
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

    raw = float(np.dot(_centroid, emb))

    # --- AS-norm ---------------------------------------------------------
    # Without normalisation the raw cosine shifts whenever the microphone or
    # room changes, so a fixed threshold rejects the right person on a
    # different device. Expressing the score relative to how this same audio
    # scores against other speakers cancels most of that channel effect:
    # when conditions shift, the impostor scores shift too.
    if _cohort is not None and _cohort_enroll_stats is not None:
        n = max(COHORT_TOP_N_MIN, int(len(_cohort) * COHORT_TOP_FRACTION))
        test_sims = np.sort(_cohort @ emb)[-n:]
        t_mean, t_std = float(test_sims.mean()), float(test_sims.std()) or 1e-6
        e_mean, e_std = _cohort_enroll_stats
        z = 0.5 * (((raw - t_mean) / t_std) + ((raw - e_mean) / e_std))
        return {"raw": raw, "z": z, "normalized": True}

    return {"raw": raw, "z": None, "normalized": False}


def verify(pcm_bytes: bytes, sample_rate: int = 16000):
    """Returns (is_krish, score, threshold_used).

    When AS-norm is available the decision is made on the NORMALISED score,
    which is stable across microphones and rooms. The raw cosine is still
    reported for logging, since it's the number that's comparable with the
    enrollment statistics."""
    result = voice_similarity(pcm_bytes, sample_rate)
    if result is None:
        return False, None, None

    if isinstance(result, dict) and result.get("normalized"):
        z = result["z"]
        thr = float(getattr(config, "VOICE_ZNORM_THRESHOLD", 3.0))
        return (z >= thr), result, thr

    raw = result["raw"] if isinstance(result, dict) else result
    duration = len(pcm_bytes) / 2 / sample_rate
    thr = threshold_for(duration)
    return (raw >= thr), result, thr


def score_value(result):
    """The single number to threshold on / average, whichever scoring path
    produced it: the normalised z when available, else the raw cosine."""
    if result is None:
        return None
    if isinstance(result, dict):
        return result["z"] if result.get("normalized") else result["raw"]
    return result


def score_text(result) -> str:
    """Human-readable score for logs, whichever scoring path was used."""
    if result is None:
        return "unscoreable"
    if isinstance(result, dict):
        if result.get("normalized"):
            return f"z={result['z']:.2f} (raw {result['raw']:.2f})"
        return f"{result['raw']:.2f}"
    return f"{result:.2f}"


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
