"""Step 2 of enrolment: python3 voice/voice_id/enroll.py

Builds the voiceprint from voice/voice_id/recordings/ and calibrates the
accept threshold from my own voice instead of a guessed number:

  - every recording is cut into windows the same length as a live clip
  - leave-one-out: each window is scored against the centroid of the others
  - windows that clearly don't match the rest (bad recordings) are dropped
  - threshold = mean - SAFETY_SIGMAS * std of what's left

Then it builds the impostor cohort used for score normalisation.
"""
import json
import os
import sys
import traceback
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from voice.voice_id.verify import PIPELINE_VERSION

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")       # centroid
EMBEDDINGS_PATH = os.path.join(PROFILE_DIR, "embeddings.npy")    # per window, for re-calibrating
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")

SAFETY_SIGMAS = 2.0             # ~97.7% of genuine attempts accepted if scores are roughly normal
OUTLIER_ABSOLUTE_FLOOR = 0.35   # a "me" window below this is a broken recording; other people sit at 0.0-0.3
MIN_THRESHOLD = 0.40            # never go lower, whatever calibration says
TYPICAL_IMPOSTOR = 0.30         # where different speakers usually land with ECAPA


def score_against(embedding, embeddings):
    """Score against the centroid of `embeddings`, exactly how verify.py scores
    at runtime. Calibration is meaningless if it scores differently."""
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return float(np.dot(centroid, embedding))


def loo_scores(embs):
    return np.array([score_against(embs[i], np.delete(embs, i, axis=0)) for i in range(len(embs))])


def main():
    if not os.path.isdir(RECORDINGS_DIR) or not os.listdir(RECORDINGS_DIR):
        print("No recordings found. Run voice/voice_id/record.py first.")
        return

    from voice.voice_id import encoder

    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav"))
    print(f"Found {len(files)} recordings.\n")

    window = config.VOICE_ID_CLIP_SECONDS
    print(f"  (embedding in {window}s windows to match live clips)\n")

    embeddings, sources = [], []
    for fname in files:
        try:
            embs = encoder.embed_file_windows(os.path.join(RECORDINGS_DIR, fname), window_seconds=window)
        except Exception as e:
            print(f"  skipped {fname} ({e})")
            continue
        if not embs:
            print(f"  skipped {fname} (too short / too quiet)")
            continue
        embeddings.extend(embs)
        sources.extend([fname] * len(embs))
        print(f"  processed {fname}  ({len(embs)} window(s))")

    if len(embeddings) < 3:
        print("\nNeed at least 3 usable recordings to calibrate. Record more.")
        return
    embeddings = np.stack(embeddings)

    print("\nCalibrating threshold (leave-one-out)...")
    scores = loo_scores(embeddings)

    # A bad recording drags the centroid away from my real voice and widens the
    # spread, which forces a uselessly low threshold. Find outliers with
    # median/MAD, since mean/std are skewed by the very outliers we're after.
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) or 1e-6
    robust_sigma = 1.4826 * mad     # MAD scaled to match a std for normal data
    cutoff = max(OUTLIER_ABSOLUTE_FLOOR, median - 2.5 * robust_sigma)

    keep = scores >= cutoff
    dropped = [sources[i] for i in range(len(scores)) if not keep[i]]
    if dropped:
        print(f"\n  Dropped {len(dropped)} window(s) that don't match the rest (cutoff {cutoff:.3f}):")
        for name, n in Counter(dropped).most_common():
            print(f"    {name}: {n} window(s) dropped")
        if keep.sum() < 3:
            print("\n  Too few clips left after dropping outliers. Re-record enrollment in a quiet room.")
            return
        embeddings = embeddings[keep]
        scores = loo_scores(embeddings)
        print(f"  Recalibrated on the {len(embeddings)} good clips.")

    mean, std = float(scores.mean()), float(scores.std())
    threshold = max(MIN_THRESHOLD, mean - SAFETY_SIGMAS * std)

    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    np.save(PROFILE_PATH, centroid)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({
            "backend": encoder.backend_name(),
            # bump PIPELINE_VERSION in verify.py whenever the way audio becomes an
            # embedding changes, so old voiceprints get rejected instead of misused
            "pipeline_version": PIPELINE_VERSION,
            "n_clips": int(len(embeddings)),
            "n_dropped": len(dropped),
            "genuine_mean": mean,
            "genuine_std": std,
            "genuine_min": float(scores.min()),
            "genuine_max": float(scores.max()),
            "threshold": threshold,
            "safety_sigmas": SAFETY_SIGMAS,
        }, f, indent=2)

    print(f"\n  clips used:        {len(embeddings)}")
    print(f"  genuine scores:    mean {mean:.3f}, std {std:.3f}, "
          f"range {scores.min():.3f}-{scores.max():.3f}")
    print(f"  chosen threshold:  {threshold:.3f}")
    print(f"\nSaved to {PROFILE_DIR}/")

    try:
        from voice.voice_id import cohort
        cohort.build(force=True)
    except Exception as e:
        print(f"\n  (cohort build skipped: {e})")
        print("  Recognition will still work, but will be more sensitive to")
        print("  changes of microphone or environment.")

    if std > 0.10:
        print("\n  NOTE: your remaining clips still vary a lot. Inconsistent mic")
        print("  distance or background noise during recording. Re-recording more")
        print("  consistently would tighten the threshold.")
    # ECAPA spreads scores wide (same speaker ~0.5-0.8). That looks lower than
    # Resemblyzer's 0.7-0.95 but the spread is what lets it separate people.
    if mean < 0.45:
        print("\n  NOTE: genuine similarity is low even for ECAPA. Longer clips")
        print("  recorded in a quiet room would improve separation.")
    else:
        margin = mean - TYPICAL_IMPOSTOR
        print(f"\n  Separation margin over a typical impostor: {margin:.2f}")
        if margin > 0.25:
            print("  That's healthy, impostors should be rejected reliably.")

    print("\nStart the backend: the wake word and every command will now be checked")
    print("against this voiceprint.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== ENROLLMENT FAILED ===")
        traceback.print_exc()
        sys.exit(1)
