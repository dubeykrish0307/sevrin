"""
Run second: `python3 voice/voice_id/enroll.py`

Builds Krish's voiceprint AND calibrates the accept/reject threshold from
real data instead of guessing.

WHY THIS CHANGED
The first version averaged every clip into one centroid embedding and used
a hand-picked threshold. Averaging throws away detail — a single centroid
sits "between" all your recordings and matches none of them strongly, which
is why genuine attempts only scored 0.57-0.65 and left almost no margin
against an impostor.

Now:
  * the threshold is CALIBRATED by leave-one-out: each clip is scored
    against the centroid of the OTHERS, as if it were an unknown speaker.
    That yields a real distribution of genuine scores, and the threshold is
    set below it by a safety margin — grounded in your actual voice instead
    of a number I invented.
  * per-clip embeddings are saved too, but only so calibration can be
    re-run; scoring itself uses the centroid (tested as more discriminative
    than per-clip matching).
"""
import json
import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")       # centroid (compat)
EMBEDDINGS_PATH = os.path.join(PROFILE_DIR, "embeddings.npy")    # per-clip
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")

# How many standard deviations below the genuine mean to place the threshold.
# Larger = more forgiving to you, easier for an impostor. 2.0 accepts ~97.7%
# of genuine attempts if scores are roughly normal.
SAFETY_SIGMAS = 2.0
# Any clip scoring below this against the others is almost certainly a bad
# recording rather than a quirk of your voice — ECAPA puts genuinely
# different speakers around 0.0-0.3, so a "self" clip down there is broken.
OUTLIER_ABSOLUTE_FLOOR = 0.35
# Never go below this no matter what calibration says; ECAPA impostor scores
# cluster under ~0.3, so this keeps real separation.
MIN_THRESHOLD = 0.40


def score_against(embedding, embeddings):
    """Score a sample the same way verify.py does at runtime: against the
    centroid of the reference set. Calibration MUST use the same scoring
    function as production, or the threshold it produces is meaningless."""
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return float(np.dot(centroid, embedding))


def main():
    if not os.path.isdir(RECORDINGS_DIR) or not os.listdir(RECORDINGS_DIR):
        print("No recordings found. Run voice/voice_id/record.py first.")
        return

    from voice.voice_id import encoder as enc

    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav"))
    print(f"Found {len(files)} recordings.\n")

    # Embed every recording as windows the SAME LENGTH as a live wake clip,
    # so the voiceprint represents the audio we actually verify against.
    # See encoder.embed_file_windows for why this matters.
    window = float(getattr(config, "VOICE_ID_CLIP_SECONDS", 2.5))
    print(f"  (embedding in {window}s windows to match live clips)\n")

    embeddings = []
    files_used = []
    for fname in files:
        path = os.path.join(RECORDINGS_DIR, fname)
        try:
            embs = enc.embed_file_windows(path, window_seconds=window)
            if not embs:
                print(f"  skipped {fname} (too short / too quiet)")
                continue
            for e in embs:
                embeddings.append(e)
                files_used.append(fname)
            print(f"  processed {fname}  ({len(embs)} window(s))")
        except Exception as e:
            print(f"  skipped {fname} ({e})")

    if len(embeddings) < 3:
        print("\nNeed at least 3 usable recordings to calibrate. Record more.")
        return

    embeddings = np.stack(embeddings)

    # --- leave-one-out calibration with outlier rejection ----------------
    # A bad recording (mostly silence, clipped, someone else talking over it)
    # poisons BOTH the centroid and the calibration: it drags the average
    # voiceprint away from your real voice and inflates the spread, which
    # forces a uselessly low threshold. So we score every clip against the
    # others, drop the ones that don't look like the same speaker, and then
    # rebuild from what's left.
    def loo_scores(embs):
        out = []
        for i in range(len(embs)):
            others = np.delete(embs, i, axis=0)
            out.append(score_against(embs[i], others))
        return np.array(out)

    print("\nCalibrating threshold (leave-one-out)...")
    scores = loo_scores(embeddings)

    # Robust outlier detection: median and MAD rather than mean/std, because
    # mean and std are themselves distorted by the outliers we're hunting.
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median))) or 1e-6
    # 1.4826 makes MAD comparable to a standard deviation for normal data
    robust_sigma = 1.4826 * mad
    cutoff = max(OUTLIER_ABSOLUTE_FLOOR, median - 2.5 * robust_sigma)

    keep = scores >= cutoff
    dropped = [(files_used[i], float(scores[i])) for i in range(len(scores)) if not keep[i]]

    if dropped:
        from collections import Counter
        by_file = Counter(name for name, _ in dropped)
        print(f"\n  Dropped {len(dropped)} window(s) that don't match the rest "
              f"(cutoff {cutoff:.3f}):")
        for name, n in by_file.most_common():
            print(f"    {name}: {n} window(s) dropped")
        if keep.sum() < 3:
            print("\n  Too few clips left after dropping outliers. Re-record "
                  "enrollment in a quiet room.")
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
            "backend": enc.backend_name(),
            # bumped whenever the embedding PIPELINE changes (not just the
            # model), since a profile built without silence-trimming isn't
            # comparable to live audio that is trimmed
            "pipeline_version": 4,
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

    if std > 0.10:
        print("\n  NOTE: your remaining clips still vary a lot. Inconsistent mic")
        print("  distance or background noise during recording. Re-recording more")
        print("  consistently would tighten the threshold.")
    # NOTE on scale: ECAPA-TDNN spreads scores across the full range —
    # same-speaker typically 0.5-0.8, different speakers 0.0-0.3. That looks
    # "lower" than Resemblyzer, which crushed everything into 0.7-0.95, but
    # the wide spread is exactly what makes ECAPA able to tell speakers apart.
    # Don't read a 0.6 here as worse than Resemblyzer's 0.86; it isn't.
    if mean < 0.45:
        print("\n  NOTE: genuine similarity is low even for ECAPA. Longer clips")
        print("  recorded in a quiet room would improve separation.")
    else:
        margin = mean - 0.30   # 0.30 ~ where different speakers typically land
        print(f"\n  Separation margin over a typical impostor: {margin:.2f}")
        if margin > 0.25:
            print("  That's healthy — impostors should be rejected reliably.")

    print("\nRun main.py — the wake word and every command will now be checked")
    print("against this voiceprint.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== ENROLLMENT FAILED ===")
        traceback.print_exc()
        sys.exit(1)
