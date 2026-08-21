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

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
PROFILE_DIR = os.path.join(os.path.dirname(__file__), "profile")
PROFILE_PATH = os.path.join(PROFILE_DIR, "voiceprint.npy")       # centroid (compat)
EMBEDDINGS_PATH = os.path.join(PROFILE_DIR, "embeddings.npy")    # per-clip
CALIBRATION_PATH = os.path.join(PROFILE_DIR, "calibration.json")

# How many standard deviations below the genuine mean to place the threshold.
# Larger = more forgiving to you, easier for an impostor. 2.0 accepts ~97.7%
# of genuine attempts if scores are roughly normal.
SAFETY_SIGMAS = 2.0


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

    print("Importing resemblyzer...")
    from resemblyzer import VoiceEncoder, preprocess_wav

    print("Loading speaker-encoder model (one-time download on first run)...")
    encoder = VoiceEncoder()

    files = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav"))
    print(f"Found {len(files)} recordings.\n")

    embeddings = []
    for fname in files:
        path = os.path.join(RECORDINGS_DIR, fname)
        try:
            wav = preprocess_wav(path)
            if len(wav) < 8000:   # under ~0.5s — too short to embed reliably
                print(f"  skipped {fname} (too short)")
                continue
            emb = encoder.embed_utterance(wav)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            print(f"  processed {fname}")
        except Exception as e:
            print(f"  skipped {fname} ({e})")

    if len(embeddings) < 3:
        print("\nNeed at least 3 usable recordings to calibrate. Record more.")
        return

    embeddings = np.stack(embeddings)

    # --- leave-one-out calibration ---------------------------------------
    # Treat each clip as if it were an unknown speaker and score it against
    # the rest. That gives the real distribution of GENUINE scores.
    print("\nCalibrating threshold (leave-one-out)...")
    genuine_scores = []
    for i in range(len(embeddings)):
        others = np.delete(embeddings, i, axis=0)
        genuine_scores.append(score_against(embeddings[i], others))
    genuine_scores = np.array(genuine_scores)

    mean, std = float(genuine_scores.mean()), float(genuine_scores.std())
    threshold = max(0.45, mean - SAFETY_SIGMAS * std)

    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    os.makedirs(PROFILE_DIR, exist_ok=True)
    np.save(PROFILE_PATH, centroid)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({
            "n_clips": int(len(embeddings)),
            "genuine_mean": mean,
            "genuine_std": std,
            "genuine_min": float(genuine_scores.min()),
            "genuine_max": float(genuine_scores.max()),
            "threshold": threshold,
            "safety_sigmas": SAFETY_SIGMAS,
        }, f, indent=2)

    print(f"\n  clips used:        {len(embeddings)}")
    print(f"  genuine scores:    mean {mean:.3f}, std {std:.3f}, "
          f"range {genuine_scores.min():.3f}-{genuine_scores.max():.3f}")
    print(f"  chosen threshold:  {threshold:.3f}")
    print(f"\nSaved to {PROFILE_DIR}/")

    if std > 0.08:
        print("\n  NOTE: your clips vary a lot from each other. That usually means")
        print("  inconsistent distance from the mic or background noise during")
        print("  recording. Re-recording more consistently would tighten the")
        print("  threshold and make impostors easier to reject.")
    if mean < 0.75:
        print("\n  NOTE: genuine similarity is lowish. More/longer clips recorded")
        print("  in a quiet room would improve separation.")

    print("\nRun main.py — the wake word and every command will now be checked")
    print("against this voiceprint.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n=== ENROLLMENT FAILED ===")
        traceback.print_exc()
        sys.exit(1)
