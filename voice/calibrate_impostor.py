"""
Run: `python3 voice/calibrate_impostor.py`

Every threshold decision so far in this system has gone wrong exactly
once, in exactly the same way: guessed from theory or simulation, then
proven wrong by reality. VOICE_ZNORM_THRESHOLD = 3.0 was set from a
synthetic simulation of speaker embeddings — not from an actual other
human being. That's why friends were getting accepted.

This script fixes that by measuring the real thing: it records YOU saying
"hey jarvis" a few times, then records SOMEONE ELSE — an actual friend, in
the room — saying it too. It reports both real distributions and
recommends a threshold that genuinely separates them, because it saw them
both, not because it assumed a shape for them.

Needs someone else physically present. If you run it alone, it'll still
report your own numbers, but can't give you a real threshold.
"""
import os
import sys

import numpy as np
import pyaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice.voice_id import verify as speaker
from voice.voice_id import cohort as cohort_mod

SAMPLES_PER_PERSON = 5
CLIP_SECONDS = 2.5


def record(pa, seconds):
    stream = pa.open(rate=config.SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(config.SAMPLE_RATE / 1024 * seconds)):
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.close()
    return b"".join(frames)


def collect_scores(pa, label):
    scores = []
    for i in range(SAMPLES_PER_PERSON):
        input(f"  [{i+1}/{SAMPLES_PER_PERSON}] Press Enter, then {label} says \"hey jarvis\"...")
        print("  recording...", end="", flush=True)
        clip = record(pa, CLIP_SECONDS)
        print(" done")
        result = speaker.voice_similarity(clip)
        val = speaker.score_value(result)
        if val is None:
            print("     couldn't score (too short/quiet) — try again, speak clearly")
            continue
        scores.append(val)
        print(f"     z = {val:.2f}")
    return scores


def main():
    if not speaker.voiceprint_available():
        print("No voiceprint enrolled. Run record.py and enroll.py first.")
        return
    coh = cohort_mod.load()
    if coh is None:
        print("No impostor cohort built — voice_similarity will return raw cosine,")
        print("not a z-score, so this calibration doesn't apply. Run enroll.py")
        print("(it builds the cohort automatically) or check ffmpeg is installed.")
        return
    print(f"Cohort: {len(coh)} voices ({cohort_mod.source()})\n")

    pa = pyaudio.PyAudio()

    print("=== Round 1: YOU ===")
    genuine = collect_scores(pa, "you")

    print("\n=== Round 2: SOMEONE ELSE ===")
    print("Hand the mic to an actual other person now — a friend, whoever's around.")
    impostor = collect_scores(pa, "they")

    pa.terminate()

    print("\n" + "=" * 60)
    print("RESULTS\n")

    if genuine:
        g = np.array(genuine)
        print(f"YOU:   min {g.min():.2f}  mean {g.mean():.2f}  max {g.max():.2f}")
    if impostor:
        imp = np.array(impostor)
        print(f"THEM:  min {imp.min():.2f}  mean {imp.mean():.2f}  max {imp.max():.2f}")

    if not genuine or not impostor:
        print("\nNeed at least one usable sample from each person to recommend anything.")
        return

    if g.min() <= imp.max():
        print(f"\n  WARNING: your lowest score ({g.min():.2f}) is at or below their")
        print(f"  highest ({imp.max():.2f}) — the two distributions overlap. No single")
        print(f"  threshold cleanly separates you from this person on this data. This")
        print(f"  usually means either your enrollment needs more variety, or this")
        print(f"  particular person's voice is unusually close to yours. More samples")
        print(f"  from both would sharpen this.")
        # split the difference — the least-bad option when they overlap
        threshold = (g.min() + imp.max()) / 2
    else:
        # sits in the real gap between the two measured distributions,
        # closer to the impostor's ceiling for safety margin
        gap = g.min() - imp.max()
        threshold = imp.max() + gap * 0.35

    print(f"\nRECOMMENDED VOICE_ZNORM_THRESHOLD = {threshold:.2f}")
    print("(measured from real data, not simulated — set this in config.py)")


if __name__ == "__main__":
    main()
