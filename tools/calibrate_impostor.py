"""Set the speaker threshold from real people: python3 tools/calibrate_impostor.py

Records me saying "hey jarvis" a few times, then someone else in the room,
and suggests a VOICE_ZNORM_THRESHOLD that sits between the two measured
distributions. The current 3.0 came from a simulated cohort, which is why
friends were getting accepted. Needs a second person; alone it can only
report my own scores.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice import audio_device
from voice.voice_id import cohort
from voice.voice_id import verify as speaker

SAMPLES_PER_PERSON = 5
CLIP_SECONDS = 2.5


def collect_scores(label):
    scores = []
    for i in range(SAMPLES_PER_PERSON):
        input(f'  [{i + 1}/{SAMPLES_PER_PERSON}] Press Enter, then {label} says "hey jarvis"...')
        print("  recording...", end="", flush=True)
        clip = audio_device.record(CLIP_SECONDS)
        print(" done")
        value = speaker.score_value(speaker.voice_similarity(clip))
        if value is None:
            print("     couldn't score (too short/quiet), try again and speak clearly")
            continue
        scores.append(value)
        print(f"     z = {value:.2f}")
    return scores


def main():
    if not speaker.voiceprint_available():
        print("No voiceprint enrolled. Run record.py and enroll.py first.")
        return
    voices = cohort.load()
    if voices is None:
        print("No impostor cohort, so scores are raw cosine rather than z-scores and")
        print("this calibration doesn't apply. Run enroll.py (it builds the cohort)")
        print("and check ffmpeg is installed.")
        return
    print(f"Cohort: {len(voices)} voices ({cohort.source()})\n")

    print("=== Round 1: YOU ===")
    genuine = collect_scores("you")
    print("\n=== Round 2: SOMEONE ELSE ===")
    print("Hand the mic to another person now.")
    impostor = collect_scores("they")
    audio_device.shutdown()

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
        print(f"\n  WARNING: your lowest score ({g.min():.2f}) is at or below their highest "
              f"({imp.max():.2f}).")
        print("  The distributions overlap, so no threshold cleanly separates you on this")
        print("  data. Either enrolment needs more variety or this person sounds unusually")
        print("  like you. More samples from both would sharpen it.")
        threshold = (g.min() + imp.max()) / 2     # least bad option
    else:
        # in the gap, nearer their ceiling than my floor
        threshold = imp.max() + (g.min() - imp.max()) * 0.35

    print(f"\nRECOMMENDED VOICE_ZNORM_THRESHOLD = {threshold:.2f}")
    print("(measured from real voices, set it in config.py)")


if __name__ == "__main__":
    main()
