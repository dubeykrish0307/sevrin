"""Quick check of the real wake path: python3 tools/check_wake.py

Same ring buffer, noise gate and speaker verification as the live wake word,
with every number printed. Run it after any change under voice/. Twice a
change broke verification by punching holes in the buffer and the only sign
was scores quietly collapsing; this catches that in 30 seconds.
"""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice import audio_device, audio_gate
from voice.voice_id import cohort, encoder
from voice.voice_id import verify as speaker
from voice.wake_word import FRAME_SAMPLES, _join_ring

ATTEMPTS = 5


def capture_clip(seconds):
    """Fill a ring exactly like WakeWordListener.listen_once does, indices and all."""
    n = int(config.SAMPLE_RATE * seconds / FRAME_SAMPLES) + 1
    ring = collections.deque(maxlen=n)
    stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
    try:
        for i in range(n):
            ring.append((i, stream.read(FRAME_SAMPLES, exception_on_overflow=False)))
    finally:
        stream.close()
    return _join_ring(ring)


def verdict(arr, thr, normalised):
    passed = int((arr >= thr).sum())
    print(f"  voice scores      : min {arr.min():.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}")
    print(f"  threshold in use  : {thr:.3f}")
    print(f"  would wake        : {passed}/{len(arr)} times\n")
    if passed == len(arr):
        print("  VERDICT: healthy. Wake should be reliable.")
    elif passed == 0:
        print("  VERDICT: nothing passes. Either the voiceprint doesn't match how you")
        print("  actually speak to it (re-record enrolment the way you really talk), or")
        print(f"  the threshold is too high for live audio (your scores centre on {arr.mean():.2f}).")
    else:
        print("  VERDICT: borderline. Some attempts pass and some don't, which feels like")
        print("  'it works sometimes'. Re-record enrolment with more variety in distance")
        print("  and volume.")

    if normalised:
        print(f"\n  z-scores: impostors sit near 0, you're averaging {arr.mean():.1f}.")
        if arr.mean() < 4:
            print("  Thinner than ideal, more varied enrolment would help.")
    else:
        margin = arr.mean() - 0.30
        print(f"\n  margin over a typical impostor (~0.30): {margin:.2f}")
        if margin < 0.15:
            print("  That's thin, a similar-sounding person could get through.")


def main():
    print("=== SEVRIN wake-path check ===\n")
    if not speaker.voiceprint_available():
        print("No voiceprint enrolled. Run:")
        print("  python3 voice/voice_id/record.py && python3 voice/voice_id/enroll.py")
        return

    voices = cohort.load()
    if voices is not None:
        src = cohort.source()
        quality = ("realistic human voices" if src == "elevenlabs"
                   else "synthetic system voices, blunter reference")
        print(f"score normalisation: ON ({len(voices)} {src} voices, {quality})")
    else:
        print("score normalisation: OFF, so recognition will be sensitive to a")
        print("  change of mic or room. Re-run enroll.py to build the cohort.")
    cal = speaker.calibration_summary() or {}
    print(f"speaker model : {encoder.backend_name()}")
    print(f"voiceprint    : {cal.get('n_clips', '?')} clips, "
          f"enrollment mean {cal.get('genuine_mean', 0):.3f}, "
          f"stored threshold {cal.get('threshold', 0):.3f}")

    print("\nMeasuring room noise, stay quiet for 2s...")
    profile = audio_gate.NoiseProfile()
    profile.calibrate(seconds=2.0)

    clip_seconds = config.VOICE_ID_CLIP_SECONDS
    print(f'\nSay "hey jarvis" {ATTEMPTS} times. Each capture is {clip_seconds}s, '
          f"same as the live wake path.\n")

    scores, gates, contiguous_all = [], [], True
    for i in range(ATTEMPTS):
        input(f'  [{i + 1}/{ATTEMPTS}] Press Enter, then say "hey jarvis"...')
        print("  capturing...", end="", flush=True)
        clip, contiguous = capture_clip(clip_seconds)
        print(" done")
        if not contiguous:
            contiguous_all = False
            print("      BUFFER NOT CONTIGUOUS: this is a bug, not a mic issue")
            continue

        passed, detail = audio_gate.looks_like_speech(clip, profile)
        gates.append(passed)
        print(f"      gate : {'PASS' if passed else 'FAIL'}  "
              f"({detail.get('db_above_floor')}dB, band {detail.get('speech_band_ratio')}, "
              f"flat {detail.get('spectral_flatness')})"
              + ("" if passed else f"  <- {detail.get('reason')}"))

        ok, score, thr = speaker.verify(clip)
        if score is None:
            print("      voice: unscoreable (too short/quiet)")
        else:
            scores.append(speaker.score_value(score))
            print(f"      voice: {speaker.score_text(score)} vs threshold {thr:.2f}  "
                  f"{'PASS' if ok else 'FAIL'}")
        print()

    audio_device.shutdown()

    print("=" * 58)
    print("SUMMARY\n")
    print(f"  buffer contiguity : {'OK' if contiguous_all else 'BROKEN, fix before anything else'}")
    if gates:
        print(f"  gate passed       : {sum(gates)}/{len(gates)}")
    if scores:
        normalised = voices is not None
        thr = config.VOICE_ZNORM_THRESHOLD if normalised else speaker.threshold_for(clip_seconds)
        verdict(np.array(scores), thr, normalised)


if __name__ == "__main__":
    main()
