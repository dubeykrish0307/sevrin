"""
Run: `python3 voice/test_wake.py`

Exercises the REAL wake path — same ring buffer, same audio gate, same
speaker verification — and reports every number, so a regression is caught
here in 30 seconds instead of showing up as "it stopped recognising me"
after a full app run.

This exists because two separate changes silently broke verification by
punching holes in the audio buffer, and both times the only symptom was
scores quietly collapsing. Run it after ANY change to voice/.
"""
import os
import sys
import time
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice import audio_device, audio_gate
from voice.voice_id import verify as speaker
from voice.wake_word import FRAME_SAMPLES, _join_ring

ATTEMPTS = 5


def capture_clip(seconds):
    """Fill a ring exactly the way wake_word.listen_once does, indices and all."""
    frames_needed = int(config.SAMPLE_RATE * seconds / FRAME_SAMPLES) + 1
    ring = collections.deque(maxlen=frames_needed)
    stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
    try:
        for i in range(frames_needed):
            raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            ring.append((i, raw))
    finally:
        stream.close()
    return _join_ring(ring)


def main():
    print("=== SEVRIN wake-path test ===\n")

    if not speaker.voiceprint_available():
        print("No voiceprint enrolled. Run:")
        print("  python3 voice/voice_id/record.py && python3 voice/voice_id/enroll.py")
        return

    from voice.voice_id import encoder as enc
    cal = speaker.calibration_summary() or {}
    print(f"speaker model : {enc.backend_name()}")
    print(f"voiceprint    : {cal.get('n_clips','?')} clips, "
          f"enrollment mean {cal.get('genuine_mean',0):.3f}, "
          f"stored threshold {cal.get('threshold',0):.3f}")

    print("\nMeasuring room noise — stay quiet for 2s...")
    profile = audio_gate.NoiseProfile()
    profile.calibrate(audio_device.get_pa(), seconds=2.0)

    clip_seconds = float(getattr(config, "VOICE_ID_CLIP_SECONDS", 2.5))
    print(f"\nSay \"hey jarvis\" {ATTEMPTS} times. Each capture is {clip_seconds}s,")
    print("exactly matching the live wake path.\n")

    scores, gates, contiguous_all = [], [], True

    for i in range(ATTEMPTS):
        input(f"  [{i+1}/{ATTEMPTS}] Press Enter, then say \"hey jarvis\"...")
        print("  capturing...", end="", flush=True)
        clip, contiguous = capture_clip(clip_seconds)
        print(" done")

        if not contiguous:
            contiguous_all = False
            print("      BUFFER NOT CONTIGUOUS — this is a bug, not a mic issue")
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
            scores.append(score)
            print(f"      voice: {score:.3f} vs threshold {thr:.3f}  "
                  f"{'PASS' if ok else 'FAIL'}")
        print()

    audio_device.shutdown()

    print("=" * 58)
    print("SUMMARY\n")
    print(f"  buffer contiguity : {'OK' if contiguous_all else 'BROKEN — fix before anything else'}")
    if gates:
        print(f"  gate passed       : {sum(gates)}/{len(gates)}")
    if scores:
        arr = np.array(scores)
        thr = speaker.threshold_for(clip_seconds)
        passed_n = int((arr >= thr).sum())
        print(f"  voice scores      : min {arr.min():.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}")
        print(f"  threshold in use  : {thr:.3f}")
        print(f"  would wake        : {passed_n}/{len(arr)} times")
        print()
        if passed_n == len(arr):
            print("  VERDICT: healthy. Wake should be reliable.")
        elif passed_n == 0:
            print("  VERDICT: nothing passes. Either the voiceprint doesn't match how")
            print("  you actually speak to it (re-record enrollment the way you really")
            print(f"  talk), or the threshold is too high for live audio (your scores")
            print(f"  centre on {arr.mean():.2f}).")
        else:
            print("  VERDICT: borderline. Some attempts pass, some don't — you'll")
            print("  experience this as 'it works sometimes'. Re-record enrollment")
            print("  with more variety in distance and volume.")
        margin = arr.mean() - 0.30
        print(f"\n  margin over a typical impostor (~0.30): {margin:.2f}")
        if margin < 0.15:
            print("  That's thin — a similar-sounding person could get through.")


if __name__ == "__main__":
    main()
