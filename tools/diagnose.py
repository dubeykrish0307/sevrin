"""Measure my real voice through my real mic: python3 tools/diagnose.py

Records "hey jarvis" a few times and prints what the noise gate and speaker
verification compute for each clip, then suggests config.py values. The
thresholds in config.py came from this; guessing them (or tuning on
synthetic signals) kept rejecting real speech.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice import audio_device, audio_gate
from voice.voice_id import cohort, encoder
from voice.voice_id import verify as speaker

SAMPLES = 5
CLIP_SECONDS = config.VOICE_ID_CLIP_SECONDS    # same length as the live wake buffer


def print_setup(has_voiceprint):
    print(f"Speaker model: {encoder.backend_name()}")
    if encoder.backend_name() != "ecapa":
        print("  WARNING: running the weak fallback. `pip install speechbrain`")
        print("  then re-run enroll.py for far better accuracy.")
    print()

    if has_voiceprint:
        cal = speaker.calibration_summary() or {}
        print(f"Voiceprint: {cal.get('n_clips', '?')} clips, threshold {cal.get('threshold', '?')}")
        print(f"  (enrollment genuine scores: mean {cal.get('genuine_mean', 0):.3f}, "
              f"range {cal.get('genuine_min', 0):.3f}-{cal.get('genuine_max', 0):.3f})\n")
    else:
        print("!" * 60)
        print("NO VOICEPRINT ENROLLED: anyone can wake SEVRIN and give it commands.")
        print("Fix with:  python3 voice/voice_id/enroll.py")
        print("(recordings in voice/voice_id/recordings/ are reused if they exist)")
        print("!" * 60 + "\n")


def recommend(band_ratios, flatnesses, db_margins, voice_scores, normalised):
    print("\nSuggested config.py values from what was just measured:")
    if band_ratios:
        # Band ratio is the real discriminator, so leave margin below my worst
        # clip. Flatness is only a loose upper bound: voiced speech scores very
        # low, and "max seen + a little" would reject me the moment I speak quietly.
        print(f"  GATE_MIN_BAND_RATIO   = {max(0.10, min(band_ratios) - 0.10):.2f}")
        print(f"  GATE_MAX_FLATNESS     = {max(0.60, min(0.95, max(flatnesses) + 0.35)):.2f}")
        print(f"  GATE_NOISE_MARGIN_DB  = {max(3.0, min(db_margins) - 5.0):.1f}")
    if not voice_scores:
        return

    lowest = min(voice_scores)
    if normalised:
        print(f"  VOICE_ZNORM_THRESHOLD = {max(2.0, lowest - 1.0):.1f}   "
              f"(your lowest was {lowest:.2f}; impostors sit near 0)")
        if lowest < 3:
            print("\n  One or more of your own clips scored low. Re-recording enrolment")
            print("  across the mics, distances and rooms you actually use would help.")
    else:
        print(f"  VOICE_MATCH_THRESHOLD = {max(0.45, lowest - 0.05):.2f}   "
              f"(fallback; calibration.json normally wins)")
        print("\n  If that's much lower than the enrolment threshold, the enrolment")
        print("  clips don't sound like how you talk to SEVRIN. Re-record rather")
        print("  than lowering it.")


def main():
    print("=== SEVRIN recognition diagnostic ===\n")
    print("Step 1: measuring room noise. Stay quiet for 2 seconds...")
    time.sleep(0.4)
    profile = audio_gate.NoiseProfile()
    profile.calibrate(seconds=2.0)
    print(f"  floor_rms = {profile.floor_rms:.6f}\n")

    has_voiceprint = speaker.voiceprint_available()
    print_setup(has_voiceprint)

    print(f'Step 2: say "hey jarvis" {SAMPLES} times when prompted.')
    print(f"Each clip is {CLIP_SECONDS}s, the same as the live wake path, so the numbers compare.")
    print("Vary it like real use: normal, quieter, closer to and further from the mic.")
    print("Include at least one very close, mic distance matters a lot.\n")

    gates, band_ratios, flatnesses, db_margins, voice_scores = [], [], [], [], []
    for i in range(SAMPLES):
        input(f'  [{i + 1}/{SAMPLES}] Press Enter, then say "hey jarvis"...')
        print("  recording...", end="", flush=True)
        clip = audio_device.record(CLIP_SECONDS)
        print(" done")

        passed, detail = audio_gate.looks_like_speech(clip, profile)
        gates.append(passed)
        band_ratios.append(detail.get("speech_band_ratio", 0))
        flatnesses.append(detail.get("spectral_flatness", 1))
        db_margins.append(detail.get("db_above_floor", 0))
        line = (f"      gate: {'PASS' if passed else 'FAIL'}  "
                f"({detail.get('db_above_floor')}dB above floor, "
                f"band {detail.get('speech_band_ratio')}, "
                f"flatness {detail.get('spectral_flatness')})")
        print(line if passed else f"{line}  <- {detail.get('reason')}")

        if has_voiceprint:
            ok, score, thr = speaker.verify(clip)
            if score is None:
                print("      voice: could not score (too short/quiet)")
            else:
                voice_scores.append(speaker.score_value(score))
                print(f"      voice: {speaker.score_text(score)} vs threshold {thr:.3f}  "
                      f"{'PASS' if ok else 'FAIL'}")
        print()

    audio_device.shutdown()

    print("=" * 60)
    print("RESULTS\n")
    print(f"Acoustic gate passed {sum(gates)}/{SAMPLES} times")
    if band_ratios:
        print(f"  speech-band ratio : min {min(band_ratios):.3f}  mean {np.mean(band_ratios):.3f}")
        print(f"  spectral flatness : max {max(flatnesses):.3f}  mean {np.mean(flatnesses):.3f}")
        print(f"  dB above floor    : min {min(db_margins):.1f}  mean {np.mean(db_margins):.1f}")

    normalised = cohort.load() is not None
    if voice_scores:
        print(f"\nSpeaker verification on live clips ({'z' if normalised else 'cosine'} scores):")
        print(f"  min {min(voice_scores):.2f}  mean {np.mean(voice_scores):.2f}  "
              f"max {max(voice_scores):.2f}")
        if normalised:
            thr = config.VOICE_ZNORM_THRESHOLD
        else:
            thr = (speaker.calibration_summary() or {}).get("threshold", config.VOICE_MATCH_THRESHOLD)
        below = sum(1 for v in voice_scores if v < thr)
        if below:
            print(f"  {below}/{len(voice_scores)} of YOUR OWN clips scored below the threshold {thr:.2f}")

    recommend(band_ratios, flatnesses, db_margins, voice_scores, normalised)


if __name__ == "__main__":
    main()
