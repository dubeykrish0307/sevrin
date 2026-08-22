"""
Run: `python3 voice/diagnose.py`

Measures YOUR real voice through YOUR real microphone and reports exactly
what every stage of the recognition pipeline computes. Built because tuning
these thresholds by guesswork (or against synthetic test signals) produced
values that silently rejected genuine speech.

It reports, for each thing you say:
  * the acoustic gate's measurements and whether they passed
  * your speaker-verification score against the enrolled voiceprint
  * whether the wake word / a command would have been accepted

At the end it recommends threshold values based on what it actually saw.
"""
import os
import sys
import time

import numpy as np
import pyaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice import audio_gate
from voice.voice_id import verify as speaker

SAMPLES = 5
# Match the LIVE wake path's buffer length exactly. An earlier version used a
# different duration, and the resulting numbers didn't match what the wake
# word actually measured — which sent tuning in the wrong direction. If you
# change VOICE_ID_CLIP_SECONDS, this follows it automatically.
CLIP_SECONDS = float(getattr(config, "VOICE_ID_CLIP_SECONDS", 2.5))


def record(pa, seconds):
    stream = pa.open(rate=config.SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(config.SAMPLE_RATE / 1024 * seconds)):
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.close()
    return b"".join(frames)


def main():
    pa = pyaudio.PyAudio()

    print("=== SEVRIN recognition diagnostic ===\n")
    print("Step 1: measuring room noise. Stay quiet for 2 seconds...")
    time.sleep(0.4)
    profile = audio_gate.NoiseProfile()
    profile.calibrate(pa, seconds=2.0)
    print(f"  floor_rms = {profile.floor_rms:.6f}\n")

    from voice.voice_id import encoder as enc
    print(f"Speaker model: {enc.backend_name()}")
    if enc.backend_name() != "ecapa":
        print("  WARNING: running the weak fallback. `pip install speechbrain`")
        print("  then re-run enroll.py for far better accuracy.")
    print()

    has_print = speaker.voiceprint_available()
    if has_print:
        cal = speaker.calibration_summary() or {}
        print(f"Voiceprint: {cal.get('n_clips', '?')} clips, "
              f"threshold {cal.get('threshold', '?')}")
        print(f"  (enrollment genuine scores: mean {cal.get('genuine_mean', 0):.3f}, "
              f"range {cal.get('genuine_min', 0):.3f}-{cal.get('genuine_max', 0):.3f})\n")
    else:
        print("!" * 60)
        print("NO VOICEPRINT ENROLLED — SEVRIN IS CURRENTLY UNPROTECTED.")
        print("Anyone can wake it and issue commands. Fix with:")
        print("    python3 voice/voice_id/enroll.py")
        print("(your recordings are already in voice/voice_id/recordings/,")
        print(" so you shouldn't need to record again)")
        print("!" * 60 + "\n")

    print(f"Step 2: say \"hey jarvis\" {SAMPLES} times when prompted.")
    print(f"Each clip is {CLIP_SECONDS}s — the same length the live wake path uses,")
    print("so these numbers are directly comparable to what it measures.")
    print("Vary it like real use: normal, quieter, closer and further from the mic.")
    print("Mic distance matters a lot — include at least one very close.\n")

    gate_results = []
    voice_scores = []
    band_ratios = []
    flatnesses = []
    db_margins = []

    for i in range(SAMPLES):
        input(f"  [{i+1}/{SAMPLES}] Press Enter, then say \"hey jarvis\"...")
        print("  recording...", end="", flush=True)
        clip = record(pa, CLIP_SECONDS)
        print(" done")

        passed, detail = audio_gate.looks_like_speech(clip, profile)
        gate_results.append(passed)
        band_ratios.append(detail.get("speech_band_ratio", 0))
        flatnesses.append(detail.get("spectral_flatness", 1))
        db_margins.append(detail.get("db_above_floor", 0))

        line = (f"      gate: {'PASS' if passed else 'FAIL'}  "
                f"({detail.get('db_above_floor')}dB above floor, "
                f"band {detail.get('speech_band_ratio')}, "
                f"flatness {detail.get('spectral_flatness')})")
        if not passed:
            line += f"  <- {detail.get('reason')}"
        print(line)

        if has_print:
            ok, score, thr = speaker.verify(clip)
            if score is None:
                print("      voice: could not score (too short/quiet)")
            else:
                voice_scores.append(score)
                print(f"      voice: {score:.3f} vs threshold {thr:.3f}  "
                      f"{'PASS' if ok else 'FAIL'}")
        print()

    pa.terminate()

    # ---------------- recommendations ----------------
    print("=" * 60)
    print("RESULTS\n")
    passes = sum(gate_results)
    print(f"Acoustic gate passed {passes}/{SAMPLES} times")
    if band_ratios:
        print(f"  speech-band ratio : min {min(band_ratios):.3f}  "
              f"mean {np.mean(band_ratios):.3f}")
        print(f"  spectral flatness : max {max(flatnesses):.3f}  "
              f"mean {np.mean(flatnesses):.3f}")
        print(f"  dB above floor    : min {min(db_margins):.1f}  "
              f"mean {np.mean(db_margins):.1f}")

    if voice_scores:
        print(f"\nSpeaker verification on live clips:")
        print(f"  min {min(voice_scores):.3f}  mean {np.mean(voice_scores):.3f}  "
              f"max {max(voice_scores):.3f}")
        cal = speaker.calibration_summary() or {}
        thr = cal.get("threshold", config.VOICE_MATCH_THRESHOLD)
        below = sum(1 for v in voice_scores if v < thr)
        if below:
            print(f"  {below}/{len(voice_scores)} of YOUR OWN clips scored below "
                  f"the threshold {thr:.3f}")

    print("\nRECOMMENDED config.py values based on what was just measured:")
    if band_ratios:
        # Band ratio is the discriminator that matters: fans/rumble score low
        # (~0.25), voice scores high. Leave real margin below your worst sample.
        rec_band = max(0.10, min(band_ratios) - 0.10)
        # Flatness is an UPPER bound, and voiced speech scores very low — so
        # "observed max + small delta" would produce an absurdly tight limit
        # that rejects you the moment you speak more quietly or add room noise.
        # Keep a generous floor; this is a sanity check, not the discriminator.
        rec_flat = max(0.60, min(0.95, max(flatnesses) + 0.35))
        rec_db = max(3.0, min(db_margins) - 5.0)
        print(f"  GATE_MIN_BAND_RATIO   = {rec_band:.2f}")
        print(f"  GATE_MAX_FLATNESS     = {rec_flat:.2f}")
        print(f"  GATE_NOISE_MARGIN_DB  = {rec_db:.1f}")
    if voice_scores:
        rec_voice = max(0.45, min(voice_scores) - 0.05)
        print(f"  VOICE_MATCH_THRESHOLD = {rec_voice:.2f}   "
              f"(fallback; calibration.json normally wins)")
        print("\n  If that recommended voice threshold is much lower than your")
        print("  enrollment threshold, your enrollment clips don't resemble how")
        print("  you actually speak to SEVRIN — re-record enrollment the way you")
        print("  really talk to it, rather than lowering the threshold.")


if __name__ == "__main__":
    main()
