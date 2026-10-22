"""Compare whisper.cpp and faster-whisper on my voice: python3 tools/bench_stt.py

Starts the whisper.cpp server, records a few sentences and transcribes each
with both engines, printing the text and the time each took. If whisper.cpp
won't start it says why (path, binary or model).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice import audio_device

SAMPLES = 4
CLIP_SECONDS = 3.0


def summary(name, times):
    print(f"{name}: mean {sum(times) / len(times):.2f}s  (min {min(times):.2f}s, max {max(times):.2f}s)")


def main():
    print("=== whisper.cpp integration test ===\n")
    if not config.WHISPER_CPP_DIR:
        print("WHISPER_CPP_DIR isn't set in .env, nothing to test. See README.md for setup.")
        return

    from voice import stt_whispercpp
    try:
        t0 = time.time()
        stt_whispercpp.start_server()
        print(f"(startup took {time.time() - t0:.1f}s, this only happens once per run)\n")
    except stt_whispercpp.WhisperCppUnavailable as e:
        print(f"FAILED TO START: {e}\n")
        print("Fix the issue above and re-run.")
        return

    print("Loading faster-whisper for comparison...")
    from faster_whisper import WhisperModel
    fw_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"(comparing against faster-whisper/{config.WHISPER_MODEL}, the current default)\n")

    cpp_times, fw_times = [], []
    for i in range(SAMPLES):
        input(f"[{i + 1}/{SAMPLES}] Press Enter, then say a full sentence "
              f'(real words, not just "hey jarvis")...')
        print("  recording...", end="", flush=True)
        clip = audio_device.record(CLIP_SECONDS)
        print(" done")

        t0 = time.time()
        try:
            cpp_text = stt_whispercpp.transcribe(clip)
            cpp_times.append(time.time() - t0)
            cpp_time = f"{cpp_times[-1]:.2f}s"
        except stt_whispercpp.WhisperCppUnavailable as e:
            cpp_text, cpp_time = f"[FAILED: {e}]", "  ?  "

        t0 = time.time()
        audio = np.frombuffer(clip, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = fw_model.transcribe(audio, language="en", beam_size=1)
        fw_text = " ".join(s.text.strip() for s in segments).strip()
        fw_times.append(time.time() - t0)

        print(f"  whisper.cpp    ({cpp_time}): {cpp_text!r}")
        print(f"  faster-whisper ({fw_times[-1]:.2f}s): {fw_text!r}\n")

    audio_device.shutdown()
    stt_whispercpp.stop_server()

    print("=" * 60)
    print("SUMMARY\n")
    if cpp_times:
        summary("whisper.cpp   ", cpp_times)
    else:
        print("whisper.cpp    : every request failed, see errors above")
    summary("faster-whisper", fw_times)
    print("\nCompare the transcribed text too. The real question is whether")
    print("whisper.cpp gets words right that faster-whisper missed, not just speed.")

    if cpp_times and sum(cpp_times) / len(cpp_times) > sum(fw_times) / len(fw_times) * 1.5:
        print("\nwhisper.cpp was notably slower here. If the accuracy gain isn't worth it,")
        print("drop WHISPER_CPP_MODEL to 'medium' or 'small', or unset WHISPER_CPP_DIR.")


if __name__ == "__main__":
    main()
