"""
Run: `python3 voice/test_whispercpp.py`

Starts the whisper.cpp server and measures REAL transcription latency and
accuracy on YOUR machine, for YOUR voice — not a benchmark blog post from
someone else's Mac. Transcribes the same clips with faster-whisper too,
so you see both side by side and can judge the actual trade-off yourself.

If whisper.cpp fails to start, this tells you exactly why (wrong path,
missing binary, missing model) rather than a stack trace.
"""
import os
import sys
import time

import pyaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

SAMPLES = 4
CLIP_SECONDS = 3.0


def record(pa, seconds):
    stream = pa.open(rate=config.SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(config.SAMPLE_RATE / 1024 * seconds)):
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.close()
    return b"".join(frames)


def main():
    print("=== whisper.cpp integration test ===\n")

    if not getattr(config, "WHISPER_CPP_DIR", None):
        print("WHISPER_CPP_DIR isn't set in .env — nothing to test.")
        print("See README.md for setup.")
        return

    from voice import stt_whispercpp
    try:
        t0 = time.time()
        stt_whispercpp.start_server()
        print(f"(startup took {time.time()-t0:.1f}s — this only happens once per run)\n")
    except stt_whispercpp.WhisperCppUnavailable as e:
        print(f"FAILED TO START: {e}\n")
        print("Fix the issue above and re-run. Nothing further to test until this works.")
        return

    print("Loading faster-whisper for comparison...")
    from faster_whisper import WhisperModel
    fw_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"(comparing against faster-whisper/{config.WHISPER_MODEL}, your current default)\n")

    pa = pyaudio.PyAudio()
    cpp_times, fw_times = [], []

    for i in range(SAMPLES):
        input(f"[{i+1}/{SAMPLES}] Press Enter, then say a full sentence "
              f"(something with real words, not just \"hey jarvis\")...")
        print("  recording...", end="", flush=True)
        clip = record(pa, CLIP_SECONDS)
        print(" done")

        t0 = time.time()
        try:
            cpp_text = stt_whispercpp.transcribe(clip)
            cpp_dt = time.time() - t0
            cpp_times.append(cpp_dt)
        except stt_whispercpp.WhisperCppUnavailable as e:
            cpp_text, cpp_dt = f"[FAILED: {e}]", None

        t0 = time.time()
        import numpy as np
        audio = np.frombuffer(clip, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = fw_model.transcribe(audio, language="en", beam_size=1)
        fw_text = " ".join(s.text.strip() for s in segments).strip()
        fw_dt = time.time() - t0
        fw_times.append(fw_dt)

        cpp_time_str = f"{cpp_dt:.2f}s" if cpp_dt is not None else "  ?  "
        print(f"  whisper.cpp    ({cpp_time_str}): {cpp_text!r}")
        print(f"  faster-whisper ({fw_dt:.2f}s): {fw_text!r}\n")

    pa.terminate()
    stt_whispercpp.stop_server()

    print("=" * 60)
    print("SUMMARY\n")
    if cpp_times:
        print(f"whisper.cpp    : mean {sum(cpp_times)/len(cpp_times):.2f}s  "
              f"(min {min(cpp_times):.2f}s, max {max(cpp_times):.2f}s)")
    else:
        print("whisper.cpp    : every request failed — see errors above")
    print(f"faster-whisper : mean {sum(fw_times)/len(fw_times):.2f}s  "
          f"(min {min(fw_times):.2f}s, max {max(fw_times):.2f}s)")
    print("\nCompare the actual transcribed TEXT above too — the real question is")
    print("whether whisper.cpp is getting words right that faster-whisper missed,")
    print("not just which one is faster.")

    if cpp_times and (sum(cpp_times)/len(cpp_times)) > (sum(fw_times)/len(fw_times)) * 1.5:
        print("\nNOTE: whisper.cpp was notably slower here. If the accuracy gain isn't")
        print("worth it for your use, either drop WHISPER_CPP_MODEL to 'medium' or")
        print("'small', or unset WHISPER_CPP_DIR to stay on faster-whisper.")


if __name__ == "__main__":
    main()
