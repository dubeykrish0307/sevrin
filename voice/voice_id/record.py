"""Step 1 of enrolment: python3 voice/voice_id/record.py

Records the wake phrase several times plus a minute of free speech, into
voice/voice_id/recordings/. Takes 3-4 minutes. Nothing leaves the machine.
"""
import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from voice import audio_device

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings")
SAMPLE_RATE = 16000
REPS_PER_PHRASE = 6
PHRASE_SECONDS = 3.5        # the voiceprint uses 2.5 s windows, so 2 s clips were too short to window
FREE_SPEECH_SECONDS = 60


def record_clip(seconds, path):
    audio = audio_device.record(seconds, rate=SAMPLE_RATE)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio)


def countdown(n=2):
    for i in range(n, 0, -1):
        print(f"  recording in {i}...", end="\r")
        time.sleep(1)
    print("  recording now!        ")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_index = 0

    print("=== Voice enrollment ===")
    print("Recording you saying each wake phrase, then some free speech.")
    print("Everything stays on this machine.\n")
    print(f"Each wake-phrase capture is {PHRASE_SECONDS}s and the free-speech one is "
          f"{FREE_SPEECH_SECONDS}s.\n")
    print("For a strong voiceprint (this decides how well SEVRIN can tell you")
    print("apart from other people):")
    print("  * quiet room, no fan or music running")
    print("  * vary it: some clips close to the mic, some further back; some at")
    print("    normal volume, some quieter; sitting and standing")
    print("  * speak naturally, don't over-enunciate")
    print("Variety matters more than count. A voiceprint from one distance and")
    print("tone will reject you the moment you move.\n")

    for phrase in config.WAKE_PHRASES:
        print(f'\nPhrase: "{phrase}", say it {REPS_PER_PHRASE} times, one at a time.')
        for rep in range(REPS_PER_PHRASE):
            input(f'  [{rep + 1}/{REPS_PER_PHRASE}] Press Enter, then say "{phrase}" '
                  f'(keep talking briefly after: say it, pause, say it again)...')
            countdown()
            record_clip(PHRASE_SECONDS, os.path.join(OUTPUT_DIR, f"clip_{clip_index:03d}.wav"))
            clip_index += 1

    print(f"\nLast step: talk normally for {FREE_SPEECH_SECONDS} seconds. Describe")
    print("your day, read something out loud, whatever. This is the most useful")
    print("clip: continuous natural speech describes your voice far better than")
    print("short repeated phrases, so don't rush it or read in a monotone.")
    input("Press Enter when ready...")
    countdown()
    record_clip(FREE_SPEECH_SECONDS, os.path.join(OUTPUT_DIR, f"clip_{clip_index:03d}.wav"))
    audio_device.shutdown()

    print(f"\nDone. Recorded {clip_index + 1} clips to {OUTPUT_DIR}/")
    print("Next: run `python3 voice/voice_id/enroll.py`")


if __name__ == "__main__":
    main()
