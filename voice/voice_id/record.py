"""
Run this first: `python3 voice/voice_id/record.py`

Records you saying each wake phrase several times, plus a stretch of
natural free speech. More variety = a more reliable voiceprint. Takes
about 3-4 minutes. Everything stays in voice/voice_id/recordings/ on this
machine — nothing is sent anywhere.
"""
import os
import sys
import time
import wave
import pyaudio

# voice/voice_id/record.py -> voice/voice_id -> voice -> project root.
# One level deeper than before now that voice_id lives under voice/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings")
SAMPLE_RATE = 16000
REPS_PER_PHRASE = 10
FREE_SPEECH_SECONDS = 45


def record_clip(pa, seconds, path):
    stream = pa.open(rate=SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
                      input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(int(SAMPLE_RATE / 1024 * seconds)):
        frames.append(stream.read(1024, exception_on_overflow=False))
    stream.close()

    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def countdown(n=2):
    for i in range(n, 0, -1):
        print(f"  recording in {i}...", end="\r")
        time.sleep(1)
    print("  recording now!        ")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pa = pyaudio.PyAudio()
    clip_index = 0

    print("=== Voice enrollment ===")
    print("Recording you saying each wake phrase, then some free speech.")
    print("Everything stays on this machine.\n")
    print("HOW TO GET A STRONG VOICEPRINT — this decides how well SEVRIN can")
    print("tell you apart from other people, so it's worth doing properly:")
    print("  * quiet room, no fan or music running")
    print("  * VARY it deliberately: some clips close to the mic, some further")
    print("    back; some normal volume, some quieter; sitting and standing")
    print("  * speak naturally, don't over-enunciate")
    print("The variety matters more than the count — a voiceprint built only")
    print("from one distance and tone will reject you the moment you move.\n")

    for phrase in config.WAKE_PHRASES:
        print(f'\nPhrase: "{phrase}" — say it {REPS_PER_PHRASE} times, one at a time.')
        for rep in range(REPS_PER_PHRASE):
            input(f'  [{rep + 1}/{REPS_PER_PHRASE}] Press Enter, then say "{phrase}"...')
            countdown()
            path = os.path.join(OUTPUT_DIR, f"clip_{clip_index:03d}.wav")
            record_clip(pa, 2.0, path)
            clip_index += 1

    print(f"\nLast step: talk normally for {FREE_SPEECH_SECONDS} seconds — describe")
    print("your day, read something out loud, whatever. This is the most")
    print("valuable clip of the lot: continuous natural speech characterises")
    print("your voice far better than short repeated phrases, so don't rush it")
    print("and don't read in a flat monotone.")
    input("Press Enter when ready...")
    countdown()
    path = os.path.join(OUTPUT_DIR, f"clip_{clip_index:03d}.wav")
    record_clip(pa, FREE_SPEECH_SECONDS, path)

    pa.terminate()
    print(f"\nDone. Recorded {clip_index + 1} clips to {OUTPUT_DIR}/")
    print("Next: run `python3 voice/voice_id/enroll.py`")


if __name__ == "__main__":
    main()