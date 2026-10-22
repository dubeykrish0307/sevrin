"""Terminal-only SEVRIN: wake word, command, Claude, speech. No desktop app,
no proactive speech. Prints a latency breakdown after every turn.

    python3 main.py      (Ctrl+C to quit)
"""
import queue
import threading
import time

import config
from brain.llm import ask_streaming_packets
from voice import tts
from voice.listener import UtteranceListener
from voice.stt import get_model, transcribe
from voice.wake_word import WakeWordListener


def _play(packets: queue.Queue, timings: dict):
    while True:
        text = packets.get()
        if text is None:
            break
        timings.setdefault("first_audio", time.time())
        tts.play(text)


def _print_timings(t: dict):
    print("\n[timing]")
    print(f"  recording:                 {t['recorded'] - t['start']:.2f}s")
    print(f"  transcription:             {t['transcribed'] - t['recorded']:.2f}s")
    if "first_packet" in t:
        print(f"  claude -> first packet:    {t['first_packet'] - t['transcribed']:.2f}s")
    if "first_audio" in t:
        print(f"  TOTAL to first audio:      {t['first_audio'] - t['start']:.2f}s")
    print()


def handle_turn(listener: UtteranceListener, max_seconds: float = None) -> bool:
    """One command and reply. Returns False if nothing was said."""
    timings = {"start": time.time()}
    print("[listening] speak now...")
    audio = listener.record_utterance(max_seconds=max_seconds)
    timings["recorded"] = time.time()

    print("[transcribing]...")
    text = transcribe(audio)
    timings["transcribed"] = time.time()
    if not text:
        return False

    print(f"[you] {text}")
    print("[sevrin] ", end="", flush=True)

    packets = queue.Queue()
    player = threading.Thread(target=_play, args=(packets, timings), daemon=True)
    player.start()
    for packet in ask_streaming_packets(text):
        timings.setdefault("first_packet", time.time())
        print(packet, end=" ", flush=True)
        packets.put(packet)
    packets.put(None)
    player.join()
    print()

    _print_timings(timings)
    return True


def main():
    print("[stt] loading whisper model...")
    get_model()     # load now, or the first command pays for it

    listener = UtteranceListener()
    print("[stt] measuring room noise (stay quiet for a second)...")
    listener.calibrate_noise()

    wake = WakeWordListener()
    wake.set_noise_profile(listener.noise)
    wake.start()

    try:
        while True:
            print(f"\n[wake] triggered by '{wake.listen_once()}'")
            # the first command gets the full window; after that, going quiet for
            # FOLLOWUP_WINDOW_SECONDS drops back to waiting for the wake word
            timeout = config.MAX_UTTERANCE_SECONDS
            while handle_turn(listener, max_seconds=timeout):
                timeout = config.FOLLOWUP_WINDOW_SECONDS
    except KeyboardInterrupt:
        print("\n[sevrin] shutting down.")


if __name__ == "__main__":
    main()
