"""
Run this. Ctrl+C to quit.

Same pipeline as before, with timing instrumentation added so we can see
exactly where time is going: record -> transcribe -> first Claude token ->
first packet ready for TTS -> first audio actually playing.
"""
import queue
import threading
import time

import config
from voice.wake_word import WakeWordListener
from voice.listener import UtteranceListener
from voice.stt import transcribe, get_model
from brain.llm import ask_streaming_packets
from voice.tts import prepare, play


def _fetch_worker(text_queue, audio_queue, timings):
    first = True
    while True:
        text = text_queue.get()
        if text is None:
            audio_queue.put(None)
            break
        try:
            prepared = prepare(text)
            if first:
                timings["first_packet_prepared"] = time.time()
                first = False
        except Exception as e:
            print(f"\n[tts error] {e}")
            prepared = None
        audio_queue.put(prepared)


def _playback_worker(audio_queue, timings):
    first = True
    while True:
        prepared = audio_queue.get()
        if prepared is None:
            break
        if first:
            timings["first_audio_start"] = time.time()
            first = False
        play(prepared)


def handle_turn(utterance_listener: UtteranceListener, max_seconds: float = None) -> bool:
    timings = {}
    timings["t0_start_recording"] = time.time()

    print("[listening] speak now...")
    audio = utterance_listener.record_utterance(max_seconds=max_seconds)
    timings["t1_recording_done"] = time.time()

    print("[transcribing]...")
    text = transcribe(audio)
    timings["t2_transcribed"] = time.time()
    if not text:
        return False

    print(f"[you] {text}")
    print("[sevrin] ", end="", flush=True)

    text_queue = queue.Queue()
    audio_queue = queue.Queue()

    fetch_thread = threading.Thread(target=_fetch_worker, args=(text_queue, audio_queue, timings), daemon=True)
    playback_thread = threading.Thread(target=_playback_worker, args=(audio_queue, timings), daemon=True)
    fetch_thread.start()
    playback_thread.start()

    first_token = True
    for packet in ask_streaming_packets(text):
        if first_token:
            timings["t3_first_packet_from_claude"] = time.time()
            first_token = False
        print(packet, end=" ", flush=True)
        text_queue.put(packet)

    text_queue.put(None)
    fetch_thread.join()
    playback_thread.join()
    print()

    # --- print the breakdown ---
    t0 = timings["t0_start_recording"]
    print("\n[timing]")
    print(f"  recording:                 {timings['t1_recording_done'] - t0:.2f}s")
    print(f"  transcription:              {timings['t2_transcribed'] - timings['t1_recording_done']:.2f}s")
    if "t3_first_packet_from_claude" in timings:
        print(f"  claude -> first packet:     {timings['t3_first_packet_from_claude'] - timings['t2_transcribed']:.2f}s")
    if "first_packet_prepared" in timings:
        print(f"  tts fetch (first packet):   {timings['first_packet_prepared'] - timings.get('t3_first_packet_from_claude', timings['t2_transcribed']):.2f}s")
    if "first_audio_start" in timings:
        print(f"  TOTAL to first audio:       {timings['first_audio_start'] - t0:.2f}s")
    print()

    return True


def main():
    print("[stt] loading whisper model...")
    get_model()  # eager load — otherwise your first-ever transcription eats the model load time

    utterance_listener = UtteranceListener()
    print("[stt] measuring room noise (stay quiet for a second)...")
    utterance_listener.calibrate_noise()

    wake = WakeWordListener()
    wake.set_noise_profile(utterance_listener.noise)
    wake.start()

    try:
        while True:
            triggered_by = wake.listen_once()
            print(f"\n[wake] triggered by '{triggered_by}'")

            first_turn = True
            while True:
                # First command after waking gets the full recording window
                # (you just said the wake word, give it a beat). Every turn
                # after that only waits FOLLOWUP_WINDOW_SECONDS of silence
                # before dropping back to wake-word listening.
                timeout = config.MAX_UTTERANCE_SECONDS if first_turn else config.FOLLOWUP_WINDOW_SECONDS
                first_turn = False
                said_something = handle_turn(utterance_listener, max_seconds=timeout)
                if not said_something:
                    break

    except KeyboardInterrupt:
        print("\n[sevrin] shutting down.")
    finally:
        wake.close()


if __name__ == "__main__":
    main()
