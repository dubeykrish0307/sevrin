"""
Minimal always-on backend for the desktop UI (Phase 1).

Runs the existing voice loop (wake word -> STT -> Claude -> TTS) in a
background thread, exactly as main.py did, and broadcasts what's happening
to any connected UI over a local WebSocket server so the app can show
live state/transcript/orb-animation instead of a terminal.

Also accepts typed text from the UI as a fallback to voice — same
pipeline, just skips the mic.

WebSocket protocol (backend -> UI):
  {"type": "state", "value": "idle"|"listening"|"thinking"|"speaking"}
  {"type": "transcript", "role": "user", "text": "..."}            (one-shot, user turns)
  {"type": "transcript_start", "role": "assistant"}                (assistant reply begins)
  {"type": "transcript_delta", "text": "..."}                      (one per packet, as generated)
  {"type": "transcript_end"}                                       (assistant reply complete)
  {"type": "amplitude", "value": 0.0-1.0}                          (while speaking — see note below)
  {"type": "error", "message": "..."}

(UI -> backend):
  {"type": "text_input", "text": "..."}

NOTE on amplitude: for the elevenlabs backend this is now REAL — the TTS
layer streams raw PCM and computes the actual loudness of each chunk as it
plays, so the orb moves to Jarvis's real voice. Only the local `say`
backend (no tappable stream) falls back to a text-length estimate.
"""
import asyncio
import json
import os
import queue
import random
import sys
import threading
import time

import websockets

# backend/service.py -> backend -> project root. Needed because running
# `python3 backend/service.py` only puts backend/'s own directory on
# sys.path, not the project root where config.py actually lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from voice.wake_word import WakeWordListener
from voice.listener import UtteranceListener
from voice.stt import transcribe, get_model
from voice.barge_in import BargeInListener
from voice import audio_device
from brain.llm import ask_streaming_packets
from brain import llm as brain_llm
from brain import store as memory_store
from voice import tts

HOST = "localhost"
PORT = 8765

_clients = set()
_loop = None
# shared mic listener, set up in _voice_loop; _run_turn needs it to capture
# the rest of an interruption after the short barge-in clip
utterance_listener_ref = [None]
# ONE barge-in listener reused across turns. Constructing a new one per turn
# (each with its own PyAudio instance, terminated on close) is what wedged
# CoreAudio into returning permanent silence.
barge_listener_ref = [None]  # the asyncio event loop, captured at startup so the voice
              # thread (which is plain blocking code) can hand events
              # across to it safely


def emit(event: dict):
    """Thread-safe: call from any thread, including the blocking voice
    loop thread, to broadcast an event to all connected UI clients."""
    if _loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast(event), _loop)


async def _broadcast(event: dict):
    if not _clients:
        return
    message = json.dumps(event)
    dead = set()
    for ws in _clients:
        try:
            await ws.send(message)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def _handler(websocket):
    _clients.add(websocket)
    # send current memory state so the HUD shows it immediately on connect
    try:
        await websocket.send(json.dumps({"type": "memory_stats", "data": memory_store.stats()}))
    except Exception:
        pass
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") == "text_input" and msg.get("text", "").strip():
                threading.Thread(target=_run_turn_chained, args=(msg["text"].strip(),), daemon=True).start()
    finally:
        _clients.discard(websocket)


def _speak_with_animation(text: str):
    """Plays TTS audio while driving the orb.

    For the elevenlabs backend the orb reacts to REAL audio: tts.play
    computes the actual RMS loudness of each PCM chunk as it plays and
    hands it back through the callback, which we forward straight to the
    UI. Loud syllables punch, pauses go still — genuinely voice-reactive.

    The local `say` backend has no tappable audio stream, so there we fall
    back to a procedural envelope estimated from text length (the old
    behavior), which is the best possible for that backend."""
    if config.TTS_BACKEND == "elevenlabs":
        tts.play(text, on_amplitude=lambda level: emit({"type": "amplitude", "value": round(level, 3)}))
        return

    # local-backend fallback: estimated envelope, no real audio to read
    stop = threading.Event()

    def animate():
        est_seconds = max(0.8, len(text) / 15.0)
        end = time.time() + est_seconds
        level = 0.4
        while time.time() < end and not stop.is_set():
            level += random.uniform(-0.25, 0.25)
            level = max(0.15, min(1.0, level))
            emit({"type": "amplitude", "value": round(level, 2)})
            time.sleep(0.06)

    anim_thread = threading.Thread(target=animate, daemon=True)
    anim_thread.start()
    try:
        tts.play(text)
    finally:
        stop.set()
        anim_thread.join(timeout=0.5)
        emit({"type": "amplitude", "value": 0.0})


def _run_turn_chained(user_text: str):
    """Entry point for typed input. If SEVRIN gets interrupted while
    answering, the interruption becomes the next turn — same behaviour as
    the voice path, so typing and speaking don't diverge."""
    interruption = _run_turn(user_text)
    while interruption is not None:
        context = {"spoken": interruption["spoken"], "unspoken": interruption["unspoken"]}
        interruption = _run_turn(interruption["heard"], interruption=context)


def _run_turn(user_text: str, interruption: dict = None):
    """Runs one exchange, pipelined so Claude keeps generating the next
    packet while the current one plays.

    Interruptible: while SEVRIN speaks, a BargeInListener watches the mic.
    If Krish talks over him, playback is killed mid-sentence and we record
    exactly what got said versus what didn't — that split is what lets
    SEVRIN judge whether the unfinished part still mattered (see
    brain/llm.py's _interruption_context).

    Returns a dict describing an interruption if one occurred, else None."""
    emit({"type": "transcript", "role": "user", "text": user_text})
    emit({"type": "state", "value": "thinking"})
    print("[sevrin] ", end="", flush=True)

    tts.clear_stop()

    text_queue = queue.Queue()
    audio_queue = queue.Queue()

    # what actually left the speakers, versus what was still queued
    spoken_packets = []
    all_packets = []
    spoken_lock = threading.Lock()

    interrupted = threading.Event()
    if barge_listener_ref[0] is None:
        barge_listener_ref[0] = BargeInListener()
    barge = barge_listener_ref[0]
    # reuse the room profile the main listener already measured — barge-in
    # can't calibrate itself while audio is playing
    _ul = utterance_listener_ref[0]
    if _ul is not None:
        barge.set_noise_profile(_ul.noise)

    def on_interrupt():
        interrupted.set()
        tts.stop_playback()
        emit({"type": "interrupted"})

    def fetch_worker():
        while True:
            text = text_queue.get()
            if text is None:
                audio_queue.put(None)
                break
            if interrupted.is_set():
                continue          # don't bother preparing audio we won't play
            try:
                prepared = tts.prepare(text)
            except Exception as e:
                emit({"type": "error", "message": f"tts error: {e}"})
                prepared = None
            audio_queue.put(prepared)

    def playback_worker():
        first = True
        while True:
            prepared = audio_queue.get()
            if prepared is None:
                break
            if interrupted.is_set():
                continue          # drain the queue without playing
            if first:
                emit({"type": "state", "value": "speaking"})
                first = False
                # only start listening for interruptions once audio is
                # actually playing, so we don't catch the tail of his own
                # command as an "interruption"
                barge.start(on_interrupt)
            _speak_with_animation(prepared)
            if not interrupted.is_set():
                with spoken_lock:
                    spoken_packets.append(prepared)

    fetch_thread = threading.Thread(target=fetch_worker, daemon=True)
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    fetch_thread.start()
    playback_thread.start()

    emit({"type": "transcript_start", "role": "assistant"})
    for packet in ask_streaming_packets(user_text, interruption=interruption):
        all_packets.append(packet)
        if interrupted.is_set():
            # stop pulling more from Claude — he's been cut off
            break
        print(packet, end=" ", flush=True)
        emit({"type": "transcript_delta", "text": packet})
        text_queue.put(packet)
    emit({"type": "transcript_end"})
    print()

    text_queue.put(None)
    fetch_thread.join(timeout=5)
    playback_thread.join(timeout=5)
    barge.stop()

    if not interrupted.is_set():
        barge.stop()   # stop listening, but keep the object (see barge_listener_ref)
        emit({"type": "state", "value": "idle"})
        return None

    # --- an interruption happened: work out what he did and didn't say ---
    with spoken_lock:
        spoken = " ".join(spoken_packets)
    unspoken = " ".join(p for p in all_packets if p not in spoken_packets)

    print(f"\n[interrupt] cut off after: {spoken[:60]!r}...")

    # transcribe what Krish said over him
    clip = barge.captured_audio
    barge.stop()
    heard = ""
    try:
        if clip:
            heard = transcribe(clip)
    except Exception as e:
        print(f"[interrupt] couldn't transcribe interruption: {e}")

    # The barge-in clip is short and may have caught only part of it — give
    # him a moment to finish the thought before responding.
    emit({"type": "state", "value": "listening"})
    try:
        more = utterance_listener_ref[0].record_utterance(max_seconds=config.FOLLOWUP_WINDOW_SECONDS)
        tail = transcribe(more) if more else ""
        if tail:
            heard = f"{heard} {tail}".strip() if heard else tail
    except Exception as e:
        print(f"[interrupt] couldn't capture the rest: {e}")

    if not heard:
        # he made a noise but nothing intelligible — treat as "stop"
        heard = "(stopped you, said nothing further)"

    print(f"[interrupt] he said: {heard}")

    return {"spoken": spoken, "unspoken": unspoken, "heard": heard}


def _voice_loop():
    print("[backend] loading whisper model...")
    get_model()

    # Calibrate the room BEFORE starting the wake listener, and share one
    # profile everywhere — previously the wake listener calibrated first,
    # before the mic was warm, and measured silence.
    utterance_listener = UtteranceListener()
    print("[backend] measuring room noise (stay quiet for a second)...")
    utterance_listener.calibrate_noise()
    utterance_listener_ref[0] = utterance_listener

    wake = WakeWordListener()
    wake.set_noise_profile(utterance_listener.noise)
    wake.start()
    emit({"type": "state", "value": "idle"})

    try:
        while True:
            emit({"type": "state", "value": "idle"})
            triggered_by = wake.listen_once()
            print(f"\n[backend] wake triggered by '{triggered_by}'")
            emit({"type": "state", "value": "listening"})

            first_turn = True
            pending_interruption = None

            while True:
                if pending_interruption is not None:
                    # He cut SEVRIN off. Whatever he said over him becomes the
                    # next turn, carrying the interrupted state as context so
                    # SEVRIN can judge whether the unfinished part mattered.
                    text = pending_interruption["heard"]
                    context = {
                        "spoken": pending_interruption["spoken"],
                        "unspoken": pending_interruption["unspoken"],
                    }
                    pending_interruption = _run_turn(text, interruption=context)
                    continue

                timeout = config.MAX_UTTERANCE_SECONDS if first_turn else config.FOLLOWUP_WINDOW_SECONDS
                first_turn = False

                print("[backend] listening for your command...")
                audio = utterance_listener.record_utterance(max_seconds=timeout)
                text = transcribe(audio)
                if not text:
                    print("[backend] heard nothing usable — back to wake-word listening")
                    break

                print(f"[backend] you said: {text}")
                pending_interruption = _run_turn(text)
                if pending_interruption is None:
                    emit({"type": "state", "value": "listening"})
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        raise
    finally:
        wake.close()
        if barge_listener_ref[0] is not None:
            barge_listener_ref[0].stop()
        audio_device.shutdown()


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()

    # route memory-formation events to the UI so fact verification is
    # visible live in the HUD rather than happening invisibly
    def _memory_event(ev):
        emit({"type": "memory_event", "data": ev})
        emit({"type": "memory_stats", "data": memory_store.stats()})
    brain_llm.on_memory_event = _memory_event

    # Bind the port BEFORE starting the voice loop. The old order started the
    # voice loop first, so on a port conflict the process would claim the
    # microphone and then immediately exit — briefly holding the audio device
    # for no reason and making the next start flakier.
    try:
        server = await websockets.serve(_handler, HOST, PORT)
    except OSError as e:
        if getattr(e, "errno", None) == 48:
            print(f"\n[backend] PORT {PORT} IS ALREADY IN USE.")
            print("[backend] Another SEVRIN backend is still running — probably")
            print("[backend] one started from a terminal earlier. Kill it with:")
            print(f"[backend]     lsof -ti:{PORT} | xargs kill -9")
            print("[backend] (the desktop app now does this automatically on start)\n")
            return
        raise

    print(f"[backend] websocket server on ws://{HOST}:{PORT}")

    threading.Thread(target=_voice_loop, daemon=True).start()

    async with server:
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(_main())
