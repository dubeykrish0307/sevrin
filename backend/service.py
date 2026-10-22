"""SEVRIN backend: runs the voice loop and talks to the desktop app over a
local WebSocket (ws://localhost:8765).

Backend -> app events:
  state             value: idle | listening | thinking | speaking
  transcript        a user turn
  transcript_start, transcript_delta, transcript_end
                    an assistant reply. Each delta is sent when its audio
                    starts playing, so the text on screen keeps pace with the voice.
  amplitude         0-1 while speaking, drives the orb
  ui_panel          show or hide a HUD panel
  proactive, interrupted, schedules, memory_event, memory_stats, error

App -> backend:
  text_input        typed text, handled the same way as speech
"""
import asyncio
import errno
import json
import os
import queue
import random
import re
import sys
import threading
import time
from datetime import datetime

import websockets

# so `python3 backend/service.py` can import the rest of the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backend import proactive, timeparse
from backend.scheduler import Scheduler
from brain import llm, prompts, store
from voice import audio_device, tts
from voice.barge_in import BargeInListener
from voice.listener import UtteranceListener
from voice.stt import get_model, transcribe
from voice.wake_word import WakeWordListener

HOST = "localhost"
PORT = 8765
ACK_TIMEOUT = 12.0      # wait this long for an answer to a check-in, then say it anyway

# Claude can start a reply with e.g. "<<panel:schedule>>" to open a HUD panel.
# A tag costs nothing, where tool use would add a round trip before any text streams.
_PANEL_TAG = re.compile(r"<<panel:(\w+)>>")
_PANELS = {"schedule", "memory", "system", "alerts", "telemetry"}

_loop = None            # asyncio loop, so other threads can broadcast
_clients = set()
_state = "idle"

_listener = None        # UtteranceListener shared by the voice loop and interruptions
_barge = None           # one BargeInListener for every turn; making one per turn wedged CoreAudio
_queue = None           # proactive.ProactiveQueue
_scheduler = None

# Only one thing speaks at a time. The voice loop, typed input (a thread per
# message) and proactive speech all take this lock.
_turn_lock = threading.Lock()

_awaiting_ack = threading.Event()
_ack_received = threading.Event()


# --- events to the app ----------------------------------------------------------

def emit(event: dict):
    """Broadcast an event to every connected window. Safe from any thread."""
    global _state
    if event.get("type") == "state":
        _state = event.get("value", _state)
    if _loop is not None:
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


def current_state() -> str:
    return _state


def _show_line(text: str):
    emit({"type": "transcript_start", "role": "assistant"})
    emit({"type": "transcript_delta", "text": text})
    emit({"type": "transcript_end"})


def _strip_panel_tags(packet: str) -> str:
    """Remove panel tags from a packet and open the panels they name. Must run
    before the packet is spoken or shown."""
    def open_panel(m):
        panel = m.group(1).lower()
        if panel in _PANELS:
            emit({"type": "ui_panel", "panel": panel, "action": "show", "source": "assistant"})
        return ""
    return _PANEL_TAG.sub(open_panel, packet).strip()


def _on_memory_event(ev):
    emit({"type": "memory_event", "data": ev})
    emit({"type": "memory_stats", "data": store.stats()})
    # only worth putting on screen when something was actually remembered
    if ev.get("outcome") in ("stored", "updated"):
        emit({"type": "ui_panel", "panel": "memory", "action": "show", "source": "auto"})


# --- speaking ---------------------------------------------------------------------

def _speak(text: str):
    """Play TTS and drive the orb. ElevenLabs gives real loudness per chunk;
    macOS `say` has no stream to measure, so the orb gets a rough envelope."""
    if config.TTS_BACKEND == "elevenlabs":
        tts.play(text, on_amplitude=lambda level: emit({"type": "amplitude", "value": round(level, 3)}))
        return

    stop = threading.Event()

    def wobble():
        end = time.time() + max(0.8, len(text) / 15.0)
        level = 0.4
        while time.time() < end and not stop.is_set():
            level += random.uniform(-0.25, 0.25)
            level = max(0.15, min(1.0, level))
            emit({"type": "amplitude", "value": round(level, 2)})
            time.sleep(0.06)

    animator = threading.Thread(target=wobble, daemon=True)
    animator.start()
    try:
        tts.play(text)
    finally:
        stop.set()
        animator.join(timeout=0.5)
        emit({"type": "amplitude", "value": 0.0})


def _say_unprompted(text: str, kind: str = None):
    emit({"type": "state", "value": "speaking"})
    if kind:
        emit({"type": "proactive", "text": text, "kind": kind})
    _show_line(text)
    print(f"[sevrin, unprompted] {text}")
    try:
        _speak(text)
    finally:
        llm.record_proactive(text)


def _confirm_reminder(when: float, subject: str, recurrence: dict = None):
    """Say the reminder back with its time. A misheard time is the failure that
    matters, and hearing it repeated is how he'd catch it."""
    delta = when - time.time()
    target = datetime.fromtimestamp(when)

    if recurrence:
        rtype = recurrence.get("type")
        if rtype == "weekdays":
            spoken_time = f"every weekday at {target:%H:%M}"
        elif rtype == "weekly":
            spoken_time = f"every {target:%A} at {target:%H:%M}"
        else:
            spoken_time = f"every day at {target:%H:%M}"
    elif delta < 90:
        spoken_time = f"in {int(delta)} seconds"
    elif delta < 3600:
        spoken_time = f"in {int(delta // 60)} minutes"
    elif target.date() == datetime.now().date():
        spoken_time = f"at {target:%H:%M}"
    else:
        spoken_time = f"{target:%A} at {target:%H:%M}"

    if subject:
        line = f"Noted — {subject}, {spoken_time}."
    elif delta < 3600 and not recurrence:
        line = f"Timer set, {spoken_time}."   # he knows what the timer is for
    else:
        line = f"Noted. I'll tell you {spoken_time}."

    _show_line(line)
    print(f"[sevrin] {line}")
    _speak(line)
    llm.record_proactive(line)


# --- proactive speech ----------------------------------------------------------

def _speak_proactively(item):
    """Deliver one queued item. For critical ones (a reminder going off), check
    he's there first and wait for any reply before saying what it's about."""
    with _turn_lock:
        if item.urgency == proactive.CRITICAL:
            _say_unprompted(llm.phrase_attention_check(item), kind="check_in")
            _ack_received.clear()
            _awaiting_ack.set()
            emit({"type": "state", "value": "listening"})
            got_ack = _ack_received.wait(timeout=ACK_TIMEOUT)
            _awaiting_ack.clear()
            if not got_ack:
                print("[proactive] no response within timeout; delivering anyway")

        _say_unprompted(llm.phrase_proactive(item), kind=item.kind)
        emit({"type": "state", "value": "idle"})


def _consume_if_ack(text: str) -> bool:
    """If we're waiting on a check-in, any input ("yeah?") is the answer and
    shouldn't get a reply of its own. Returns True if it was swallowed."""
    if not _awaiting_ack.is_set():
        return False
    _awaiting_ack.clear()
    _ack_received.set()
    print(f"[proactive] check-in acknowledged: {text!r}")
    return True


def _push_schedules():
    """Keep the app's schedule panel in sync."""
    while True:
        try:
            emit({"type": "schedules", "data": [
                {"subject": r["subject"] or "(no subject)",
                 "due": r["next_due"],
                 "recurring": bool(r.get("recurrence"))}
                for r in _scheduler.upcoming(limit=4)
            ]})
        except Exception:
            pass
        time.sleep(5)


def _start_proactive():
    global _queue, _scheduler
    _queue = proactive.ProactiveQueue(
        state_fn=current_state,
        speak_fn=_speak_proactively,
        judge_fn=llm.judge_proactive,
    )
    _queue.start()
    _scheduler = Scheduler(_queue)
    _scheduler.start()
    threading.Thread(target=_push_schedules, daemon=True).start()


# --- one exchange ---------------------------------------------------------------

def _run_turn(user_text: str, interruption: dict = None):
    with _turn_lock:
        return _reply(user_text, interruption)


def _follow_up(interruption: dict):
    """Whatever he said over a reply becomes the next turn, with the cut-off
    reply as context."""
    context = {"spoken": interruption["spoken"], "unspoken": interruption["unspoken"]}
    return _run_turn(interruption["heard"], interruption=context)


def _handle_typed(text: str):
    interruption = _run_turn(text)
    while interruption is not None:
        interruption = _follow_up(interruption)


def _reply(user_text: str, interruption: dict = None):
    """Answer one turn. Claude keeps generating while earlier packets play.

    Returns {"spoken", "unspoken", "heard"} if he talked over the reply,
    otherwise None. Call through _run_turn so the turn lock is held.
    """
    global _barge
    if _queue is not None:
        _queue.note_user_activity()
    emit({"type": "transcript", "role": "user", "text": user_text})

    # Reminders are parsed by rules, not the model, because the time has to be exact.
    extra_context = None
    if _scheduler is not None:
        scheduled = _scheduler.schedule_from_text(user_text)
        if scheduled:
            emit({"type": "state", "value": "thinking"})
            _confirm_reminder(*scheduled)
            emit({"type": "ui_panel", "panel": "schedule", "action": "show", "source": "auto"})
            emit({"type": "state", "value": "idle"})
            return None
        if timeparse.looks_like_schedule(user_text):
            # otherwise the model happily claims it set the reminder
            extra_context = prompts.REMINDER_NOT_SET

    emit({"type": "state", "value": "thinking"})
    print("[sevrin] ", end="", flush=True)
    tts.clear_stop()

    to_play = queue.Queue()
    generated, spoken = [], []
    spoken_lock = threading.Lock()
    interrupted = threading.Event()

    if _barge is None:
        _barge = BargeInListener()
    if _listener is not None:
        # barge-in can't measure the room while we're talking, so borrow the profile
        _barge.set_noise_profile(_listener.noise)

    def on_interrupt():
        interrupted.set()
        tts.stop_playback()
        emit({"type": "interrupted"})

    def play_packets():
        first = True
        while True:
            packet = to_play.get()
            if packet is None:
                break
            if interrupted.is_set():
                continue    # drain the queue without playing
            if first:
                first = False
                emit({"type": "state", "value": "speaking"})
                # start listening for interruptions only once audio is playing,
                # otherwise the tail of his own command counts as one
                _barge.start(on_interrupt)
            emit({"type": "transcript_delta", "text": packet})
            _speak(packet)
            if not interrupted.is_set():
                with spoken_lock:
                    spoken.append(packet)

    player = threading.Thread(target=play_packets, daemon=True)
    player.start()

    emit({"type": "transcript_start", "role": "assistant"})
    for packet in llm.ask_streaming_packets(user_text, interruption=interruption,
                                            extra_context=extra_context):
        packet = _strip_panel_tags(packet)
        generated.append(packet)
        if interrupted.is_set():
            break       # he cut us off, stop pulling from Claude
        print(packet, end=" ", flush=True)
        to_play.put(packet)
    print()

    to_play.put(None)
    player.join(timeout=5)
    _barge.stop()
    emit({"type": "transcript_end"})

    if not interrupted.is_set():
        emit({"type": "state", "value": "idle"})
        return None
    return _capture_interruption(generated, spoken, spoken_lock)


def _capture_interruption(generated, spoken, spoken_lock):
    """Work out what was said before the cut-off and what he said over it."""
    with spoken_lock:
        said = " ".join(spoken)
    unsaid = " ".join(p for p in generated if p not in spoken)
    print(f"\n[interrupt] cut off after: {said[:60]!r}...")

    clip = _barge.captured_audio
    _barge.stop()
    heard = ""
    try:
        if clip:
            heard = transcribe(clip)
    except Exception as e:
        print(f"[interrupt] couldn't transcribe interruption: {e}")

    # the barge-in clip is short and may only hold the start, so let him finish
    emit({"type": "state", "value": "listening"})
    try:
        more = _listener.record_utterance(max_seconds=config.FOLLOWUP_WINDOW_SECONDS)
        tail = transcribe(more) if more else ""
        if tail:
            heard = f"{heard} {tail}".strip() if heard else tail
    except Exception as e:
        print(f"[interrupt] couldn't capture the rest: {e}")

    if not heard:
        heard = "(stopped you, said nothing further)"   # a noise, but nothing intelligible
    print(f"[interrupt] he said: {heard}")
    return {"spoken": said, "unspoken": unsaid, "heard": heard}


# --- voice loop -----------------------------------------------------------------

def _conversation(listener: UtteranceListener):
    """Take turns after a wake until he stays quiet for the follow-up window."""
    first_turn = True
    interruption = None
    while True:
        if interruption is not None:
            interruption = _follow_up(interruption)
            continue

        timeout = config.MAX_UTTERANCE_SECONDS if first_turn else config.FOLLOWUP_WINDOW_SECONDS
        first_turn = False
        print("[backend] listening for your command...")
        text = transcribe(listener.record_utterance(max_seconds=timeout))
        if not text:
            print("[backend] heard nothing usable, back to wake-word listening")
            return

        print(f"[backend] you said: {text}")
        if _consume_if_ack(text):
            emit({"type": "state", "value": "listening"})
            continue
        interruption = _run_turn(text)
        if interruption is None:
            emit({"type": "state", "value": "listening"})


def _voice_loop():
    global _listener
    print("[backend] loading whisper model...")
    get_model()

    # Measure the room once the mic is open and share that profile everywhere.
    # Calibrating inside the wake listener ran before the mic was warm.
    listener = UtteranceListener()
    print("[backend] measuring room noise (stay quiet for a second)...")
    listener.calibrate_noise()
    _listener = listener

    wake = WakeWordListener()
    wake.set_noise_profile(listener.noise)
    wake.start()

    try:
        while True:
            emit({"type": "state", "value": "idle"})
            triggered_by = wake.listen_once()
            print(f"\n[backend] wake triggered by '{triggered_by}'")
            emit({"type": "state", "value": "listening"})
            _conversation(listener)
    except Exception as e:
        emit({"type": "error", "message": str(e)})
        raise
    finally:
        _shutdown()


def _shutdown():
    if _barge is not None:
        _barge.stop()
    if _queue is not None:
        _queue.stop()
    if _scheduler is not None:
        _scheduler.stop()
    audio_device.shutdown()
    try:
        from voice import stt_whispercpp
        stt_whispercpp.stop_server()
    except Exception:
        pass


# --- websocket server -----------------------------------------------------------

async def _handle_client(websocket):
    _clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "memory_stats", "data": store.stats()}))
    except Exception:
        pass
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("type") != "text_input":
                continue
            text = msg.get("text", "").strip()
            if text and not _consume_if_ack(text):
                threading.Thread(target=_handle_typed, args=(text,), daemon=True).start()
    finally:
        _clients.discard(websocket)


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()
    llm.on_memory_event = _on_memory_event
    _start_proactive()

    # Bind the port before touching the mic, so a second copy exits without
    # grabbing the audio device first.
    try:
        server = await websockets.serve(_handle_client, HOST, PORT)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(f"\n[backend] port {PORT} is already in use, probably by another backend.")
            print(f"[backend] stop it with: lsof -ti:{PORT} | xargs kill -9")
            print("[backend] (the desktop app does this for you on start)\n")
            return
        raise

    print(f"[backend] websocket server on ws://{HOST}:{PORT}")
    threading.Thread(target=_voice_loop, daemon=True).start()
    async with server:
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(_main())
