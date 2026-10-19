"""
Backend service for the desktop UI — voice loop, memory, scheduling,
proactive speech, and light UI control, all over one WebSocket.

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
  {"type": "transcript_delta", "text": "..."}                      (one per packet, emitted when
                                                                     ITS AUDIO STARTS PLAYING, not
                                                                     when Claude generated it —
                                                                     keeps on-screen text and voice
                                                                     in lockstep on long replies)
  {"type": "transcript_end"}                                       (assistant reply complete —
                                                                     fires after playback finishes,
                                                                     not after generation finishes)
  {"type": "amplitude", "value": 0.0-1.0}                          (while speaking — see note below)
  {"type": "ui_panel", "panel": NAME, "action": "show"|"hide",
   "source": "assistant"|"auto"}                                   (reveal/hide a HUD panel — see
                                                                     _handle_ui_directives and the
                                                                     auto-triggers on scheduling/
                                                                     memory events)
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
import re
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
from backend import proactive as proactive_mod
from backend import scheduler as scheduler_mod
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


_current_state = "idle"


def current_state():
    return _current_state


def emit(event: dict):
    """Thread-safe: call from any thread, including the blocking voice
    loop thread, to broadcast an event to all connected UI clients."""
    # Keep a local copy of state — the proactive queue's structural rules
    # depend on knowing exactly what's happening right now.
    global _current_state
    if event.get("type") == "state":
        _current_state = event.get("value", _current_state)
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
                text = msg["text"].strip()
                if _consume_if_ack(text):
                    continue
                threading.Thread(target=_run_turn_chained, args=(text,), daemon=True).start()
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


# ---------------------------------------------------------------------------
# Proactive speech — SEVRIN talking without being asked.
# ---------------------------------------------------------------------------

proactive_queue = None
scheduler = None

# --- attention-check acknowledgment -----------------------------------
# When SEVRIN speaks up unprompted about something that actually needs a
# response (a timer, a reminder), she checks Krish is there first rather
# than launching straight into it — the way a person would. Any input that
# arrives while we're waiting counts as the acknowledgment and is consumed
# silently rather than sent through the normal reply pipeline: "yeah?"
# doesn't need or deserve a full generated answer, it just means "go on".
_pending_ack = {"active": False}
_ack_event = threading.Event()
PROACTIVE_ACK_TIMEOUT = 12.0   # if he doesn't respond, deliver anyway rather than hang


def _consume_if_ack(text: str) -> bool:
    """Returns True if `text` was swallowed as an acknowledgment (caller
    should NOT process it as a normal command). Works for both voice and
    typed input — see the call sites in the voice loop and the websocket
    handler."""
    if _pending_ack["active"]:
        _pending_ack["active"] = False
        _ack_event.set()
        print(f"[proactive] check-in acknowledged: {text!r}")
        return True
    return False


def _speak_line(text: str, kind: str = None):
    """Shared plumbing for anything SEVRIN says unprompted: emit the events,
    speak it, log it, record it in history."""
    emit({"type": "state", "value": "speaking"})
    if kind:
        emit({"type": "proactive", "text": text, "kind": kind})
    emit({"type": "transcript_start", "role": "assistant"})
    emit({"type": "transcript_delta", "text": text})
    emit({"type": "transcript_end"})
    print(f"[sevrin, unprompted] {text}")
    try:
        _speak_with_animation(text)
    finally:
        brain_llm.record_proactive(text)


def _speak_proactively(item):
    """Locked entry point — see _turn_lock (defined near _run_turn below)."""
    _turn_lock.acquire()
    try:
        _speak_proactively_locked(item)
    finally:
        _turn_lock.release()


def _speak_proactively_locked(item):
    """Deliver one proactive item.

    For anything critical (a timer/reminder actually firing), this is now a
    two-step delivery: get his attention, wait for a real acknowledgment,
    THEN say what it's actually about — rather than blurting the content at
    him the instant he might not even be listening. Lower-urgency items
    (background info he didn't ask to be interrupted for) skip the check-in;
    demanding his attention for something that doesn't need it would be
    worse than the problem it solves.

    Only ever called with _turn_lock already held — see _speak_proactively.
    """
    needs_ack = item.urgency == proactive_mod.CRITICAL

    if needs_ack:
        attention_line = brain_llm.phrase_attention_check(item)
        _speak_line(attention_line, kind="check_in")

        _ack_event.clear()
        _pending_ack["active"] = True
        emit({"type": "state", "value": "listening"})
        got_ack = _ack_event.wait(timeout=PROACTIVE_ACK_TIMEOUT)
        _pending_ack["active"] = False
        if not got_ack:
            print("[proactive] no response within timeout; delivering anyway")

    line = brain_llm.phrase_proactive(item)
    _speak_line(line, kind=item.kind)
    emit({"type": "state", "value": "idle"})


def _confirm_reminder(when_epoch, subject, recurrence=None):
    """Acknowledge a scheduled reminder briefly, in character.

    Says the time back deliberately: a misheard time is the failure mode that
    matters here, and hearing it repeated is how he'd catch it."""
    import time as _time
    from datetime import datetime as _dt

    delta = when_epoch - _time.time()
    target = _dt.fromtimestamp(when_epoch)

    if recurrence:
        rtype = recurrence.get("type")
        clock = target.strftime("%H:%M")
        if rtype == "weekdays":
            when = f"every weekday at {clock}"
        elif rtype == "weekly":
            when = f"every {target.strftime('%A')} at {clock}"
        else:
            when = f"every day at {clock}"
    elif delta < 90:
        when = f"in {int(delta)} seconds"
    elif delta < 3600:
        when = f"in {int(delta // 60)} minutes"
    elif target.date() == _dt.now().date():
        when = f"at {target.strftime('%H:%M')}"
    else:
        when = f"{target.strftime('%A')} at {target.strftime('%H:%M')}"

    # A timer with no subject reads oddly as "I'll tell you in 5 minutes" —
    # he knows what he asked for; just confirm the duration.
    if subject:
        line = f"Noted — {subject}, {when}."
    elif delta < 3600 and not recurrence:
        line = f"Timer set, {when}."
    else:
        line = f"Noted. I'll tell you {when}."
    emit({"type": "transcript_start", "role": "assistant"})
    emit({"type": "transcript_delta", "text": line})
    emit({"type": "transcript_end"})
    print(f"[sevrin] {line}")
    _speak_with_animation(line)
    brain_llm.record_proactive(line)


def _setup_proactive():
    global proactive_queue
    proactive_queue = proactive_mod.ProactiveQueue(
        state_fn=current_state,
        speak_fn=_speak_proactively,
        judge_fn=brain_llm.judge_proactive,
    )
    proactive_queue.start()

    global scheduler
    scheduler = scheduler_mod.Scheduler(proactive_queue)
    scheduler.start()

    # keep the UI's schedule panel in step with reality
    def _push_schedules():
        import time as _t
        while True:
            try:
                rows = scheduler.upcoming(limit=4)
                emit({"type": "schedules", "data": [
                    {"subject": r["subject"] or "(no subject)",
                     "due": r["next_due"],
                     "recurring": bool(r.get("recurrence"))}
                    for r in rows
                ]})
            except Exception:
                pass
            _t.sleep(5)

    threading.Thread(target=_push_schedules, daemon=True).start()


# Matches a directive Claude embeds directly in her reply text when she
# decides a panel should be shown, e.g. "<<panel:schedule>>Here's what's
# coming up." This is deliberately NOT full Anthropic tool-use — a proper
# tool-call round trip means a second API request before any text streams
# back, which costs real latency on every single reply, not just the ones
# that touch a panel. A packet-level tag costs nothing extra and is stripped
# before the text is ever spoken or shown, so it's invisible either way.
_UI_DIRECTIVE = re.compile(r"<<panel:(\w+)>>")
_VALID_PANELS = {"schedule", "memory", "system", "alerts", "telemetry"}


def _handle_ui_directives(packet: str) -> str:
    """Strips any panel directive out of a packet and emits the
    corresponding UI event. Returns the cleaned packet — this MUST run
    before the packet reaches TTS or the transcript, or a directive would
    get spoken aloud."""
    def _emit_and_drop(m):
        panel = m.group(1).lower()
        if panel in _VALID_PANELS:
            emit({"type": "ui_panel", "panel": panel, "action": "show", "source": "assistant"})
        return ""
    return _UI_DIRECTIVE.sub(_emit_and_drop, packet).strip()


def _run_turn_chained(user_text: str):
    """Entry point for typed input. If SEVRIN gets interrupted while
    answering, the interruption becomes the next turn — same behaviour as
    the voice path, so typing and speaking don't diverge."""
    interruption = _run_turn(user_text)
    while interruption is not None:
        context = {"spoken": interruption["spoken"], "unspoken": interruption["unspoken"]}
        interruption = _run_turn(interruption["heard"], interruption=context)


# One reply speaking at a time, globally. Without this, the voice loop and
# the typed-input path (which spawns a fresh thread per message) had no
# mutual exclusion — if the mic picked up speech (Krish, a friend,
# background noise) while a typed message was also being answered, TWO
# independent _run_turn calls could run concurrently, each spawning its own
# ffplay process. That's genuine overlapping audio, not a display glitch.
# _speak_proactively takes the same lock, so an unprompted remark can't
# collide with a normal reply either.
_turn_lock = threading.Lock()


def _run_turn(user_text: str, interruption: dict = None):
    """Locked entry point — see _turn_lock above."""
    _turn_lock.acquire()
    try:
        return _run_turn_locked(user_text, interruption)
    finally:
        _turn_lock.release()


def _run_turn_locked(user_text: str, interruption: dict = None):
    """Runs one exchange, pipelined so Claude keeps generating the next
    packet while the current one plays.

    Interruptible: while SEVRIN speaks, a BargeInListener watches the mic.
    If Krish talks over him, playback is killed mid-sentence and we record
    exactly what got said versus what didn't — that split is what lets
    SEVRIN judge whether the unfinished part still mattered (see
    brain/llm.py's _interruption_context).

    Returns a dict describing an interruption if one occurred, else None.
    Only ever called with _turn_lock already held — see _run_turn above."""
    if proactive_queue is not None:
        proactive_queue.note_user_activity()
    emit({"type": "transcript", "role": "user", "text": user_text})

    # Reminder requests are handled here rather than by the model — the time
    # has to be parsed exactly, and a scheduling error is worse than a clumsy
    # phrasing. SEVRIN still confirms it in his own words.
    schedule_note = None
    if scheduler is not None:
        scheduled = scheduler.schedule_from_text(user_text)
        if scheduled:
            when, subject, recurrence = scheduled
            emit({"type": "state", "value": "thinking"})
            _confirm_reminder(when, subject, recurrence)
            # He just scheduled something — show him what's coming up without
            # having to ask. This is deterministic (a schedule genuinely
            # exists now), unlike the assistant-triggered <<panel:x>> tag.
            emit({"type": "ui_panel", "panel": "schedule", "action": "show", "source": "auto"})
            emit({"type": "state", "value": "idle"})
            return None

        # He asked for a reminder but the time couldn't be parsed. Without
        # telling SEVRIN this, he has no idea a reminder was even attempted
        # and will happily claim he "made a note of it" — which happened,
        # and is exactly the fabrication the personality is meant to prevent.
        if scheduler_mod.timeparse.looks_like_schedule(user_text):
            schedule_note = (
                "\n\n--- REMINDER NOT SET ---\n"
                "He appears to be asking for a reminder, timer or alarm, but the "
                "time could not be parsed, so NOTHING has been scheduled. Do not "
                "say you've noted it, set it, or will remind him — none of that is "
                "true. Do NOT say you're unable to set timers or reminders either; "
                "you can, this phrasing just didn't parse. Ask him to restate the "
                "time, plainly, in one short sentence."
            )

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
            # Emit the on-screen text HERE, when this packet's audio is
            # actually about to start — not when Claude generated it. Text
            # generation runs far ahead of speech (Claude can produce several
            # sentences in the time it takes to voice one), so emitting from
            # the generation loop made later sentences appear on screen while
            # an earlier one was still being spoken. Tying display to
            # playback keeps text and voice in lockstep.
            emit({"type": "transcript_delta", "text": prepared})
            _speak_with_animation(prepared)
            if not interrupted.is_set():
                with spoken_lock:
                    spoken_packets.append(prepared)

    fetch_thread = threading.Thread(target=fetch_worker, daemon=True)
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    fetch_thread.start()
    playback_thread.start()

    emit({"type": "transcript_start", "role": "assistant"})
    for packet in ask_streaming_packets(user_text, interruption=interruption,
                                        extra_context=schedule_note):
        packet = _handle_ui_directives(packet)
        all_packets.append(packet)
        if interrupted.is_set():
            # stop pulling more from Claude — he's been cut off
            break
        print(packet, end=" ", flush=True)
        text_queue.put(packet)
    print()

    text_queue.put(None)
    fetch_thread.join(timeout=5)
    playback_thread.join(timeout=5)
    barge.stop()
    # The reply is only really "done" once the audio for it has finished
    # playing, not once Claude finished generating text — so signal the end
    # of the transcript here, after playback_thread has joined.
    emit({"type": "transcript_end"})

    if not interrupted.is_set():
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
                if _consume_if_ack(text):
                    emit({"type": "state", "value": "listening"})
                    continue
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
        if proactive_queue is not None:
            proactive_queue.stop()
        if scheduler is not None:
            scheduler.stop()
        audio_device.shutdown()
        try:
            from voice import stt_whispercpp
            stt_whispercpp.stop_server()
        except Exception:
            pass   # not the active backend, or never started — nothing to clean up


async def _main():
    global _loop
    _loop = asyncio.get_running_loop()

    # route memory-formation events to the UI so fact verification is
    # visible live in the HUD rather than happening invisibly
    def _memory_event(ev):
        emit({"type": "memory_event", "data": ev})
        emit({"type": "memory_stats", "data": memory_store.stats()})
        # Only surface the panel for outcomes worth seeing — a rejected or
        # duplicate candidate isn't something to interrupt the screen for.
        if ev.get("outcome") in ("stored", "updated"):
            emit({"type": "ui_panel", "panel": "memory", "action": "show", "source": "auto"})
    brain_llm.on_memory_event = _memory_event

    _setup_proactive()

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
