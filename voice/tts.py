"""Text to speech: macOS `say`, or ElevenLabs streamed into ffplay.

With ElevenLabs the same mp3 bytes also go to a PyAV decoder thread that
measures real loudness for the orb, so playback never waits on decoding.
stop_playback() cuts the voice off immediately when Krish interrupts.
"""
import io
import queue
import subprocess
import threading
import time

import numpy as np
import requests

import config

_session = requests.Session()
_stop_flag = threading.Event()
_current_proc = None
_proc_lock = threading.Lock()


def stop_playback():
    """Cut off whatever is playing, right now."""
    _stop_flag.set()
    with _proc_lock:
        proc = _current_proc
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass


def clear_stop():
    """Call before a new reply so the last interrupt doesn't carry over."""
    _stop_flag.clear()


def play(text: str, on_amplitude=None):
    if not text or not text.strip():
        return
    if _stop_flag.is_set():
        return      # interrupted while this packet was waiting
    if config.TTS_BACKEND == "elevenlabs":
        _speak_elevenlabs(text, on_amplitude)
    else:
        _speak_local(text)


def _speak_local(text: str):
    cmd = ["say"]
    if config.LOCAL_VOICE_NAME:
        cmd += ["-v", config.LOCAL_VOICE_NAME]
    subprocess.run(cmd + [text])


def _speak_elevenlabs(text: str, on_amplitude=None):
    global _current_proc
    response = _session.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream",
        params={"optimize_streaming_latency": 4, "output_format": "mp3_44100_128"},
        headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": config.ELEVENLABS_MODEL,
            # fairly stable: SEVRIN is even-toned; much higher goes flat and robotic
            "voice_settings": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.0},
        },
        stream=True,
    )
    if response.status_code != 200:
        try:
            detail = response.text[:200]
        except Exception:
            detail = ""
        raise RuntimeError(f"ElevenLabs HTTP {response.status_code}: {detail}")

    # ffplay takes mp3 on stdin fine; it wouldn't take raw PCM on this machine
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
    )
    with _proc_lock:
        _current_proc = proc

    chunks = None
    decoder = None
    if on_amplitude is not None:
        chunks = queue.Queue()
        decoder = threading.Thread(target=_decode_amplitude, args=(chunks, on_amplitude), daemon=True)
        decoder.start()

    try:
        for chunk in response.iter_content(chunk_size=2048):
            if _stop_flag.is_set():
                break
            if not chunk:
                continue
            try:
                proc.stdin.write(chunk)
            except (BrokenPipeError, ValueError):
                break       # stop_playback() killed ffplay
            if chunks is not None:
                chunks.put(chunk)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        if chunks is not None:
            chunks.put(None)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        with _proc_lock:
            _current_proc = None
        if decoder is not None:
            decoder.join(timeout=2.0)
        if on_amplitude is not None:
            on_amplitude(0.0)


def _decode_amplitude(chunks: queue.Queue, on_amplitude):
    """Decode the mp3 with PyAV and report the RMS of each frame. If PyAV is
    missing the voice still plays, the orb just doesn't follow it."""
    try:
        import av
    except Exception as e:
        print(f"[tts] amplitude decoder unavailable ({e}); voice still plays, orb won't react to it")
        return

    buf = bytearray()
    done = False
    while not done:
        while True:
            try:
                item = chunks.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            buf.extend(item)
        if not buf:
            time.sleep(0.02)
            continue
        try:
            container = av.open(io.BytesIO(bytes(buf)))
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray().astype(np.float32)
                if arr.size == 0:
                    continue
                norm = arr / max(1.0, float(np.abs(arr).max()))
                on_amplitude(min(1.0, float(np.sqrt(np.mean(norm * norm))) * 1.8))
            container.close()
            break
        except Exception:
            time.sleep(0.03)    # not enough data to decode yet
