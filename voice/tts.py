"""
TTS with real amplitude — built on the audio path that actually works on
this machine.

Earlier the raw-PCM-to-ffplay path died with "Broken pipe" (this ffplay
build won't take the raw PCM flags), so real amplitude never worked. This
version instead streams mp3 (which ffplay plays fine here) AND decodes that
same mp3 to PCM in-process with PyAV to compute real loudness for the orb.
Playback is reliable; amplitude is real; nothing depends on ffplay's raw-PCM
support.

If mp3 playback itself ever fails, errors print loudly to the backend
terminal instead of being swallowed.
"""
import subprocess
import threading

import numpy as np
import requests

import config

_session = requests.Session()

# --- interruption support ---------------------------------------------
# When Krish speaks over SEVRIN, playback has to stop mid-sentence. We keep
# a handle on the currently-playing process and a stop flag; stop_playback()
# kills the audio immediately rather than waiting for the sentence to end.
_stop_flag = threading.Event()
_current_proc = None
_proc_lock = threading.Lock()


def stop_playback():
    """Cuts off whatever SEVRIN is saying, right now."""
    _stop_flag.set()
    with _proc_lock:
        proc = _current_proc
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass


def clear_stop():
    """Called before a new reply so the previous interrupt doesn't linger."""
    _stop_flag.clear()


def was_stopped() -> bool:
    return _stop_flag.is_set()


def prepare(text: str):
    return text


def play(text: str, on_amplitude=None):
    if not text or not text.strip():
        return
    if _stop_flag.is_set():
        return   # an interrupt landed while this packet was queued — drop it
    if config.TTS_BACKEND == "elevenlabs":
        _speak_elevenlabs(text, on_amplitude)
    else:
        _speak_local(text)


def _speak_local(text: str):
    cmd = ["say"]
    if config.LOCAL_VOICE_NAME:
        cmd += ["-v", config.LOCAL_VOICE_NAME]
    cmd.append(text)
    subprocess.run(cmd)


def _speak_elevenlabs(text: str, on_amplitude=None):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}/stream"
    params = {"optimize_streaming_latency": 4, "output_format": "mp3_44100_128"}
    headers = {"xi-api-key": config.ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": config.ELEVENLABS_MODEL,
        # stability 0.55: SEVRIN is even-toned and controlled, not expressive.
        # Lower would add emotional swing that fights the character; much
        # higher goes flat and robotic.
        "voice_settings": {"stability": 0.55, "similarity_boost": 0.8, "style": 0.0},
    }

    response = _session.post(url, params=params, json=payload, headers=headers, stream=True)
    if response.status_code != 200:
        detail = ""
        try:
            detail = response.text[:200]
        except Exception:
            pass
        raise RuntimeError(f"ElevenLabs HTTP {response.status_code}: {detail}")

    # ffplay plays the mp3 (works reliably here). We also tee the same bytes
    # to a background decoder thread that turns mp3 -> PCM and emits real
    # amplitude, so playback never waits on decoding.
    proc = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
        stdin=subprocess.PIPE,
    )
    global _current_proc
    with _proc_lock:
        _current_proc = proc

    amp_queue = None
    decoder_thread = None
    if on_amplitude is not None:
        amp_queue = []
        decoder_thread = threading.Thread(
            target=_decode_amplitude, args=(amp_queue, on_amplitude), daemon=True
        )
        decoder_thread.start()

    try:
        for chunk in response.iter_content(chunk_size=2048):
            if _stop_flag.is_set():
                break        # interrupted — stop feeding audio immediately
            if not chunk:
                continue
            try:
                proc.stdin.write(chunk)
            except (BrokenPipeError, ValueError):
                break        # process was killed by stop_playback()
            if amp_queue is not None:
                amp_queue.append(chunk)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        if amp_queue is not None:
            amp_queue.append(None)  # sentinel: no more audio
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        with _proc_lock:
            _current_proc = None
        if decoder_thread is not None:
            decoder_thread.join(timeout=2.0)
        if on_amplitude is not None:
            on_amplitude(0.0)


def _decode_amplitude(byte_chunks, on_amplitude):
    """Decodes the mp3 byte stream to PCM with PyAV and emits real RMS
    amplitude. Runs in its own thread so playback is never blocked. If PyAV
    isn't importable for some reason, we just skip amplitude (voice still
    plays) rather than crash."""
    try:
        import av
    except Exception as e:
        print(f"[tts] amplitude decoder unavailable ({e}); voice still plays, orb won't react to it")
        return

    import io
    import time

    # Wait for enough bytes to start, then decode incrementally. We buffer
    # the growing stream and let PyAV pull frames as they become available.
    buf = bytearray()
    done = False
    while not done:
        # drain whatever's queued
        while byte_chunks:
            item = byte_chunks.pop(0)
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
                # normalize (PyAV int16 planar -> scale), compute RMS
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                peak = max(1.0, float(np.abs(arr).max()))
                norm = arr / peak
                rms = float(np.sqrt(np.mean(norm * norm)))
                on_amplitude(min(1.0, rms * 1.8))
            container.close()
            break  # decoded the full buffer we had
        except Exception:
            # not enough data yet to open/decode — wait for more
            time.sleep(0.03)
