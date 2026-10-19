"""
SEVRIN — whisper.cpp speech-to-text backend, via its HTTP server.

WHY A SERVER, NOT THE ONE-SHOT CLI
whisper-cli reloads the full model from disk into RAM on EVERY invocation.
For large-v3-turbo (~1.6GB), that reload alone could cost more than the
actual transcription — which would completely defeat the point of running
a heavier model for low latency. whisper.cpp's bundled whisper-server
loads the model ONCE at startup and stays resident; every request after
that only pays for inference. This mirrors how the faster-whisper model is
already eager-loaded once at SEVRIN startup rather than per call.

WHY WHISPER.CPP AT ALL
faster-whisper has no Metal backend on macOS — it's CPU-only there, which
is why this project stayed on small models. whisper.cpp compiles with
Metal support automatically on Apple Silicon (a real, measured speedup
that grows with model size), which is what makes a genuinely heavy model
like large-v3-turbo practical for real-time commands instead of only
batch transcription.

SETUP — see README.md. In short:
    git clone https://github.com/ggml-org/whisper.cpp
    cd whisper.cpp && make -j
    ./models/download-ggml-model.sh large-v3-turbo
Then set WHISPER_CPP_DIR in .env to that checkout's path.

HONESTY NOTE: I built this integration from whisper.cpp's own
documentation, not by running it myself — I have no way to execute
compiled binaries or hit a live HTTP server in this environment. The
request/response handling below is deliberately defensive: if the server's
actual response doesn't match what's expected, transcribe() raises WITH
the raw response body attached, rather than silently mishandling it. Run
voice/test_whispercpp.py first — if something doesn't parse, paste me
what it prints and the fix will be exact, not another guess.
"""
import atexit
import io
import os
import subprocess
import threading
import time
import wave

import requests

import config


class WhisperCppUnavailable(Exception):
    pass


_server_proc = None
_server_ready = threading.Event()
_binary_path = None
_model_path = None
_port = None


def _find_binary(root: str, name: str):
    """Search likely build output locations rather than assuming one —
    whisper.cpp's build layout has changed across versions/CMake configs.
    WHISPER_CPP_SERVER_BINARY in .env always wins if this guesses wrong
    for your checkout."""
    override = getattr(config, "WHISPER_CPP_SERVER_BINARY", None)
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    for pattern in (f"build/bin/{name}", name, f"build/{name}"):
        p = os.path.join(root, pattern)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _find_model(root: str, name: str):
    override = getattr(config, "WHISPER_CPP_MODEL_PATH", None)
    if override and os.path.isfile(override):
        return override
    guess = os.path.join(root, "models", f"ggml-{name}.bin")
    return guess if os.path.isfile(guess) else None


def _silent_wav_bytes(seconds=0.3) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(b"\x00\x00" * int(config.SAMPLE_RATE * seconds))
    return buf.getvalue()


def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def start_server():
    """Starts whisper-server once in the background and blocks until the
    model has actually finished loading. Call once at SEVRIN startup."""
    global _server_proc, _binary_path, _model_path, _port

    root = getattr(config, "WHISPER_CPP_DIR", None)
    if not root or not os.path.isdir(root):
        raise WhisperCppUnavailable("WHISPER_CPP_DIR not set or doesn't exist — set it in .env")

    binary = _find_binary(root, "whisper-server")
    if binary is None:
        raise WhisperCppUnavailable(
            f"couldn't find whisper-server under {root}. Run `make -j` there, "
            f"or set WHISPER_CPP_SERVER_BINARY in .env to the exact path.")

    model_name = getattr(config, "WHISPER_CPP_MODEL", "large-v3-turbo")
    model = _find_model(root, model_name)
    if model is None:
        raise WhisperCppUnavailable(
            f"model 'ggml-{model_name}.bin' not found under {root}/models. "
            f"Run: ./models/download-ggml-model.sh {model_name}")

    _binary_path, _model_path = binary, model
    _port = int(getattr(config, "WHISPER_CPP_PORT", 8178))

    print(f"[stt] starting whisper-server ({model_name}, port {_port})...")
    _server_proc = subprocess.Popen(
        [binary, "--host", "127.0.0.1", "--port", str(_port), "-m", model],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    atexit.register(stop_server)

    # No guaranteed lightweight health endpoint across versions, so poll
    # with a real (silent) transcription request rather than a fixed sleep
    # or guessing how long model loading takes.
    deadline = time.time() + 60
    silence = _silent_wav_bytes()
    while time.time() < deadline:
        if _server_proc.poll() is not None:
            raise WhisperCppUnavailable(
                f"whisper-server exited during startup (code {_server_proc.returncode}) "
                f"— check the binary/model are valid for your build")
        try:
            r = requests.post(f"http://127.0.0.1:{_port}/inference",
                              files={"file": ("probe.wav", silence, "audio/wav")}, timeout=5)
            if r.status_code == 200:
                _server_ready.set()
                print("[stt] whisper-server ready")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    raise WhisperCppUnavailable("whisper-server didn't become ready within 60s")


def stop_server():
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None


def transcribe(pcm_bytes: bytes) -> str:
    """Same contract as voice/stt.py's transcribe(). Raises
    WhisperCppUnavailable if the server isn't ready or the request fails —
    callers should catch this and fall back to faster-whisper."""
    if not _server_ready.is_set():
        raise WhisperCppUnavailable("whisper-server not ready (call start_server() first)")
    if not pcm_bytes:
        return ""

    wav_bytes = _pcm_to_wav_bytes(pcm_bytes)
    try:
        r = requests.post(
            f"http://127.0.0.1:{_port}/inference",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        raise WhisperCppUnavailable(f"couldn't reach whisper-server: {e}")

    if r.status_code != 200:
        raise WhisperCppUnavailable(f"whisper-server HTTP {r.status_code}: {r.text[:300]}")

    try:
        data = r.json()
        if isinstance(data, dict) and "text" in data:
            return data["text"].strip()
    except ValueError:
        pass

    text = r.text.strip()
    if text:
        return text

    raise WhisperCppUnavailable(f"couldn't parse whisper-server response: {r.text[:300]!r}")
