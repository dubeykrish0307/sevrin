"""whisper.cpp backend, through its bundled HTTP server.

The server loads the model once and keeps it in memory. The one-shot CLI
reloads it from disk on every call, which for large-v3-turbo (~1.6 GB) costs
more than the transcription itself.

Setup is in the README. tools/bench_stt.py checks the server works and
compares it with faster-whisper on your own voice.
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


_server = None
_ready = threading.Event()
_port = None


def _find_binary(root: str, name: str):
    # the build output location has moved between whisper.cpp versions
    override = config.WHISPER_CPP_SERVER_BINARY
    if override and os.path.isfile(override) and os.access(override, os.X_OK):
        return override
    for candidate in (f"build/bin/{name}", name, f"build/{name}"):
        path = os.path.join(root, candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _find_model(root: str, name: str):
    override = config.WHISPER_CPP_MODEL_PATH
    if override and os.path.isfile(override):
        return override
    path = os.path.join(root, "models", f"ggml-{name}.bin")
    return path if os.path.isfile(path) else None


def _wav_bytes(pcm_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _post(wav: bytes, timeout: float, **data):
    return requests.post(f"http://127.0.0.1:{_port}/inference",
                         files={"file": ("audio.wav", wav, "audio/wav")},
                         data=data or None, timeout=timeout)


def start_server():
    """Start whisper-server in the background and block until the model is loaded."""
    global _server, _port

    root = config.WHISPER_CPP_DIR
    if not root or not os.path.isdir(root):
        raise WhisperCppUnavailable("WHISPER_CPP_DIR not set or doesn't exist, set it in .env")

    binary = _find_binary(root, "whisper-server")
    if binary is None:
        raise WhisperCppUnavailable(
            f"couldn't find whisper-server under {root}. Run `make -j` there, "
            f"or set WHISPER_CPP_SERVER_BINARY in .env to the exact path.")

    model_name = config.WHISPER_CPP_MODEL
    model = _find_model(root, model_name)
    if model is None:
        raise WhisperCppUnavailable(
            f"model 'ggml-{model_name}.bin' not found under {root}/models. "
            f"Run: ./models/download-ggml-model.sh {model_name}")

    _port = config.WHISPER_CPP_PORT
    print(f"[stt] starting whisper-server ({model_name}, port {_port})...")
    _server = subprocess.Popen(
        [binary, "--host", "127.0.0.1", "--port", str(_port), "-m", model],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    atexit.register(stop_server)

    # No health endpoint that works across versions, so poll with a short
    # silent clip until a real transcription succeeds.
    silence = _wav_bytes(b"\x00\x00" * int(config.SAMPLE_RATE * 0.3))
    deadline = time.time() + 60
    while time.time() < deadline:
        if _server.poll() is not None:
            raise WhisperCppUnavailable(
                f"whisper-server exited during startup (code {_server.returncode}), "
                f"check the binary and model are valid for your build")
        try:
            if _post(silence, timeout=5).status_code == 200:
                _ready.set()
                print("[stt] whisper-server ready")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    raise WhisperCppUnavailable("whisper-server didn't become ready within 60s")


def stop_server():
    global _server
    if _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server.kill()
    _server = None


def transcribe(pcm_bytes: bytes) -> str:
    """Same contract as stt.transcribe(). Raises WhisperCppUnavailable on any
    failure so the caller can fall back to faster-whisper."""
    if not _ready.is_set():
        raise WhisperCppUnavailable("whisper-server not ready (call start_server() first)")
    if not pcm_bytes:
        return ""

    try:
        r = _post(_wav_bytes(pcm_bytes), timeout=15, response_format="json")
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
