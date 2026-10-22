"""One PyAudio instance for the whole process.

Creating and terminating PyAudio instances while other input streams are open
wedges CoreAudio on macOS: the device starts returning all-zero buffers and
never recovers. So everything shares this one and it's only terminated at exit.
"""
import threading

import pyaudio

import config

_pa = None
_lock = threading.Lock()


def get_pa():
    global _pa
    with _lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
        return _pa


def open_input_stream(rate, frames_per_buffer):
    """Mono 16-bit input stream on the shared instance."""
    return get_pa().open(
        rate=rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )


def record(seconds: float, rate: int = config.SAMPLE_RATE, chunk: int = 1024) -> bytes:
    """Blocking fixed-length recording, used by enrolment and the tools/ scripts."""
    stream = open_input_stream(rate, chunk)
    try:
        frames = [stream.read(chunk, exception_on_overflow=False)
                  for _ in range(int(rate / chunk * seconds))]
    finally:
        stream.close()
    return b"".join(frames)


def shutdown():
    """Only call at process exit."""
    global _pa
    with _lock:
        if _pa is not None:
            try:
                _pa.terminate()
            except Exception:
                pass
            _pa = None


def is_silent(raw: bytes) -> bool:
    """True for exact digital silence, which is what a wedged device returns.
    A quiet room still has some noise in it."""
    return raw is not None and len(raw) > 0 and not any(raw)
