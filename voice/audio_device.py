"""
One shared PyAudio instance for the whole process.

Previously every component made its own: WakeWordListener, UtteranceListener,
and — worst of all — a fresh BargeInListener constructed on EVERY turn, each
calling pa.terminate() when done. Repeatedly creating and terminating PyAudio
instances while other input streams are live wedges CoreAudio on macOS: the
device starts returning all-zero buffers and never recovers. The symptom in
the logs was the gate reporting "band 0.0, flat 1.0, -123dB" — not quiet
audio, but literal digital silence — after which the wake loop span forever
on a dead stream.

Everything now shares this instance and nothing calls terminate() until the
process exits.
"""
import threading

import pyaudio

_pa = None
_lock = threading.Lock()


def get_pa():
    global _pa
    with _lock:
        if _pa is None:
            _pa = pyaudio.PyAudio()
        return _pa


def open_input_stream(rate, frames_per_buffer):
    """Open a mono 16-bit input stream on the shared instance."""
    return get_pa().open(
        rate=rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )


def shutdown():
    """Only at process exit."""
    global _pa
    with _lock:
        if _pa is not None:
            try:
                _pa.terminate()
            except Exception:
                pass
            _pa = None


def is_silent(raw: bytes) -> bool:
    """True if a frame is exactly digital silence — the signature of a wedged
    CoreAudio device, as opposed to a quiet room which still has dither."""
    return raw is not None and len(raw) > 0 and not any(raw)
