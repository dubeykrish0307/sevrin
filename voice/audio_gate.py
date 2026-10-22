"""Cheap physical checks that reject noise before anything treats it as speech.

webrtcvad was built for phone calls and calls fans and air conditioning
"speech" far too readily. So audio also has to pass:

  1. loudness: far enough above the room's own measured noise floor
  2. speech-band ratio: voice energy sits in 300-3400 Hz, fans and rumble
     are mostly below that
  3. spectral flatness: noise is flat, voiced speech has harmonic peaks

All plain numpy on a short clip. Thresholds live in config.py.
"""
import numpy as np

import config
from voice import audio_device

NOISE_FLOOR_MARGIN_DB = config.GATE_NOISE_MARGIN_DB
MIN_SPEECH_BAND_RATIO = config.GATE_MIN_BAND_RATIO
MAX_SPECTRAL_FLATNESS = config.GATE_MAX_FLATNESS
ABSOLUTE_FLOOR_RMS = config.GATE_ABSOLUTE_FLOOR_RMS

# Assume a quiet room until calibrated. A floor that's too high quietly
# rejects real speech, so err low.
DEFAULT_FLOOR_RMS = 0.0015
_CHUNK = 480


def _to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)) + 1e-9)


class NoiseProfile:
    """The room's ambient level, so the loudness check adapts to the room."""

    def __init__(self):
        self.floor_rms = DEFAULT_FLOOR_RMS

    def calibrate(self, seconds: float = 1.0):
        """Listen to the (hopefully quiet) room and take its median level."""
        try:
            stream = audio_device.open_input_stream(config.SAMPLE_RATE, _CHUNK)
        except Exception as e:
            print(f"[audio_gate] calibration skipped ({e}); using default floor")
            return

        levels = []
        try:
            # CoreAudio hands back zeroed buffers for a moment after a stream
            # opens, so throw away the first ~300 ms
            for _ in range(int(config.SAMPLE_RATE * 0.3 / _CHUNK)):
                stream.read(_CHUNK, exception_on_overflow=False)
            for _ in range(int(config.SAMPLE_RATE * seconds / _CHUNK)):
                levels.append(rms(_to_float(stream.read(_CHUNK, exception_on_overflow=False))))
        finally:
            try:
                stream.close()
            except Exception:
                pass
        if not levels:
            return

        measured = float(np.median(levels))   # median, so a cough doesn't skew it
        if measured < 1e-4:
            # the mic wasn't ready and we measured silence
            self.floor_rms = DEFAULT_FLOOR_RMS
            print(f"[audio_gate] calibration read silence "
                  f"({20 * np.log10(max(measured, 1e-9)):.0f} dBFS), mic wasn't ready; "
                  f"assuming a quiet room (floor {self.floor_rms})")
            return
        self.floor_rms = measured
        print(f"[audio_gate] room noise floor: {20 * np.log10(self.floor_rms):.1f} dBFS")


def _spectrum(samples: np.ndarray, sample_rate: int):
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    return power, freqs


def _band_ratio(power, freqs) -> float:
    """Energy in 300-3400 Hz as a share of everything above 150 Hz.

    The cut at 150 Hz matters. Talking close to the mic adds a big bass boost
    (proximity effect), and with all energy in the denominator the same voice
    dropped from ~0.5 at arm's length to ~0.27 up close.
    """
    usable = float(power[freqs >= 150].sum()) + 1e-12
    band = float(power[(freqs >= 300) & (freqs <= 3400)].sum())
    return band / usable


def _flatness(power) -> float:
    """Geometric over arithmetic mean: ~1 for flat noise, low for voiced speech."""
    power = power + 1e-12
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def looks_like_speech(pcm_bytes: bytes, profile: NoiseProfile, sample_rate: int = None):
    """Returns (passed, detail). Only audio that passes every check counts as a voice."""
    sample_rate = sample_rate or config.SAMPLE_RATE
    samples = _to_float(pcm_bytes)
    if samples.size < 160:
        return False, {"reason": "too short"}

    level = rms(samples)
    power, freqs = _spectrum(samples, sample_rate)
    ratio = _band_ratio(power, freqs)
    flat = _flatness(power)

    detail = {
        "rms": round(level, 5),
        "floor": round(profile.floor_rms, 5),
        "db_above_floor": round(20 * np.log10(level / profile.floor_rms), 1),
        "speech_band_ratio": round(ratio, 3),
        "spectral_flatness": round(flat, 3),
    }

    if level < ABSOLUTE_FLOOR_RMS:
        detail["reason"] = "below absolute silence floor"
        return False, detail

    # Quieter than the stored floor means the floor is stale (the room got
    # quieter). Pull it down and let this one through rather than blocking speech.
    if detail["db_above_floor"] < 0:
        profile.floor_rms = max(level * 0.7, 1e-5)
        detail["reason"] = "ok (noise floor was stale, re-baselined)"
        return True, detail

    if detail["db_above_floor"] < NOISE_FLOOR_MARGIN_DB:
        detail["reason"] = "not loud enough above room noise"
        return False, detail
    if ratio < MIN_SPEECH_BAND_RATIO:
        detail["reason"] = "energy outside speech band (rumble/fan)"
        return False, detail
    if flat > MAX_SPECTRAL_FLATNESS:
        detail["reason"] = "spectrally flat (noise-like, not voiced)"
        return False, detail

    detail["reason"] = "ok"
    return True, detail
