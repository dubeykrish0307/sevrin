"""
SEVRIN — acoustic gating.

Before any audio is treated as speech, it has to pass physical checks that
noise fails. This is what stops a fan, an air-conditioner, or distant room
rumble from triggering the assistant — webrtcvad alone classifies broadband
noise as "speech" far too readily, because it was designed for telephony
where anything non-silent is probably a caller.

Three cheap, independent tests:

  1. LOUDNESS vs the room's own noise floor. The floor is measured at
     startup (and drifts slowly), so this adapts to your actual room rather
     than a fixed number that's wrong everywhere. Fan noise sits at the
     floor by definition; speech sits well above it.

  2. SPEECH-BAND ENERGY RATIO. Human speech puts most of its energy between
     ~300Hz and ~3400Hz. Fans, HVAC and desk rumble are dominated by very
     low frequencies. If most of the energy is below the speech band, it
     isn't a voice.

  3. SPECTRAL FLATNESS. Noise is flat/broadband; voiced speech has strong
     harmonic peaks, so it's far "peakier". Flatness near 1 means noise-like,
     near 0 means tonal/harmonic.

All three run on numpy over a short frame — microseconds, no model needed.
"""
import numpy as np

import config

# --- tunables -------------------------------------------------------------
# IMPORTANT: these were originally tuned against SYNTHETIC test signals (a
# pure harmonic stack scores ~0.0 flatness), which is nothing like real mic
# audio. Real speech through a real microphone carries room noise, breath and
# unvoiced consonants, and typically lands around 0.3-0.7 flatness with a
# band ratio well under the old 0.45. Those strict values silently rejected
# genuine speech. Defaults below are deliberately permissive: the gate exists
# to reject FANS, not to second-guess a human voice. Speaker verification is
# what actually decides who's talking.
#
# Run `python3 voice/diagnose.py` to measure YOUR real numbers and tune these.
NOISE_FLOOR_MARGIN_DB = float(getattr(config, "GATE_NOISE_MARGIN_DB", 6.0))
MIN_SPEECH_BAND_RATIO = float(getattr(config, "GATE_MIN_BAND_RATIO", 0.20))
MAX_SPECTRAL_FLATNESS = float(getattr(config, "GATE_MAX_FLATNESS", 0.85))
ABSOLUTE_FLOOR_RMS = float(getattr(config, "GATE_ABSOLUTE_FLOOR_RMS", 0.003))


class NoiseProfile:
    """Tracks the room's ambient level so gating adapts to the environment
    instead of relying on a fixed threshold."""

    def __init__(self):
        self.floor_rms = 0.01     # sensible starting guess until calibrated
        self._calibrated = False

    def calibrate(self, pa, seconds: float = 1.0):
        """Listen briefly to the empty room to learn its noise floor."""
        import pyaudio
        frames_needed = int(config.SAMPLE_RATE * seconds / 480)
        try:
            stream = pa.open(rate=config.SAMPLE_RATE, channels=1,
                             format=pyaudio.paInt16, input=True,
                             frames_per_buffer=480)
        except Exception as e:
            print(f"[audio_gate] calibration skipped ({e}); using default floor")
            return
        levels = []
        try:
            for _ in range(frames_needed):
                raw = stream.read(480, exception_on_overflow=False)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                levels.append(float(np.sqrt(np.mean(samples ** 2)) + 1e-9))
        finally:
            try:
                stream.close()
            except Exception:
                pass
        if levels:
            measured = float(np.median(levels))
            # A reading this low means the mic returned silence — usually the
            # stream wasn't ready yet. Trusting it would make the gate behave
            # unpredictably, so keep the safe default instead.
            if measured < 1e-4:
                print(f"[audio_gate] calibration read silence "
                      f"({20 * np.log10(max(measured, 1e-9)):.0f} dBFS) — mic likely not ready; "
                      f"keeping default floor")
                return
            # median is robust to a cough or door slam during calibration
            self.floor_rms = measured
            self._calibrated = True
            print(f"[audio_gate] room noise floor: {20 * np.log10(self.floor_rms):.1f} dBFS")

    def update_floor(self, rms: float):
        """Slowly track the floor upward/downward on frames judged non-speech,
        so a fan switching on doesn't permanently break gating."""
        if rms < self.floor_rms * 3:
            self.floor_rms = 0.995 * self.floor_rms + 0.005 * rms


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)) + 1e-9)


def speech_band_ratio(samples: np.ndarray, sample_rate: int) -> float:
    """Fraction of spectral energy sitting in the human speech band."""
    n = len(samples)
    if n < 64:
        return 0.0
    windowed = samples * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    total = float(spec.sum()) + 1e-12
    band = float(spec[(freqs >= 300) & (freqs <= 3400)].sum())
    return band / total


def spectral_flatness(samples: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the spectrum. ~1 = flat noise,
    lower = harmonic/tonal, which is what voiced speech looks like."""
    n = len(samples)
    if n < 64:
        return 1.0
    windowed = samples * np.hanning(n)
    spec = np.abs(np.fft.rfft(windowed)) ** 2 + 1e-12
    geo = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    return float(geo / arith)


def looks_like_speech(pcm_bytes: bytes, profile: NoiseProfile,
                      sample_rate: int = None, verbose: bool = False):
    """Returns (passed, detail_dict). Only audio passing ALL checks should
    be treated as a real human voice."""
    sample_rate = sample_rate or config.SAMPLE_RATE
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size < 160:
        return False, {"reason": "too short"}

    level = rms(samples)
    ratio = speech_band_ratio(samples, sample_rate)
    flat = spectral_flatness(samples)

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
