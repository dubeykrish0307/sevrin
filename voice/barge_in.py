"""
SEVRIN — barge-in detection.

Listens on the mic WHILE SEVRIN is speaking, so Krish can cut him off
mid-sentence the way you'd interrupt a person.

THE ECHO PROBLEM
Listening during playback means the mic also picks up SEVRIN's own voice
through the speakers. Naively, he'd interrupt himself constantly. Rather
than implement acoustic echo cancellation (genuinely hard), we reuse the
speaker verification that already exists: any speech detected during
playback is checked against Krish's enrolled voiceprint. SEVRIN's TTS voice
doesn't match it, so his own output is rejected automatically.

Caveat worth knowing: this works best on headphones. On loud speakers, a
strong echo can occasionally still trip the VAD, and very loud playback can
mask a quiet interruption. If that becomes a problem in practice, the next
step is a proper AEC layer — but voiceprint gating handles the common case
without that complexity.

Two thresholds matter:
  - it must sound like speech (webrtcvad) for long enough to not be a cough
  - it must sound like KRISH (resemblyzer) to count as a real interruption
"""
import collections
import threading

import numpy as np
import pyaudio
import webrtcvad

import config
from voice.voice_id.verify import verify as verify_speaker, voiceprint_available
from voice import audio_gate

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)

# how much continuous speech before we treat it as a real interruption.
# Too low and a cough stops him; too high and you have to shout over him.
# Barge-in is the STRICTEST path in the system, deliberately. A false
# interrupt is very annoying (SEVRIN stops mid-sentence for a fan), while a
# missed one costs you only a repeat. So we demand more speech, a longer
# verification clip, and maximum VAD aggressiveness here.
SPEECH_FRAMES_TO_TRIGGER = 16         # ~480ms of sustained speech (was 240ms)
VERIFY_CLIP_SECONDS = 1.8             # longer clip = far more reliable speaker check
BARGE_VAD_AGGRESSIVENESS = 3          # most aggressive noise filtering


class BargeInListener:
    """Runs in a background thread during playback. Calls on_interrupt()
    once, the moment a verified interruption from Krish is detected."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(BARGE_VAD_AGGRESSIVENESS)
        self.noise = audio_gate.NoiseProfile()
        self._thread = None
        self._stop = threading.Event()
        self._voice_id = voiceprint_available()
        self.captured_audio = b""     # audio of the interruption itself

    def set_noise_profile(self, profile):
        """Reuse the noise floor already measured by the main listener rather
        than re-calibrating (which would need the mic during playback)."""
        if profile is not None:
            self.noise = profile

    def start(self, on_interrupt):
        self._stop.clear()
        self.captured_audio = b""
        self._thread = threading.Thread(target=self._run, args=(on_interrupt,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self, on_interrupt):
        try:
            stream = self.pa.open(
                rate=config.SAMPLE_RATE, channels=1, format=pyaudio.paInt16,
                input=True, frames_per_buffer=FRAME_SAMPLES,
            )
        except Exception as e:
            print(f"[barge_in] couldn't open mic: {e}")
            return

        ring_frames = int(config.SAMPLE_RATE * VERIFY_CLIP_SECONDS / FRAME_SAMPLES)
        ring = collections.deque(maxlen=ring_frames)
        speech_run = 0

        try:
            while not self._stop.is_set():
                try:
                    frame = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                except Exception:
                    break
                ring.append(frame)

                try:
                    is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)
                except Exception:
                    is_speech = False

                if not is_speech:
                    speech_run = max(0, speech_run - 1)
                    continue

                speech_run += 1
                if speech_run < SPEECH_FRAMES_TO_TRIGGER:
                    continue

                clip = b"".join(ring)

                # physical gate — this is what stops a fan from interrupting.
                # Barge-in demands MORE margin above the noise floor than the
                # normal listener, since a false interrupt is so disruptive.
                passed, detail = audio_gate.looks_like_speech(clip, self.noise)
                if not passed:
                    speech_run = 0
                    continue
                if detail.get("db_above_floor", 0) < audio_gate.NOISE_FLOOR_MARGIN_DB + 4:
                    # loud-ish but not clearly someone speaking up
                    speech_run = 0
                    continue

                # Is this actually Krish, or SEVRIN's own voice echoing back?
                if self._voice_id:
                    ok, score, thr = verify_speaker(clip)
                    if not ok:
                        # SEVRIN's own output bleeding in, or another person —
                        # either way, not an interruption we should honour
                        speech_run = 0
                        continue
                    print(f"[barge_in] interruption from Krish (voice {score:.2f} >= {thr:.2f})")
                else:
                    print("[barge_in] interruption detected (no voiceprint — can't verify speaker)")

                self.captured_audio = clip
                try:
                    on_interrupt()
                except Exception as e:
                    print(f"[barge_in] handler error: {e}")
                break
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def close(self):
        try:
            self.pa.terminate()
        except Exception:
            pass
