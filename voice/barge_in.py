"""Lets Krish talk over SEVRIN mid-sentence.

Listens on the mic while TTS plays. The mic also hears SEVRIN's own voice;
instead of proper echo cancellation, anything heard has to pass speaker
verification, and the TTS voice doesn't match the voiceprint. Works best on
headphones; very loud speakers can still trip it.

This is the strictest listener on purpose. Stopping mid-sentence for a fan
is far more annoying than having to repeat yourself.
"""
import collections
import threading

import webrtcvad

import config
from voice import audio_device, audio_gate
from voice.voice_id import verify as speaker

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)
SPEECH_FRAMES_TO_TRIGGER = 16       # ~480 ms of sustained speech, so a cough doesn't count
VERIFY_CLIP_SECONDS = 1.8
VAD_AGGRESSIVENESS = 3              # strictest setting
# has to be clearly someone speaking up, so 4 dB more than the normal gate wants
MIN_DB_ABOVE_FLOOR = audio_gate.NOISE_FLOOR_MARGIN_DB + 4


class BargeInListener:
    """Watches the mic during playback and calls on_interrupt() once when a
    verified interruption starts. Uses the shared PyAudio instance."""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.noise = audio_gate.NoiseProfile()
        self._thread = None
        self._stop = threading.Event()
        self._verify = speaker.voiceprint_available()
        self.captured_audio = b""       # the interruption itself, for transcription

    def set_noise_profile(self, profile):
        """Can't calibrate while audio is playing, so take the listener's profile."""
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
            stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
        except Exception as e:
            print(f"[barge_in] couldn't open mic: {e}")
            return

        ring = collections.deque(maxlen=int(config.SAMPLE_RATE * VERIFY_CLIP_SECONDS / FRAME_SAMPLES))
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
                passed, detail = audio_gate.looks_like_speech(clip, self.noise)
                if not passed or detail.get("db_above_floor", 0) < MIN_DB_ABOVE_FLOOR:
                    speech_run = 0
                    continue

                if self._verify:
                    ok, score, thr = speaker.verify(clip)
                    if not ok:
                        # our own voice coming back through the speakers, or someone else
                        speech_run = 0
                        continue
                    print(f"[barge_in] interruption from Krish ({speaker.score_text(score)} >= {thr:.2f})")
                else:
                    print("[barge_in] interruption detected (no voiceprint, can't verify speaker)")

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
