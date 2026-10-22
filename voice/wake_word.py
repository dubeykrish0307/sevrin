"""Streaming wake word ("hey jarvis" via openWakeWord) with speaker verification.

openWakeWord scores 80 ms frames as they arrive, so it fires the moment the
phrase is said. It can't tell whose voice it was, so the last
VOICE_ID_CLIP_SECONDS of audio sit in a ring buffer and get verified on a hit.
"""
import collections
import time

import numpy as np

import config
from voice import audio_device, audio_gate
from voice.voice_id import verify as speaker

FRAME_SAMPLES = 1280                # 80 ms at 16 kHz, openWakeWord's frame size
SILENT_FRAMES_BEFORE_RESET = 30     # ~2.5 s of exact zeros = wedged device, reopen it
MIN_VERIFY_FRAMES = int(config.SAMPLE_RATE * 1.2 / FRAME_SAMPLES)


def _join_ring(ring):
    """Join the buffered (index, frame) pairs. Returns (b"", False) if they
    aren't consecutive.

    Speaker embeddings need one continuous stretch of audio. Twice a change
    quietly punched holes in this buffer and my scores collapsed (0.64 -> 0.27),
    so every frame carries its index and gaps get caught here.
    """
    if not ring:
        return b"", False
    items = list(ring)
    for prev, cur in zip(items, items[1:]):
        if cur[0] != prev[0] + 1:
            return b"", False
    return b"".join(raw for _, raw in items), True


class WakeWordListener:
    def __init__(self):
        self._model = None
        self._verify = True
        self.noise = audio_gate.NoiseProfile()

    def set_noise_profile(self, profile):
        """Use the room profile the backend measured once the mic was warm."""
        if profile is not None:
            self.noise = profile

    def start(self):
        import openwakeword
        from openwakeword.model import Model

        print("[wake_word] loading openWakeWord model (downloads once on first run)...")
        openwakeword.utils.download_models([config.WAKE_WORD_MODEL])
        # onnx, because tflite wheels aren't reliable on Apple Silicon
        self._model = Model(wakeword_models=[config.WAKE_WORD_MODEL], inference_framework="onnx")

        self._verify = speaker.voiceprint_available()
        if not self._verify:
            print("\n[wake_word] WARNING: no voiceprint enrolled, so anyone's voice will wake SEVRIN.\n"
                  "  Run: python3 voice/voice_id/record.py   then   python3 voice/voice_id/enroll.py\n"
                  "  and restart the backend.\n")
        else:
            cal = speaker.calibration_summary() or {}
            if "genuine_mean" in cal:
                print(f"[wake_word] voiceprint: {cal.get('n_clips')} clips, "
                      f"genuine mean {cal['genuine_mean']:.3f}, threshold {cal['threshold']:.3f}")
            else:
                print("[wake_word] voiceprint loaded (uncalibrated, re-run enroll.py "
                      "to calibrate the threshold from your actual voice)")

        suffix = " (your voice only)" if self._verify else " (voice check disabled)"
        print(f'[wake_word] listening for "hey jarvis"{suffix}')

    def listen_once(self) -> str:
        """Block until the wake word is heard and, if enrolled, verified. Opens
        and closes its own stream so it doesn't fight the command listener."""
        ring = collections.deque(
            maxlen=int(config.SAMPLE_RATE * config.VOICE_ID_CLIP_SECONDS / FRAME_SAMPLES) + 1)
        index = 0
        silent_run = 0

        stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
        try:
            while True:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                # every frame goes in, whatever happens below; skipping any splices the audio
                ring.append((index, raw))
                index += 1

                if audio_device.is_silent(raw):
                    silent_run += 1
                    if silent_run >= SILENT_FRAMES_BEFORE_RESET:
                        print("[wake_word] microphone returned only silence, reopening the audio stream")
                        try:
                            stream.close()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
                        ring.clear()
                        silent_run = 0
                    continue
                silent_run = 0

                prediction = self._model.predict(np.frombuffer(raw, dtype=np.int16))
                if prediction.get(config.WAKE_WORD_MODEL, 0.0) < config.WAKE_WORD_THRESHOLD:
                    continue

                # Right after a drain the ring is nearly empty and openWakeWord can
                # fire again from its own state. Keep collecting instead of rejecting.
                if len(ring) < MIN_VERIFY_FRAMES:
                    continue

                clip, contiguous = _join_ring(ring)
                if not contiguous:
                    print("[wake_word] buffer wasn't contiguous; skipping this detection "
                          "rather than scoring unreliable audio")
                    ring.clear()
                    continue

                passed, detail = audio_gate.looks_like_speech(clip, self.noise)
                if not passed:
                    # always log this; a silent reject just looks like the wake word not working
                    print(f'[wake_word] heard "hey jarvis" but audio gate rejected it: '
                          f'{detail.get("reason")} '
                          f'({detail.get("db_above_floor")}dB above floor, '
                          f'band {detail.get("speech_band_ratio")}, '
                          f'flat {detail.get("spectral_flatness")})')
                    self._drain(stream, 0.5, ring)
                    continue

                if self._verify:
                    ok, score, thr = speaker.verify(clip)
                    if score is None:
                        print('[wake_word] heard "hey jarvis" but clip was too short to verify, ignoring')
                        self._drain(stream, 1.0, ring)
                        continue
                    if not ok:
                        print(f'[wake_word] heard "hey jarvis" from someone else '
                              f'({speaker.score_text(score)}, need {thr:.2f}), ignoring')
                        self._drain(stream, 1.5, ring)
                        continue
                    print(f"[wake_word] voice confirmed ({speaker.score_text(score)} >= {thr:.2f})")

                return "hey jarvis"
        finally:
            stream.close()

    def _drain(self, stream, seconds: float, ring):
        """Skip past the rest of an utterance we just handled so it doesn't
        re-trigger. The ring has to be cleared too, or the next clip gets
        spliced across the gap."""
        for _ in range(int(config.SAMPLE_RATE * seconds / FRAME_SAMPLES)):
            stream.read(FRAME_SAMPLES, exception_on_overflow=False)
        ring.clear()
