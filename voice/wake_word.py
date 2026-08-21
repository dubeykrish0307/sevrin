"""
True streaming wake-word detection using openWakeWord's pretrained
"hey jarvis" model, layered with speaker verification.

This replaces the old approach (record a full utterance -> wait for you to
go silent -> transcribe the whole clip with Whisper -> fuzzy-match the
text). That round-trip is where most of the "lag before it wakes up" was
coming from. openWakeWord instead scores small (80ms) audio frames
continuously as they stream in from the mic, so detection fires the moment
the phrase is heard.

Flow per listen_once() call:
  1. Open the mic and read 80ms frames in a loop.
  2. Feed each frame to openWakeWord. It returns a 0-1 score for how
     confident it is that "hey jarvis" was just heard.
  3. The moment the score crosses the threshold, the wake word has fired.
  4. openWakeWord only tells us THAT the phrase was heard, not whose voice
     it was — so we keep a rolling ~1.5s buffer of raw audio, and the
     moment we detect a wake, we run that buffered clip through the
     Resemblyzer voice verifier to confirm it's actually you.
  5. If no voiceprint has been enrolled yet, voice verification is skipped
     (with a warning) instead of crashing the whole program.
"""
import collections

import numpy as np
import pyaudio

import config
from voice.voice_id.verify import verify as verify_speaker, voiceprint_available, calibration_summary
from voice import audio_gate

FRAME_SAMPLES = 1280  # 80ms @ 16kHz — openWakeWord's native frame size


class WakeWordListener:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self._model = None
        self._voice_id_enabled = True
        self.noise = audio_gate.NoiseProfile()

    def set_noise_profile(self, profile):
        """Share the room profile measured elsewhere, rather than calibrating
        here (see start(): calibrating this early reads silence)."""
        if profile is not None:
            self.noise = profile

    def start(self):
        import openwakeword
        from openwakeword.model import Model

        print("[wake_word] loading openWakeWord model (downloads once on first run)...")
        openwakeword.utils.download_models([config.WAKE_WORD_MODEL])
        # inference_framework="onnx" explicitly — tflite wheels aren't
        # reliably available on Apple Silicon, onnx is.
        self._model = Model(
            wakeword_models=[config.WAKE_WORD_MODEL],
            inference_framework="onnx",
        )

        self._voice_id_enabled = voiceprint_available()
        if not self._voice_id_enabled:
            print(
                "\n[wake_word] WARNING: no enrolled voiceprint found — anyone's "
                "voice will trigger the wake word right now.\n"
                "  Run: python3 voice_id/record.py   then   python3 voice_id/enroll.py\n"
                "  Restart main.py afterward to enable voice-only wake.\n"
            )

        if self._voice_id_enabled:
            cal = calibration_summary() or {}
            if "genuine_mean" in cal:
                print(f"[wake_word] voiceprint: {cal.get('n_clips')} clips, "
                      f"genuine mean {cal['genuine_mean']:.3f}, threshold {cal['threshold']:.3f}")
            else:
                print("[wake_word] voiceprint loaded (uncalibrated — re-run enroll.py "
                      "to calibrate the threshold from your actual voice)")

        # NOTE: no calibration here. Calibrating at this point read pure
        # silence (mic not warm yet) and produced a nonsense -100 dBFS floor.
        # The backend calibrates once, after the mic has been opened, and
        # shares that profile via set_noise_profile().

        suffix = " (your voice only)" if self._voice_id_enabled else " (voice check disabled)"
        print(f'[wake_word] listening for "hey jarvis"{suffix}')

    def listen_once(self) -> str:
        """Blocks until the wake word is heard (and, if enrolled, verified
        as your voice). Opens its own mic stream and closes it before
        returning, so it doesn't fight with the conversation listener for
        the input device."""
        buffer_frames = int(config.SAMPLE_RATE * config.VOICE_ID_CLIP_SECONDS / FRAME_SAMPLES) + 1
        ring = collections.deque(maxlen=buffer_frames)

        stream = self.pa.open(
            rate=config.SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        try:
            while True:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                ring.append(raw)
                frame = np.frombuffer(raw, dtype=np.int16)

                prediction = self._model.predict(frame)
                ww_score = prediction.get(config.WAKE_WORD_MODEL, 0.0)

                if ww_score < config.WAKE_WORD_THRESHOLD:
                    continue

                clip = b"".join(ring)

                # physical gate first — cheap, and rejects fan/HVAC before we
                # spend time on speaker verification
                passed, detail = audio_gate.looks_like_speech(clip, self.noise)
                if not passed:
                    # ALWAYS log this. A previous version discarded audio here
                    # silently, which made a too-strict gate look like the wake
                    # word simply not working.
                    print(f'[wake_word] heard "hey jarvis" but audio gate rejected it: '
                          f'{detail.get("reason")} '
                          f'({detail.get("db_above_floor")}dB above floor, '
                          f'band {detail.get("speech_band_ratio")}, '
                          f'flat {detail.get("spectral_flatness")})')
                    self._drain(stream, seconds=0.5)
                    continue

                if self._voice_id_enabled:
                    ok, voice_score, thr = verify_speaker(clip)
                    if voice_score is None:
                        print('[wake_word] heard "hey jarvis" but clip was too short to verify — ignoring')
                        self._drain(stream, seconds=1.0)
                        continue
                    if not ok:
                        print(f'[wake_word] heard "hey jarvis" from someone else '
                              f'(voice {voice_score:.2f}, need {thr:.2f}) — ignoring')
                        self._drain(stream, seconds=1.5)
                        continue
                    print(f"[wake_word] voice confirmed ({voice_score:.2f} >= {thr:.2f})")

                return "hey jarvis"
        finally:
            stream.close()

    def _drain(self, stream, seconds: float):
        """Reads and discards audio for `seconds` — used to skip past the
        tail of an utterance we just rejected/handled so it doesn't
        immediately re-trigger the wake word loop."""
        frames_to_drain = int(config.SAMPLE_RATE * seconds / FRAME_SAMPLES)
        for _ in range(frames_to_drain):
            stream.read(FRAME_SAMPLES, exception_on_overflow=False)

    def close(self):
        self.pa.terminate()
