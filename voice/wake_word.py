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
import time

import numpy as np

import config
from voice import audio_device
from voice.voice_id.verify import verify as verify_speaker, voiceprint_available, calibration_summary
from voice import audio_gate

FRAME_SAMPLES = 1280  # 80ms @ 16kHz — openWakeWord's native frame size


def _join_ring(ring):
    """Concatenate the ring's audio, verifying the frames are consecutive.

    Returns (audio_bytes, is_contiguous). Speaker embeddings are only valid on
    a continuous slice of time; joining non-adjacent frames produces audio the
    model scores as a stranger. Checking indices makes that structurally
    impossible to miss instead of relying on it never happening."""
    if not ring:
        return b"", False
    items = list(ring)
    for prev, cur in zip(items, items[1:]):
        if cur[0] != prev[0] + 1:
            return b"", False
    return b"".join(raw for _, raw in items), True

# ~2.5s of pure digital silence means the audio device has wedged (a real
# quiet room still carries dither), so the stream gets reopened.
SILENT_FRAMES_BEFORE_RESET = 30


class WakeWordListener:
    def __init__(self):
        self.pa = audio_device.get_pa()
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
        # Each entry is (frame_index, raw_bytes). Storing the index makes
        # contiguity CHECKABLE rather than assumed — twice now a change has
        # silently punched holes in this buffer, and both times genuine voice
        # scores collapsed while everything looked superficially fine.
        ring = collections.deque(maxlen=buffer_frames)
        # Verification needs a real amount of audio; below this the embedding
        # is unreliable and the verifier refuses to score it anyway.
        min_frames_to_verify = int(config.SAMPLE_RATE * 1.2 / FRAME_SAMPLES)
        frame_index = 0

        stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
        silent_run = 0
        try:
            while True:
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)

                # CRITICAL: append EVERY frame, unconditionally.
                #
                # The ring buffer is what gets sent to speaker verification,
                # and it must be a CONTIGUOUS slice of time. An earlier version
                # skipped silent frames before appending, which spliced
                # non-adjacent audio together — ECAPA embeds that as garbage
                # and genuine scores collapsed from 0.64 to 0.27. Whatever
                # filtering we do, it must never punch holes in this buffer.
                ring.append((frame_index, raw))
                frame_index += 1

                # Device-health check only — deliberately does NOT affect the
                # ring. A wedged CoreAudio device returns exactly-zero buffers
                # forever; without noticing, the loop spins on a dead mic and
                # the assistant appears to hang.
                if audio_device.is_silent(raw):
                    silent_run += 1
                    if silent_run >= SILENT_FRAMES_BEFORE_RESET:
                        print("[wake_word] microphone returned only silence — "
                              "reopening the audio stream")
                        try:
                            stream.close()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        stream = audio_device.open_input_stream(
                            config.SAMPLE_RATE, FRAME_SAMPLES)
                        ring.clear()   # safe: we're deliberately restarting
                        silent_run = 0
                    continue
                silent_run = 0
                frame = np.frombuffer(raw, dtype=np.int16)

                prediction = self._model.predict(frame)
                ww_score = prediction.get(config.WAKE_WORD_MODEL, 0.0)

                if ww_score < config.WAKE_WORD_THRESHOLD:
                    continue

                # The ring may not hold enough audio yet — right after a drain
                # it's deliberately empty, and openWakeWord can re-fire within
                # a second on its own internal state. Verifying then produced
                # "clip was too short to verify" over and over. Instead of
                # rejecting, just keep collecting until there's enough.
                if len(ring) < min_frames_to_verify:
                    continue

                clip, contiguous = _join_ring(ring)
                if not contiguous:
                    # Refuse to score spliced audio — it embeds as noise and
                    # would be misreported as "someone else".
                    print("[wake_word] buffer wasn't contiguous; skipping this "
                          "detection rather than scoring unreliable audio")
                    ring.clear()
                    continue

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
                    self._drain(stream, seconds=0.5, ring=ring)
                    continue

                if self._voice_id_enabled:
                    ok, voice_score, thr = verify_speaker(clip)
                    if voice_score is None:
                        print('[wake_word] heard "hey jarvis" but clip was too short to verify — ignoring')
                        self._drain(stream, seconds=1.0, ring=ring)
                        continue
                    if not ok:
                        print(f'[wake_word] heard "hey jarvis" from someone else '
                              f'(voice {voice_score:.2f}, need {thr:.2f}) — ignoring')
                        self._drain(stream, seconds=1.5, ring=ring)
                        continue
                    print(f"[wake_word] voice confirmed ({voice_score:.2f} >= {thr:.2f})")

                return "hey jarvis"
        finally:
            stream.close()

    def _drain(self, stream, seconds: float, ring=None):
        """Reads and discards audio for `seconds` — used to skip past the
        tail of an utterance we just rejected/handled so it doesn't
        immediately re-trigger the wake word loop.

        The ring MUST be cleared alongside this. Otherwise the frames sitting
        in it from before the drain get joined to frames from after it, and
        the next verification runs on audio spliced across a gap of exactly
        `seconds` — which wrecks the embedding."""
        frames_to_drain = int(config.SAMPLE_RATE * seconds / FRAME_SAMPLES)
        for _ in range(frames_to_drain):
            stream.read(FRAME_SAMPLES, exception_on_overflow=False)
        if ring is not None:
            ring.clear()

    def close(self):
        # shared device; terminating here would break the other listeners
        pass
