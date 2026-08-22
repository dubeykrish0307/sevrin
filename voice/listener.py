"""
Records audio and uses VAD to detect when the person has actually finished
speaking.

Three layers of filtering before audio is accepted as a real command:

  1. webrtcvad decides which frames contain speech at all.
  2. voice/audio_gate.py checks the audio is physically voice-like — loud
     enough above the room's own noise floor, energy inside the speech band,
     harmonic rather than flat. This is what rejects fans and HVAC, which
     webrtcvad happily labels as speech.
  3. speaker verification confirms it's KRISH. This was previously missing
     entirely: only the wake word was verified, so once SEVRIN was awake,
     ANY nearby person could issue commands. Now every utterance is checked.
"""
import collections
import webrtcvad

import config
from voice import audio_device
from voice import audio_gate
from voice.voice_id import verify as speaker


class UtteranceListener:
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.pa = audio_device.get_pa()
        self.noise = audio_gate.NoiseProfile()
        self._verify_enabled = speaker.voiceprint_available()

    def calibrate_noise(self):
        """Learn the room's ambient level so noise gating adapts to it."""
        self.noise.calibrate(self.pa)

    def record_utterance(self, max_seconds: float = None) -> bytes:
        """
        Records from the mic until the person pauses. Returns raw PCM16
        audio, or b"" if what was heard didn't amount to real speech (too
        short / just noise, or nothing said within max_seconds) — callers
        should treat empty bytes as "nothing was said."

        max_seconds overrides config.MAX_UTTERANCE_SECONDS — use a shorter
        value for follow-up turns so silence times out on the window you
        actually configured instead of the (longer) default cap.
        """
        stream = audio_device.open_input_stream(
            config.SAMPLE_RATE,
            int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000),
        )

        silence_frames_needed = config.SILENCE_DURATION_MS // config.FRAME_DURATION_MS
        effective_max_seconds = max_seconds if max_seconds is not None else config.MAX_UTTERANCE_SECONDS
        max_frames = int(effective_max_seconds * 1000 / config.FRAME_DURATION_MS)

        PREROLL_FRAMES = 3  # ~90ms kept before speech onset so the first phoneme isn't clipped
        preroll = collections.deque(maxlen=PREROLL_FRAMES)
        ring = collections.deque(maxlen=silence_frames_needed)
        audio_frames = []
        speech_frame_count = 0
        heard_speech = False
        frame_count = 0

        try:
            while frame_count < max_frames:
                frame = stream.read(
                    int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000),
                    exception_on_overflow=False,
                )
                frame_count += 1
                is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)
                ring.append(is_speech)

                if is_speech:
                    speech_frame_count += 1

                # Before speech starts, don't accumulate audio_frames at all —
                # however long you pause before talking, none of that silence
                # gets sent to Whisper. Only the pre-roll (last ~90ms) is kept
                # so the very start of your first word isn't clipped.
                if not heard_speech:
                    preroll.append(frame)
                    if is_speech:
                        heard_speech = True
                        audio_frames.extend(preroll)
                else:
                    audio_frames.append(frame)

                if heard_speech and len(ring) == ring.maxlen and not any(ring):
                    # Speaker verification embeds a fixed-length window, and a
                    # very short command ("stop", "yes") doesn't contain enough
                    # audio for that — which showed up as genuine commands
                    # scoring 0.13. Keep capturing a little longer so there's
                    # enough to verify, while still ending promptly.
                    needed = int(config.SAMPLE_RATE * config.VOICE_ID_CLIP_SECONDS)
                    have = len(audio_frames) * int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000)
                    if have >= needed or frame_count >= max_frames:
                        break
                    extra = int((needed - have) / (config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000))
                    extra = min(extra, int(1.5 * 1000 / config.FRAME_DURATION_MS))
                    for _ in range(extra):
                        try:
                            f = stream.read(
                                int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000),
                                exception_on_overflow=False,
                            )
                        except Exception:
                            break
                        audio_frames.append(f)   # contiguous continuation
                    break
        finally:
            stream.close()

        # Not enough actual speech detected — likely just background noise.
        # Discard rather than send a near-empty/garbage clip to Whisper.
        if speech_frame_count < config.MIN_SPEECH_FRAMES:
            return b""

        audio = b"".join(audio_frames)

        # layer 2 — is this physically voice-like, or is it the fan?
        passed, detail = audio_gate.looks_like_speech(audio, self.noise)
        if not passed:
            print(f"[listener] ignored non-speech audio: {detail.get('reason')} "
                  f"(band {detail.get('speech_band_ratio')}, flat {detail.get('spectral_flatness')})")
            return b""

        # layer 3 — is this Krish? Previously commands were NOT checked at
        # all, so anyone nearby could issue them once SEVRIN was awake.
        if self._verify_enabled:
            ok, score, thr = speaker.verify(audio)
            if not ok:
                if score is None:
                    print("[listener] ignored: couldn't verify speaker (sample too short/quiet)")
                else:
                    print(f"[listener] ignored: not Krish (voice {score:.2f}, need {thr:.2f})")
                return b""
            print(f"[listener] speaker confirmed (voice {score:.2f} >= {thr:.2f})")

        return audio
