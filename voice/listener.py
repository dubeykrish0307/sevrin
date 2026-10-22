"""Records one spoken command and decides whether it counts.

  1. webrtcvad finds the speech and the pause that ends it
  2. audio_gate rejects fans and other noise that VAD lets through
  3. speaker verification on every command, not just the wake word, so
     nobody else can drive SEVRIN once it's awake
"""
import collections

import webrtcvad

import config
from voice import audio_device, audio_gate
from voice.voice_id import verify as speaker

FRAME_SAMPLES = int(config.SAMPLE_RATE * config.FRAME_DURATION_MS / 1000)
PREROLL_FRAMES = 3      # ~90 ms from before speech starts, so the first word isn't clipped
CLIP_SAMPLES = int(config.SAMPLE_RATE * config.VOICE_ID_CLIP_SECONDS)
MAX_EXTRA_FRAMES = int(1.5 * 1000 / config.FRAME_DURATION_MS)


class UtteranceListener:
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        self.noise = audio_gate.NoiseProfile()
        self._verify = speaker.voiceprint_available()

    def calibrate_noise(self):
        self.noise.calibrate()

    def record_utterance(self, max_seconds: float = None) -> bytes:
        """Record until he pauses. Returns PCM16 audio, or b"" if nothing real
        was said (noise, someone else, or silence until max_seconds)."""
        if max_seconds is None:
            max_seconds = config.MAX_UTTERANCE_SECONDS
        max_frames = int(max_seconds * 1000 / config.FRAME_DURATION_MS)
        silence_frames = config.SILENCE_DURATION_MS // config.FRAME_DURATION_MS

        preroll = collections.deque(maxlen=PREROLL_FRAMES)
        recent = collections.deque(maxlen=silence_frames)   # speech/non-speech flags
        frames = []
        speech_frames = 0
        started = False
        count = 0

        stream = audio_device.open_input_stream(config.SAMPLE_RATE, FRAME_SAMPLES)
        try:
            while count < max_frames:
                frame = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                count += 1
                is_speech = self.vad.is_speech(frame, config.SAMPLE_RATE)
                recent.append(is_speech)
                if is_speech:
                    speech_frames += 1

                # silence before he starts talking isn't kept, only the pre-roll
                if not started:
                    preroll.append(frame)
                    if is_speech:
                        started = True
                        frames.extend(preroll)
                else:
                    frames.append(frame)

                if started and len(recent) == recent.maxlen and not any(recent):
                    # Verification needs a full VOICE_ID_CLIP_SECONDS of audio. A short
                    # "stop" or "yes" scored 0.13 without it, so record a bit longer.
                    missing = CLIP_SAMPLES - len(frames) * FRAME_SAMPLES
                    if missing > 0 and count < max_frames:
                        for _ in range(min(int(missing / FRAME_SAMPLES), MAX_EXTRA_FRAMES)):
                            try:
                                frames.append(stream.read(FRAME_SAMPLES, exception_on_overflow=False))
                            except Exception:
                                break
                    break
        finally:
            stream.close()

        if speech_frames < config.MIN_SPEECH_FRAMES:
            return b""   # just background noise

        audio = b"".join(frames)

        passed, detail = audio_gate.looks_like_speech(audio, self.noise)
        if not passed:
            print(f"[listener] ignored non-speech audio: {detail.get('reason')} "
                  f"(band {detail.get('speech_band_ratio')}, flat {detail.get('spectral_flatness')})")
            return b""

        if self._verify:
            ok, score, thr = speaker.verify(audio)
            if not ok:
                if score is None:
                    print("[listener] ignored: couldn't verify speaker (sample too short/quiet)")
                else:
                    print(f"[listener] ignored: not Krish ({speaker.score_text(score)}, need {thr:.2f})")
                return b""
            print(f"[listener] speaker confirmed ({speaker.score_text(score)} >= {thr:.2f})")

        return audio
