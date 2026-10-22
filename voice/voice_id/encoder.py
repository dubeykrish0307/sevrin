"""Speaker embeddings.

ECAPA-TDNN from SpeechBrain (spkrec-ecapa-voxceleb, ~0.8% EER on VoxCeleb).
I started on Resemblyzer, but it couldn't even match my own voice across
recording conditions: 0.56-0.66 live against enrolment clips that scored 0.86
among themselves. With ECAPA the same speaker usually lands 0.6-0.9 and
different speakers 0.0-0.3. Resemblyzer is still the fallback if SpeechBrain
isn't installed.

Never build a clip by gluing non-adjacent pieces of audio together (e.g.
VAD-trimming out the silences). ECAPA embeds the joins as noise: when I tried
it, enrolment mean fell from 0.79 to 0.51 and live scores from 0.55 to 0.18.
"""
import os
import warnings
import wave

import numpy as np

SAMPLE_RATE = 16000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ecapa")

_backend = None     # "ecapa" or "resemblyzer"
_model = None


def backend_name() -> str:
    _load()
    return _backend


def _load():
    global _model, _backend
    if _model is not None:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("[voice_id] loading ECAPA-TDNN speaker model (one-time download ~80MB)...")
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODEL_DIR,
            run_opts={"device": "cpu"},
        )
        _backend = "ecapa"
        print("[voice_id] using ECAPA-TDNN (EER ~0.8% on VoxCeleb)")
        return
    except Exception as e:
        print(f"[voice_id] SpeechBrain unavailable ({e})")
        print("[voice_id] falling back to Resemblyzer, which separates speakers much worse.")
        print("[voice_id] install with:  pip install speechbrain")

    from resemblyzer import VoiceEncoder
    _model = VoiceEncoder()
    _backend = "resemblyzer"


def normalize_level(audio: np.ndarray, target_rms: float = 0.06):
    """Bring audio to a consistent loudness. A laptop mic at arm's length and an
    earbud mic against the cheek differ a lot in level, and level alone moves
    the embedding."""
    if audio is None or audio.size == 0:
        return audio
    level = float(np.sqrt(np.mean(audio ** 2)))
    if level < 1e-6:
        return audio
    out = audio * min(target_rms / level, 20.0)   # cap the gain so near-silence isn't blown up
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def _unit(vec):
    norm = np.linalg.norm(vec)
    return None if norm < 1e-8 else vec / norm


def embed(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """float32 mono audio in [-1, 1] -> unit-length embedding, or None if it's too short."""
    _load()
    if audio is None or audio.size == 0:
        return None
    audio = normalize_level(audio)

    if _backend == "ecapa":
        import torch
        if audio.size < sample_rate * 0.4:
            return None
        with torch.no_grad():
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            return _unit(_model.encode_batch(wav).squeeze().cpu().numpy())

    from resemblyzer import preprocess_wav
    try:
        wav = preprocess_wav(audio, source_sr=sample_rate)
    except Exception:
        return None
    if len(wav) < sample_rate * 0.4:
        return None
    return _unit(_model.embed_utterance(wav))


def load_wav(path: str) -> np.ndarray:
    """Read a wav as float32 mono at 16 kHz (linear resample if needed)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio)
    return audio


def embed_file(path: str):
    return embed(load_wav(path), SAMPLE_RATE)


def embed_file_windows(path: str, window_seconds: float, hop_seconds: float = None):
    """Embed a recording as overlapping fixed-length windows.

    Live clips are a fixed 2.5 s, so the voiceprint should be built from audio
    of the same length. Embedding whole files instead gave an enrolment mean of
    0.64 but live scores of only 0.42-0.46. Each window is one contiguous slice.
    """
    _load()
    audio = load_wav(path)
    win = int(SAMPLE_RATE * window_seconds)
    hop = int(SAMPLE_RATE * (hop_seconds if hop_seconds else window_seconds / 2))

    if len(audio) < win:
        e = embed(audio, SAMPLE_RATE)
        return [e] if e is not None else []

    out = []
    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]
        if float(np.sqrt(np.mean(chunk ** 2))) < 0.005:
            continue    # near-silent windows would drag the centroid towards silence
        e = embed(chunk, SAMPLE_RATE)
        if e is not None:
            out.append(e)
    return out
