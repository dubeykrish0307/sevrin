"""
Speaker embedding backend.

Uses SpeechBrain's ECAPA-TDNN (spkrec-ecapa-voxceleb), which reports ~0.8%
EER on VoxCeleb1-cleaned. Resemblyzer, which this replaces, is a 2018-era
model whose embedding space simply wasn't discriminative enough here — the
symptom was Krish's own voice scoring 0.56-0.66 live against enrollment
clips that scored 0.86 among themselves. No threshold can rescue an
embedding space that can't separate one speaker from himself across
recording conditions.

ECAPA-TDNN separates far more sharply: same-speaker cosine scores typically
land 0.6-0.9 while different speakers sit near 0.0-0.3, which leaves a real
margin to put a threshold in.

Falls back to Resemblyzer automatically if SpeechBrain isn't installed, so
the system still runs — just less accurately, with a warning.
"""
import os
import warnings

import numpy as np


# ---------------------------------------------------------------------------
# REMOVED: trim_to_speech()
#
# An earlier version ran VAD over the audio, kept only the speech frames, and
# concatenated them before embedding — the goal being to stop silence in the
# 2.5s wake buffer from diluting the embedding.
#
# It made things much WORSE, measured on real audio:
#     enrollment mean  0.786 -> 0.506
#     live wake score  0.55  -> 0.18
#
# The flaw: it spliced together frames that were never contiguous. Each
# junction is an artificial discontinuity, and ECAPA is trained on natural
# continuous speech, so the glued audio embeds badly. It also shifted the
# whole embedding space enough that different clips got flagged as outliers.
#
# If silence dilution is worth attacking again, the safe form is to slice ONE
# contiguous region (first speech frame to last) without splicing — never to
# concatenate disjoint pieces. Do not re-add the concatenating version.
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ecapa")

_backend = None      # "ecapa" | "resemblyzer"
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
            from speechbrain.inference.speaker import EncoderClassifier
            import torch  # noqa: F401
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
        print("[voice_id] falling back to Resemblyzer — MUCH weaker separation.")
        print("[voice_id] install with:  pip install speechbrain")

    from resemblyzer import VoiceEncoder
    _model = VoiceEncoder()
    _backend = "resemblyzer"


def embed(audio: np.ndarray, sample_rate: int = 16000):
    """audio: float32 mono in [-1, 1] at 16kHz. Returns a unit-norm embedding,
    or None if the audio can't be embedded."""
    _load()
    if audio is None or audio.size == 0:
        return None

    if _backend == "ecapa":
        import torch
        if audio.size < sample_rate * 0.4:
            return None
        with torch.no_grad():
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb = _model.encode_batch(wav).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return None
        return emb / norm

    # resemblyzer fallback
    from resemblyzer import preprocess_wav
    try:
        wav = preprocess_wav(audio, source_sr=sample_rate)
    except Exception:
        return None
    if len(wav) < sample_rate * 0.4:
        return None
    emb = _model.embed_utterance(wav)
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return None
    return emb / norm


def load_wav(path: str):
    """Load a wav as float32 mono at 16kHz."""
    import wave
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio)
    return audio


def embed_file(path: str):
    """Embed a whole wav file."""
    _load()
    return embed(load_wav(path), 16000)


def embed_file_windows(path: str, window_seconds: float, hop_seconds: float = None):
    """Embed a recording as a series of fixed-length CONTIGUOUS windows.

    This is domain matching, and it matters a lot. Live wake clips are a fixed
    2.5s buffer, but enrollment recordings are 2s wake phrases (mostly silence)
    and one long 45s monologue. Embedding those whole produces a voiceprint
    that doesn't represent what live audio looks like — measured effect: an
    enrollment mean of 0.64 but live scores of 0.42-0.46, leaving almost no
    margin over an impostor.

    Slicing the long recording into windows the same length as a live clip
    gives many enrollment embeddings drawn from the same distribution as the
    audio we actually verify against.

    Each window is a contiguous slice — never spliced. Returns a list of
    embeddings (possibly empty).
    """
    _load()
    audio = load_wav(path)
    sr = 16000
    win = int(sr * window_seconds)
    hop = int(sr * (hop_seconds if hop_seconds else window_seconds / 2))

    if len(audio) < win:
        e = embed(audio, sr)
        return [e] if e is not None else []

    out = []
    for start in range(0, len(audio) - win + 1, hop):
        chunk = audio[start:start + win]          # contiguous by construction
        # skip near-silent windows; they'd drag the centroid toward silence
        if float(np.sqrt(np.mean(chunk ** 2))) < 0.005:
            continue
        e = embed(chunk, sr)
        if e is not None:
            out.append(e)
    return out
