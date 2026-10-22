"""Impostor cohort for AS-norm score normalisation.

A raw cosine score shifts whenever the mic or room changes: enrol on the
MacBook mic, talk through AirPods, and every score drops even though the
speaker is the same. AS-norm measures a score relative to how the same audio
scores against a set of other speakers, and those shift with the channel too.

The cohort is built once, from ElevenLabs voices if there's an API key
(closer to real people) or the macOS system voices otherwise.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

import config

COHORT_PATH = os.path.join(os.path.dirname(__file__), "profile", "cohort.npy")
COHORT_META = os.path.join(os.path.dirname(__file__), "profile", "cohort_source.txt")
MIN_VOICES = 5

# varied phonetic content so the embeddings aren't all about one phrase
_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells seashells by the seashore on a bright summer morning.",
    "Technology moves faster than most people can reasonably follow.",
    "Please remind me about the meeting scheduled for tomorrow afternoon.",
    "Numbers like seventeen, forty two and ninety nine appear frequently.",
]


def load():
    if not os.path.exists(COHORT_PATH):
        return None
    try:
        return np.load(COHORT_PATH)
    except Exception:
        return None


def source() -> str:
    """Which voices the cohort was built from: 'elevenlabs' or 'macos'."""
    try:
        with open(COHORT_META) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _save(arr, name: str):
    os.makedirs(os.path.dirname(COHORT_PATH), exist_ok=True)
    np.save(COHORT_PATH, arr)
    try:
        with open(COHORT_META, "w") as f:
            f.write(name)
    except OSError:
        pass


def _embed_synthesized(items, synthesize, ext: str, label, verbose: bool):
    """synthesize(item, sentence, path) writes an audio file for one voice.
    Each file is converted to 16 kHz mono with ffmpeg and embedded."""
    from voice.voice_id import encoder

    embeddings = []
    tmpdir = tempfile.mkdtemp(prefix="sevrin_cohort_")
    try:
        for i, item in enumerate(items):
            raw = os.path.join(tmpdir, f"{i}{ext}")
            wav = os.path.join(tmpdir, f"{i}.wav")
            try:
                synthesize(item, _SENTENCES[i % len(_SENTENCES)], raw)
                subprocess.run(["ffmpeg", "-y", "-i", raw, "-ar", "16000", "-ac", "1", wav],
                               check=True, capture_output=True, timeout=30)
                emb = encoder.embed_file(wav)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                if verbose:
                    print(f"[cohort]   skipped {label(item)} ({e})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return embeddings


def _elevenlabs_voices(limit=60):
    import requests
    if not config.ELEVENLABS_API_KEY:
        return []
    try:
        r = requests.get("https://api.elevenlabs.io/v1/voices",
                         headers={"xi-api-key": config.ELEVENLABS_API_KEY}, timeout=15)
        if r.status_code != 200:
            return []
        voices = r.json().get("voices", [])
    except Exception:
        return []
    # SEVRIN's own voice can't be one of the "other speakers"
    own = config.ELEVENLABS_VOICE_ID
    return [(v["voice_id"], v.get("name", "?")) for v in voices
            if v.get("voice_id") and v.get("voice_id") != own][:limit]


def _from_elevenlabs(verbose: bool):
    import requests

    voices = _elevenlabs_voices()
    if len(voices) < MIN_VOICES:
        return None
    if verbose:
        print(f"[cohort] building from {len(voices)} ElevenLabs voices "
              f"(one-time, uses ~{len(voices) * 90} characters)...")

    def synthesize(voice, sentence, path):
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice[0]}",
            headers={"xi-api-key": config.ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": sentence, "model_id": config.ELEVENLABS_MODEL},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        with open(path, "wb") as f:
            f.write(r.content)

    embeddings = _embed_synthesized(voices, synthesize, ".mp3", lambda v: v[1], verbose)
    return np.stack(embeddings) if len(embeddings) >= MIN_VOICES else None


def _system_voices(limit=60):
    """English macOS voices."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    voices = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("en"):
            voices.append(parts[0])
    return voices[:limit]


def _from_system_voices(verbose: bool):
    voices = _system_voices()
    if not voices:
        if verbose:
            print("[cohort] no system voices found; score normalisation disabled")
        return None
    if verbose:
        print(f"[cohort] building impostor cohort from {len(voices)} system voices (one-time, ~1 min)...")

    def synthesize(voice, sentence, path):
        subprocess.run(["say", "-v", voice, "-o", path, sentence],
                       check=True, capture_output=True, timeout=30)

    embeddings = _embed_synthesized(voices, synthesize, ".aiff", lambda v: f"voice {v}", verbose)
    if len(embeddings) < MIN_VOICES:
        if verbose:
            print(f"[cohort] only {len(embeddings)} usable voices, too few to normalise with; disabling")
        return None
    return np.stack(embeddings)


def build(force=False, verbose=True):
    """Build (or load the cached) cohort. Returns None if it can't be built."""
    if os.path.exists(COHORT_PATH) and not force:
        return np.load(COHORT_PATH)

    if shutil.which("ffmpeg") is None:
        if verbose:
            print("[cohort] ffmpeg not found, so the cohort can't be built and score")
            print("[cohort] normalisation is disabled. Install: brew install ffmpeg")
        return None

    try:
        arr = _from_elevenlabs(verbose)
    except Exception as e:
        if verbose:
            print(f"[cohort] ElevenLabs cohort failed ({e}); falling back to system voices")
        arr = None
    if arr is not None:
        _save(arr, "elevenlabs")
        if verbose:
            print(f"[cohort] built from {len(arr)} ElevenLabs voices -> {COHORT_PATH}")
        return arr

    if verbose:
        print("[cohort] falling back to macOS system voices (more synthetic, "
              "so normalisation is blunter but still effective)")
    arr = _from_system_voices(verbose)
    if arr is not None:
        _save(arr, "macos")
        if verbose:
            print(f"[cohort] built from {len(arr)} system voices -> {COHORT_PATH}")
    return arr
