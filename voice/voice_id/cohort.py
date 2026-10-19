"""
Impostor cohort for score normalisation.

WHY THIS EXISTS
A raw cosine similarity is not comparable across conditions. Enrol on the
built-in mic, then speak through AirPods, and every score shifts downward —
not because the speaker changed, but because the channel did. A fixed
threshold then rejects the right person. That's the failure mode behind
"it worked yesterday and today it says I'm someone else".

AS-norm (adaptive score normalisation) fixes this by measuring the score
RELATIVE to how the same audio scores against a set of other speakers. When
the channel shifts, the impostor scores shift with it, so the normalised
value stays put. It's the standard technique in speaker verification for
exactly this problem.

THE COHORT
AS-norm needs recordings of other speakers. Two sources, in order of quality:

  1. ElevenLabs voices (preferred). Close enough to real human speech that
     their embeddings land in roughly the same region as actual people,
     which is what makes the normalisation accurate. One-time cost of a few
     thousand characters.

  2. macOS `say` voices (fallback). Free, offline, instant — but audibly
     synthetic, so their embeddings cluster somewhere less representative of
     real speakers. Still useful for cancelling channel effects, just a
     blunter reference distribution.

Either beats no cohort at all, which leaves raw scores that break whenever
the microphone changes.
"""
import os
import subprocess
import tempfile

import numpy as np

COHORT_PATH = os.path.join(os.path.dirname(__file__), "profile", "cohort.npy")
COHORT_META = os.path.join(os.path.dirname(__file__), "profile", "cohort_source.txt")


def _record_source(name: str):
    try:
        os.makedirs(os.path.dirname(COHORT_META), exist_ok=True)
        with open(COHORT_META, "w") as f:
            f.write(name)
    except Exception:
        pass


def source():
    """Which voices the cohort came from — ElevenLabs is the better one."""
    try:
        with open(COHORT_META) as f:
            return f.read().strip()
    except Exception:
        return "unknown"

# Varied phonetic content so embeddings aren't dominated by one phrase.
_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "She sells seashells by the seashore on a bright summer morning.",
    "Technology moves faster than most people can reasonably follow.",
    "Please remind me about the meeting scheduled for tomorrow afternoon.",
    "Numbers like seventeen, forty two and ninety nine appear frequently.",
]


def _available_voices(limit=60):
    """macOS voices, preferring English ones for closer phonetic match."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    voices = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, locale = parts[0], parts[1]
        if locale.startswith("en"):
            voices.append(name)
    return voices[:limit]


def _elevenlabs_voices(limit=60):
    """Fetch distinct voices from the ElevenLabs library."""
    import requests
    import config
    if not getattr(config, "ELEVENLABS_API_KEY", None):
        return []
    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": config.ELEVENLABS_API_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        voices = r.json().get("voices", [])
    except Exception:
        return []

    # Exclude SEVRIN's own voice — it must not be in the impostor set, or
    # normalisation would treat his own output as a reference "other speaker".
    own = getattr(config, "ELEVENLABS_VOICE_ID", None)
    out = [(v.get("voice_id"), v.get("name", "?"))
           for v in voices if v.get("voice_id") and v.get("voice_id") != own]
    return out[:limit]


def _build_from_elevenlabs(verbose=True):
    import requests
    import config
    from voice.voice_id import encoder as enc

    voices = _elevenlabs_voices()
    if len(voices) < 5:
        return None

    if verbose:
        print(f"[cohort] building from {len(voices)} ElevenLabs voices "
              f"(one-time, uses ~{len(voices) * 90} characters)...")

    embeddings = []
    tmpdir = tempfile.mkdtemp(prefix="sevrin_cohort_el_")
    try:
        for i, (vid, name) in enumerate(voices):
            sentence = _SENTENCES[i % len(_SENTENCES)]
            mp3 = os.path.join(tmpdir, f"{i}.mp3")
            wav = os.path.join(tmpdir, f"{i}.wav")
            try:
                r = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
                    headers={"xi-api-key": config.ELEVENLABS_API_KEY,
                             "Content-Type": "application/json"},
                    json={"text": sentence,
                          "model_id": getattr(config, "ELEVENLABS_MODEL", "eleven_flash_v2_5")},
                    timeout=30,
                )
                if r.status_code != 200:
                    if verbose:
                        print(f"[cohort]   skipped {name} (HTTP {r.status_code})")
                    continue
                with open(mp3, "wb") as f:
                    f.write(r.content)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                    check=True, capture_output=True, timeout=30,
                )
                emb = enc.embed_file(wav)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                if verbose:
                    print(f"[cohort]   skipped {name} ({e})")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if len(embeddings) < 5:
        return None
    return np.stack(embeddings)


def build(force=False, verbose=True):
    """Generate and embed the cohort. Cached — this takes a minute."""
    if os.path.exists(COHORT_PATH) and not force:
        return np.load(COHORT_PATH)

    from voice.voice_id import encoder as enc

    # Both paths convert audio via ffmpeg, so check once up front rather than
    # discovering it after spending API credits.
    import shutil
    if shutil.which("ffmpeg") is None:
        if verbose:
            print("[cohort] ffmpeg not found — cannot build cohort, so score")
            print("[cohort] normalisation is disabled. Install: brew install ffmpeg")
        return None

    # Prefer ElevenLabs: much closer to real human voices, so the reference
    # distribution better matches the speakers we actually need to tell apart.
    try:
        arr = _build_from_elevenlabs(verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"[cohort] ElevenLabs cohort failed ({e}); falling back to system voices")
        arr = None

    if arr is not None:
        os.makedirs(os.path.dirname(COHORT_PATH), exist_ok=True)
        np.save(COHORT_PATH, arr)
        _record_source("elevenlabs")
        if verbose:
            print(f"[cohort] built from {len(arr)} ElevenLabs voices -> {COHORT_PATH}")
        return arr

    if verbose:
        print("[cohort] falling back to macOS system voices (more synthetic, "
              "so normalisation is blunter but still effective)")

    voices = _available_voices()
    if not voices:
        if verbose:
            print("[cohort] no system voices found; score normalisation disabled")
        return None

    if verbose:
        print(f"[cohort] building impostor cohort from {len(voices)} system voices "
              f"(one-time, ~1 min)...")

    embeddings = []
    tmpdir = tempfile.mkdtemp(prefix="sevrin_cohort_")
    try:
        for i, voice in enumerate(voices):
            sentence = _SENTENCES[i % len(_SENTENCES)]
            aiff = os.path.join(tmpdir, f"{i}.aiff")
            wav = os.path.join(tmpdir, f"{i}.wav")
            try:
                subprocess.run(["say", "-v", voice, "-o", aiff, sentence],
                               check=True, capture_output=True, timeout=30)
                # convert to 16k mono wav for the encoder
                subprocess.run(
                    ["ffmpeg", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", wav],
                    check=True, capture_output=True, timeout=30,
                )
                emb = enc.embed_file(wav)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                if verbose:
                    print(f"[cohort]   skipped voice {voice} ({e})")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if len(embeddings) < 5:
        if verbose:
            print(f"[cohort] only {len(embeddings)} usable voices — too few to "
                  f"normalise with; disabling")
        return None

    arr = np.stack(embeddings)
    os.makedirs(os.path.dirname(COHORT_PATH), exist_ok=True)
    np.save(COHORT_PATH, arr)
    _record_source("macos")
    if verbose:
        print(f"[cohort] built from {len(arr)} system voices -> {COHORT_PATH}")
    return arr


def load():
    if os.path.exists(COHORT_PATH):
        try:
            return np.load(COHORT_PATH)
        except Exception:
            return None
    return None
