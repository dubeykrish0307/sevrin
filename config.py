"""SEVRIN settings. Keys and machine-specific paths come from .env."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- keys and voices ----------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

TTS_BACKEND = os.getenv("TTS_BACKEND", "local")        # "local" (macOS say) or "elevenlabs"
LOCAL_VOICE_NAME = os.getenv("LOCAL_VOICE_NAME", "")   # e.g. "Ava (Premium)", blank = system default
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Default is "Rachel". ElevenLabs is retiring its default voices at the end of
# 2026, so put a Voice Library voice ID in .env.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = "eleven_flash_v2_5"

# --- Claude -------------------------------------------------------------------
# Haiku for replies: in a live voice loop time-to-first-token matters more
# than depth. Swap to a Sonnet model if you want more reasoning.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
# memory checks and the proactive judge are small classification calls
VERIFY_MODEL = "claude-haiku-4-5-20251001"

MAX_HISTORY_TURNS = 20
PACKET_SENTENCE_LIMIT = 1       # sentences per TTS packet; 1 = speak as soon as the first sentence is done
MEMORY_MIN_CONFIDENCE = 0.7     # facts below this are never stored

# --- speech to text -------------------------------------------------------------
# faster-whisper has no Metal backend, so on a Mac it runs on the CPU.
# small.en is the most accurate model that doesn't hurt latency there.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")

# Optional whisper.cpp server (Metal on Apple Silicon). Leave WHISPER_CPP_DIR
# unset to use faster-whisper; if the server fails we fall back to it anyway.
WHISPER_CPP_DIR = os.getenv("WHISPER_CPP_DIR") or None
WHISPER_CPP_MODEL = os.getenv("WHISPER_CPP_MODEL", "large-v3-turbo")
WHISPER_CPP_PORT = int(os.getenv("WHISPER_CPP_PORT", "8178"))
# only needed if the binary/model aren't in the usual places under WHISPER_CPP_DIR
WHISPER_CPP_SERVER_BINARY = os.getenv("WHISPER_CPP_SERVER_BINARY") or None
WHISPER_CPP_MODEL_PATH = os.getenv("WHISPER_CPP_MODEL_PATH") or None

# --- audio and VAD --------------------------------------------------------------
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30          # webrtcvad only takes 10, 20 or 30 ms frames
VAD_AGGRESSIVENESS = 2          # 0-3
MIN_SPEECH_FRAMES = 4           # speech frames needed before an utterance counts
# Pause that ends an utterance. Shorter feels snappier but cuts off thinking
# pauses; 900 ms tolerates a normal breath.
SILENCE_DURATION_MS = 900
MAX_UTTERANCE_SECONDS = 30
FOLLOWUP_WINDOW_SECONDS = 30    # keep talking after a reply without the wake word

# --- noise gate (voice/audio_gate.py) -------------------------------------------
# Measured on my mic with tools/diagnose.py. Band ratio is what separates
# speech from the fan: my voice 0.44-0.66, fan rumble 0.24-0.29. Flatness
# doesn't separate them (rumble is tonal too), so it's only a loose check.
GATE_NOISE_MARGIN_DB = 8.0      # my quietest speech was 13.8 dB above the floor
GATE_MIN_BAND_RATIO = 0.30
GATE_MAX_FLATNESS = 0.70
GATE_ABSOLUTE_FLOOR_RMS = 0.0004  # only catches true digital silence

# --- wake word ------------------------------------------------------------------
# openWakeWord ships a pretrained "hey jarvis" model. A "hey sevrin" model
# needs a custom training run.
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5       # 0-1, higher = fewer false wakes, more misses
WAKE_PHRASES = ["hey jarvis"]   # what voice/voice_id/record.py asks you to say

# --- speaker verification -------------------------------------------------------
# Every embedding (live and enrolment) is taken over a window this long, so
# the two are comparable.
VOICE_ID_CLIP_SECONDS = 2.5

# With an impostor cohort the decision uses the AS-norm z-score.
# TODO: 3.0 came from a simulated cohort and lets friends through. Replace it
# with what tools/calibrate_impostor.py recommends.
VOICE_ZNORM_THRESHOLD = 3.0

# Without a cohort: raw cosine against calibration.json's threshold, lowered
# by this much because short live clips score below the enrolment clips.
VOICE_LIVE_ALLOWANCE = 0.10
VOICE_MATCH_THRESHOLD = 0.40    # only if calibration.json is missing
