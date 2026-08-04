"""
All the tunable knobs live here so you're not hunting through the pipeline
to change behavior.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys / backend selection ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TTS_BACKEND = os.getenv("TTS_BACKEND", "local")  # "local" or "elevenlabs"
LOCAL_VOICE_NAME = os.getenv("LOCAL_VOICE_NAME", "")  # e.g. "Ava (Premium)" — blank uses system default
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # fastest ElevenLabs model, slight quality trade vs turbo
# --- Wake word ---
# Real streaming detection via openWakeWord's pretrained "hey jarvis" model —
# it scores 80ms audio frames continuously as they arrive, so it fires the
# instant the phrase is heard instead of waiting for you to finish talking
# and running the whole clip through Whisper. Say "hey jarvis", not bare
# "jarvis" — the model was trained on the full phrase and misses more often
# without "hey".
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5   # 0-1, higher = fewer false wakes but more missed ones
VOICE_ID_CLIP_SECONDS = 1.5  # rolling audio buffer used for voice verification after a wake fires
# Used by voice_id/record.py to prompt you for enrollment clips — not used
# for wake matching anymore (openWakeWord handles that).
WAKE_PHRASES = ["hey jarvis"]

# --- Audio / VAD timing ---
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30          # webrtcvad requires 10/20/30ms frames
SILENCE_DURATION_MS = 900       # how long you must pause before we consider you "done"
MAX_UTTERANCE_SECONDS = 30      # hard cap so it can't listen forever
VAD_AGGRESSIVENESS = 2          # 0-3, higher = more aggressive at filtering non-speech
MIN_SPEECH_FRAMES = 4          # ~300ms of actual detected speech required before an utterance counts as real
PACKET_SENTENCE_LIMIT = 2       # sentences per "packet" — set high so most replies go out as one packet
VOICE_MATCH_THRESHOLD = 0.5  # 0-1 cosine similarity — lower = more lenient. Tune after testing.

# --- Conversation behavior ---
FOLLOWUP_WINDOW_SECONDS = 18    # how long after a reply you can keep talking without the wake word
MAX_HISTORY_TURNS = 20          # rolling conversation memory length (phase-1 scope: in-memory only)

# --- Claude ---
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = """You are a personal voice assistant, speaking out loud in a
real-time conversation. Keep replies short, natural, and conversational —
this is speech, not a written document. No markdown, no bullet lists, no
headers. Get to the point in one or two sentences unless the person clearly
wants more detail. Sound like a sharp, capable, slightly informal assistant —
professional but not stiff, the way a very competent person talks when
they're focused."""