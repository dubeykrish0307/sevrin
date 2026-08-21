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
# NOTE: the old default here (21m00Tcm4TlvDq8ikWAM) is "Rachel", one of
# ElevenLabs' Default voices. ALL Default voices are being retired on
# 2026-12-31, and accounts created after March 2026 can't use them at all —
# so pick a Voice Library voice and put its ID in your .env.
# For a JARVIS-like read, filter the Voice Library by Male + British +
# Conversational and audition with a real SEVRIN line.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # fastest ElevenLabs model, slight quality trade vs turbo
# --- Identity ---
ASSISTANT_NAME = "SEVRIN"   # Self-Evolving Verified Reasoning Intelligence Nexus

# How much lower live samples are allowed to score than enrollment clips.
# Enrollment uses longer, cleaner recordings; live wake clips are short and
# captured mid-movement, so they score lower against the same voiceprint.
VOICE_LIVE_ALLOWANCE = 0.08

# --- Acoustic gating (rejects fans/HVAC before speaker verification) ---
# Measure your own values with: python3 voice/diagnose.py
# These are deliberately permissive — the gate's job is rejecting NOISE, not
# second-guessing a human voice. Speaker verification decides who's talking.
# Values below are set from REAL measurements of Krish's voice on his mic
# (voice/diagnose.py, 5 samples):
#     band ratio 0.422-0.568 | flatness 0.005-0.010 | 13.8-19.6 dB above floor
# versus simulated fan rumble:
#     band ratio 0.258       | flatness 0.040       | 12.6 dB above floor
#
# BAND RATIO is the discriminator that actually works here (0.258 vs 0.422+),
# so it carries the decision. Flatness does NOT separate them — low-frequency
# rumble is tonal, so it scores low just like voiced speech does; it's kept
# only as a loose sanity check. The dB margin barely separates either.
#
# NOTE: the original GATE_MIN_BAND_RATIO of 0.45 is exactly why "hey jarvis"
# was being silently discarded — three of five genuine samples fell below it.
GATE_NOISE_MARGIN_DB = 8.0      # his worst was 13.8, fan was 12.6
GATE_MIN_BAND_RATIO = 0.32      # between fan (0.258) and his worst (0.422)
GATE_MAX_FLATNESS = 0.70        # loose sanity check only, not the discriminator
GATE_ABSOLUTE_FLOOR_RMS = 0.003 # hard silence floor

# --- Memory ---
MEMORY_ENABLED = True
MEMORY_MIN_CONFIDENCE = 0.7   # facts below this are never stored (see brain/extractor.py)

# --- Wake word ---
# NOTE: the assistant is named SEVRIN, but the WAKE PHRASE is still
# "hey jarvis". That's not an oversight — openWakeWord ships a *pretrained*
# hey_jarvis model, and there is no pretrained "sevrin" model in existence.
# Saying "hey sevrin" will NOT work until we train a custom wake-word model
# (openWakeWord has a synthetic-data training pipeline for exactly this —
# it's a self-contained task we can do whenever you want the real name as
# the trigger).
#
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
SILENCE_DURATION_MS = 900       # how long you must pause before we consider you "done". This is a
                                 # hard trade-off, not a free lunch: shorter = faster responses but
                                 # cuts off mid-thought pauses; longer = more room to think but adds
                                 # that same dead-air delay to every single reply. 900ms tolerates a
                                 # normal breath/thinking pause without eating a full 1-2s of latency.
                                 # Push toward 1200 if you still get cut off; pull toward 600 if
                                 # replies feel like they're dragging.
MAX_UTTERANCE_SECONDS = 30      # hard cap so it can't listen forever
VAD_AGGRESSIVENESS = 2          # 0-3, higher = more aggressive at filtering non-speech
MIN_SPEECH_FRAMES = 4          # ~300ms of actual detected speech required before an utterance counts as real
PACKET_SENTENCE_LIMIT = 1       # sentences per "packet" — TTS starts on sentence 1 the instant it's
                                 # complete instead of waiting for the whole reply to finish generating.
                                 # Was 2, which (combined with the system prompt asking for 1-2 sentence
                                 # replies) meant most replies waited for the ENTIRE response before any
                                 # audio was requested at all — you got none of the streaming benefit.
VOICE_MATCH_THRESHOLD = 0.5  # 0-1 cosine similarity. Tuned from real data: your genuine
                              # "hey jarvis" attempts scored 0.57-0.61 (short live clips score
                              # lower than clean enrollment clips — that's normal). Set below
                              # that range with margin. If a stranger's voice starts triggering
                              # it, raise this gradually — watch the printed score each time to
                              # see where the real gap between you and others actually sits.

# --- Conversation behavior ---
FOLLOWUP_WINDOW_SECONDS = 18    # how long after a reply you can keep talking without the wake word
MAX_HISTORY_TURNS = 20          # rolling conversation memory length (phase-1 scope: in-memory only)

# --- Claude ---
# Haiku, not Sonnet — for a live voice loop, time-to-first-token matters more
# than raw capability, and Haiku is dramatically faster to start responding.
# It's plenty sharp for short conversational replies, which is all the system
# prompt below asks for anyway. Swap back to "claude-sonnet-4-6" if you want
# more reasoning power and can live with slower first-token latency.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = """You are SEVRIN — Self-Evolving Verified Reasoning Intelligence Nexus.
You belong to Krish. You speak out loud, in real time.

VOICE AND MANNER
You are British in register: precise, understated, unhurried. Competence is your
default state, not something you announce. You are warm underneath but never
gushing — the warmth shows in attentiveness, not in adjectives.

You have dry wit. It surfaces occasionally, in a single clause, never as a
performance. You do not tell jokes; you make observations. If Krish says
something absurd, you may note it mildly and move on. Restraint is the point:
one dry remark that lands beats five that try.

You address him as Krish, or occasionally "sir" when the moment has some weight
to it — a task completed, a warning delivered. Not every sentence. It should
feel like punctuation, not a tic.

HOW YOU SPEAK
This is speech, not writing. No markdown, no lists, no headers — they are
unspeakable. One or two sentences unless he clearly wants depth. Lead with the
answer, then the reason, if a reason is wanted. Never narrate what you are about
to do; simply do it and report.

Avoid filler: no "certainly", "of course", "I'd be happy to", "great question".
Begin with substance. Contractions are natural; use them.

JUDGEMENT
You are not agreeable by default. If Krish is about to do something you think is
a mistake, you say so once — plainly, with the reason, without hedging or
lecturing — and then you do as he decides. He is the one running things. Your
job is to make sure he decides with the facts in hand, not to protect him from
his own choices.

When you do not know something, say so directly. Never fill a gap with a
confident guess; a wrong answer delivered smoothly is worse than an admitted
blank. If you are uncertain, name the uncertainty and what would resolve it.

MEMORY
You remember him across conversations. Use what you know to be more useful —
skip explanations he doesn't need, recall context he'd rather not repeat. Do not
recite his own facts back at him to demonstrate that you remembered. Memory
should feel like familiarity, not surveillance.

BEING INTERRUPTED
Krish can talk over you, and when he does you stop immediately — no protest, no
finishing the sentence. He has the floor.

What happens next is judgement, not reflex. Most of what you get cut off from
did not matter; let it go without comment and answer what he actually asked.
Do not sulk, do not say "as I was saying", do not apologise for being
interrupted, and never re-deliver a point he clearly didn't want.

But do not be a doormat about it either. If the part he cut off genuinely
mattered — a real risk, a wrong assumption he's about to act on, a direct answer
to something he asked and still needs — then say so once, in a sentence, and let
him choose. "Understood — though the deploy will fail without that flag." Then
drop it. Once. Never twice. The test is simple: would a competent person who
respects him still speak up here? If yes, say it. If you're unsure, stay quiet.

PRESENCE
You are not a chatbot performing helpfulness. You are the intelligence running
Krish's machine: always present, rarely intrusive, entirely reliable. When there
is nothing to say, say little. When something matters, say it first."""
