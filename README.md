# SEVRIN

**A real-time voice assistant that checks it's my voice, remembers only what it can verify, and knows when to stay quiet.**

SEVRIN (Self-Evolving Verified Reasoning Intelligence Nexus) runs on my MacBook. A Python asyncio backend runs the voice loop, and an Electron app shows the conversation. Speech-to-text and speaker verification run locally; Claude does the reasoning.

> Demo video: coming soon.

---

## What it does

| Part | How it works |
|---|---|
| Wake word | openWakeWord, streaming on 80 ms frames (pretrained "hey jarvis" model; a custom "hey sevrin" model is planned) |
| Noise gate | Rejects fans and room hum before anything else runs, using a speech-band energy ratio |
| Speaker verification | ECAPA-TDNN embeddings (SpeechBrain) with AS-norm score normalisation against an impostor cohort, so a new mic or room doesn't break the threshold |
| Speech-to-text | faster-whisper by default; optional resident whisper.cpp server (large-v3-turbo, Metal on Apple Silicon) with automatic fallback |
| Reasoning | Claude (Haiku 4.5, chosen for time-to-first-token in a live voice loop) |
| Memory | SQLite conversations + a separate facts table. Every fact passes layered checks before it's stored (see below) |
| Speaking | Sentence-by-sentence streaming TTS (macOS `say` or ElevenLabs), so audio starts after the first sentence |
| Barge-in | You can interrupt it while it's speaking |
| Proactive speech | Reminders and alerts go through a queue that decides *when* to speak (see below) |
| Desktop app | Electron + three.js orb, chat panel and text input over a local WebSocket |

## Design decisions

**Memory that can say no.** A memory system that confidently stores wrong things about you is worse than none. A candidate fact must survive:
1. a cheap filter (greetings and small talk never reach the model),
2. extraction with verbatim evidence,
3. a separate grounding check against the actual transcript,
4. a contradiction check against stored facts (update or reject),
5. a confidence threshold.

**Proactive speech: rules first, the LLM only for real dilemmas.** Expired items are dropped. Deterministic rules handle the clear cases (I'm mid-sentence: wait; nothing is happening: speak). Only the ambiguous middle reaches the model, which keeps latency and cost down and the decisions predictable.

**Measure, then set thresholds.** Thresholds in `config.py` come from recordings, not guesses. Two examples:
- The noise gate: my speech measured a band ratio of 0.44–0.66 against 0.24–0.29 for fan rumble, so the cut-off sits at 0.30.
- Speaker verification: a threshold set from a *simulated* impostor cohort let friends' voices through in real use. `tools/calibrate_impostor.py` measures it from recordings of real people instead.

## What I'm measuring next

- [ ] Speech-to-text benchmark: word error rate and latency, faster-whisper vs whisper.cpp, on ~50 of my own commands
- [ ] Speaker verification: false-accept and false-reject rates with 3–5 other people
- [ ] 90-second demo video
- [ ] Tests and CI

## Architecture

```
mic ─► noise gate ─► wake word ─► speaker verification ─► speech-to-text
                                                              │
                     ┌──────── memory (SQLite) ◄──────────────┤
                     ▼                                        ▼
            proactive queue ─────────────────────────► Claude reasoning
                                                              │
speaker ◄── streaming TTS ◄──────────── sentence packets ◄────┘
   ▲
   └── barge-in: speaking stops when I start talking

backend/service.py  ◄── WebSocket ws://localhost:8765 ──►  ui/ (Electron)
```

## Project layout

```
config.py        every tunable threshold, with the measurement behind it
main.py          terminal-only entry point (voice, no app), prints a latency breakdown per turn
backend/         WebSocket service, proactive speech queue, reminder scheduler, time parsing
brain/           replies, prompts, fact extraction and verification, SQLite store
voice/           wake word, noise gate, listener, STT (faster-whisper / whisper.cpp), TTS, barge-in
voice/voice_id/  enrolment, ECAPA encoder, verification, impostor cohort
ui/              Electron app: renderer.js (wiring), hud.js (panels), reactor.js (the orb)
tools/           mic diagnostics and calibration scripts (see below)
```

## Tools

These all use the real microphone and print every number they measure.

| Script | What it's for |
|---|---|
| `tools/diagnose.py` | Noise-gate and speaker scores for my voice on my mic; suggests `config.py` values |
| `tools/check_wake.py` | Runs the real wake path end to end. Run after any change under `voice/` |
| `tools/calibrate_impostor.py` | Records me and another person and suggests a speaker threshold between us |
| `tools/bench_stt.py` | whisper.cpp vs faster-whisper, text and timing, on my own sentences |

Lint with `ruff check .` (config in `pyproject.toml`).

## Privacy

Voice recordings, the enrolled voiceprint, session logs and the memory database stay on the machine. They're in `.gitignore` and never committed.

## Run it

Requires macOS, Python 3.11, Node.js and an Anthropic API key.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY

# enrol your voice first
python3 voice/voice_id/record.py
python3 voice/voice_id/enroll.py

# terminal 1: backend
python3 backend/service.py

# terminal 2: app
cd ui && npm install && npm start
```

Say "hey jarvis" or type into the text bar. Both go through the same pipeline.

### Optional: whisper.cpp for better speech recognition

faster-whisper has no Metal backend on macOS, so it runs on CPU there. whisper.cpp compiles with Metal on Apple Silicon, which makes large-v3-turbo fast enough for live commands.

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp && make -j
./models/download-ggml-model.sh large-v3-turbo
```

Add `WHISPER_CPP_DIR=/full/path/to/whisper.cpp` to `.env`, then compare both engines on your own machine:

```bash
python3 tools/bench_stt.py
```

If whisper.cpp isn't configured or fails to start, SEVRIN falls back to faster-whisper automatically.
