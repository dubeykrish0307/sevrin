# JARVIS

## Layout

```
config.py          shared config — every module imports from here
main.py             terminal entrypoint (still works, voice-only, no UI)
voice/              wake word, mic recording, STT, TTS, speaker verification
brain/              the LLM client (llm.py); memory.py not built yet
connections/        not built yet — computer control, internet, reminders, other tools
backend/            service.py — WebSocket server that runs the voice loop and
                    talks to the desktop app
ui/                 Electron desktop app — orb, chat panel, text input
data/               gitignored — runtime state (logs, memory db, etc.)
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and ELEVENLABS_API_KEY if using that backend)
```

## Enroll your voice (do this before running anything else)

```bash
python3 voice/voice_id/record.py
python3 voice/voice_id/enroll.py
```

## Run — with the app (recommended)

Two processes, both need to be running:

```bash
# terminal 1 — the backend
source venv/bin/activate
python3 backend/service.py
```

```bash
# terminal 2 — the app (first time only: cd ui && npm install)
cd ui
npm install   # first time only
npm start
```

The app connects to the backend over `ws://localhost:8765`. The status
dot in the top-right turns green once connected. You can talk ("hey
jarvis...") or type into the text bar — both go through the same
pipeline and both talk back out loud.

Two terminal windows is not the end state — Phase 2+ moves the backend to
a proper background service (launchd) so this becomes one double-click.
For now, both need to stay open.

## Run — terminal only (no app)

```bash
python3 main.py
```

Still works exactly as before, if you just want to test voice without the app.

## Tuning

All the knobs are in `config.py`, with comments.

## Note on the orb's "speaking" animation

It does not decode real audio to react to the literal waveform — that
would need MP3 decoding and precise sync with `ffplay`'s playback buffer.
Instead it estimates how long a reply will take to speak from its text
length and animates a natural-feeling pulse for roughly that duration.
Looks alive and responsive to speech; isn't sample-accurate. See the note
in `backend/service.py` if you want to upgrade this later.

## Optional: whisper.cpp for better speech recognition

faster-whisper (the default) has no Metal backend on macOS, so it runs on
CPU only there — which is why the default model is kept small. whisper.cpp
compiles with Metal support automatically on Apple Silicon, which makes a
genuinely heavier, more accurate model (large-v3-turbo by default — near
full large-v3 accuracy, built specifically to stay fast) practical for
real-time voice commands.

**Setup:**
```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
make -j
./models/download-ggml-model.sh large-v3-turbo
```

Then in your `.env`:
```
WHISPER_CPP_DIR=/full/path/to/whisper.cpp
```

**Test it before relying on it:**
```bash
python3 voice/test_whispercpp.py
```

This starts the server, transcribes a few real utterances, and shows you
whisper.cpp and faster-whisper side by side — both timing and actual
transcribed text — measured on your machine, not a benchmark from someone
else's. If it fails to start, it tells you exactly why.

**If it's not configured, or fails to start, nothing breaks** — SEVRIN
automatically falls back to faster-whisper, unchanged.

**Optional extra speedup (Core ML / Neural Engine):** requires Xcode
command line tools and the `coremltools` Python package. Not required —
Metal alone already gives most of the benefit — but if you want more:
```bash
./models/generate-coreml-model.sh large-v3-turbo
```
