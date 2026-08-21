#!/usr/bin/env bash
# Run from your repo root: bash restructure.sh
# Physically moves files into the new voice/brain/backend/connections/ui
# layout, preserving git history via `git mv`. Does NOT touch content —
# you'll paste in the updated file contents separately (Claude gave you
# those in full).
set -e

mkdir -p voice brain backend connections ui data

git mv wake_word.py voice/wake_word.py
git mv listener.py voice/listener.py
git mv stt.py voice/stt.py
git mv tts.py voice/tts.py
git mv voice_id voice/voice_id
touch voice/__init__.py
git add voice/__init__.py

git mv brain.py brain/llm.py
touch brain/__init__.py
git add brain/__init__.py

touch backend/__init__.py connections/__init__.py
git add backend/__init__.py connections/__init__.py

touch data/.gitkeep
git add data/.gitkeep

echo "Done. Structure moved. Now:"
echo "  1. Paste in the updated wake_word.py, main.py, voice/voice_id/record.py,"
echo "     voice/voice_id/enroll.py, README.md, .gitignore (Claude gave you full contents)."
echo "  2. Create brain/memory.py, connections/README.md, backend/README.md, ui/README.md"
echo "     (also given in full)."
echo "  3. git add -A && git commit -m 'restructure into voice/brain/backend/connections/ui'"
