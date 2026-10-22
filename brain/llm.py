"""Replies from Claude for the voice loop, plus the short calls behind
proactive speech.

Replies stream in "packets" of PACKET_SENTENCE_LIMIT sentences so TTS can
start on the first sentence while the rest is still generating. History and
verified facts come from SQLite (brain/store.py).
"""
import os
import re
import threading

import config
from brain import extractor, prompts, store
from brain.claude import ask, client, parse_json

# The backend sets this so memory events can be shown in the UI.
on_memory_event = None

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "krish_profile.md"
)
_profile = None


def _split_sentences(buffer: str):
    """Returns (finished sentences, leftover text)."""
    parts = _SENTENCE_END.split(buffer)
    return parts[:-1], parts[-1]


def _core_profile() -> str:
    """Hand-written facts from data/krish_profile.md. These are trusted as-is,
    unlike extracted facts. Read once; restart the backend after editing."""
    global _profile
    if _profile is None:
        try:
            with open(_PROFILE_PATH, encoding="utf-8") as f:
                text = f.read().strip()
        except (OSError, ValueError):
            text = ""
        _profile = f"\n\n{text}" if text else ""
    return _profile


def _memory_context() -> str:
    """Verified facts for the system prompt, grouped by subject and capped so
    memory can't crowd out the conversation."""
    facts = store.active_facts(limit=60)
    if not facts:
        return ""
    by_subject = {}
    for f in facts:
        by_subject.setdefault(f["subject"], []).append(f["fact"])
    lines = []
    for subject, items in by_subject.items():
        lines.append(f"{subject}:")
        lines.extend(f"  - {item}" for item in items[:12])
    return (
        "\n\nWhat you know about Krish (verified memory — treat as background "
        "knowledge, don't recite it back at him unprompted):\n" + "\n".join(lines)
    )


def _interruption_context(intr: dict) -> str:
    spoken = (intr.get("spoken") or "").strip()
    unspoken = (intr.get("unspoken") or "").strip()

    block = [
        "\n\n--- YOU WERE JUST INTERRUPTED ---",
        "Krish cut you off mid-reply. This is the state of it:",
        f'  What you had already said out loud: "{spoken}"' if spoken
        else "  You had barely started speaking.",
        f'  What you had NOT yet said: "{unspoken}"' if unspoken
        else "  You had essentially finished your point.",
        prompts.INTERRUPTED,
    ]
    return "\n".join(block)


def ask_streaming_packets(user_text: str, interruption: dict = None, extra_context: str = None):
    """Stream a reply to user_text, yielding packets of text.

    interruption: if the previous reply was cut off, what was said and what
    wasn't. The model decides whether the unsaid part still matters.
    extra_context: appended to the system prompt for this turn only.
    """
    turn_id = store.add_turn("user", user_text)
    history = store.recent_turns(limit=config.MAX_HISTORY_TURNS)

    system = prompts.SYSTEM_PROMPT + _core_profile() + _memory_context()
    if interruption:
        system += _interruption_context(interruption)
    if extra_context:
        system += extra_context

    buffer = ""
    reply = ""
    pending = []
    with client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            buffer += text
            reply += text
            done, buffer = _split_sentences(buffer)
            pending.extend(s.strip() for s in done if s.strip())
            if len(pending) >= config.PACKET_SENTENCE_LIMIT:
                yield " ".join(pending)
                pending = []

    if buffer.strip():
        pending.append(buffer.strip())
    if pending:
        yield " ".join(pending)

    store.add_turn("assistant", reply)

    # fact extraction makes several API calls; keep it off the reply path
    threading.Thread(target=_extract_in_background, args=(user_text, turn_id), daemon=True).start()


def _extract_in_background(user_text: str, turn_id: int):
    try:
        events = extractor.process_turn(user_text, source_turn_id=turn_id, on_event=on_memory_event)
    except Exception as e:
        print(f"[memory] extraction failed: {e}")
        return
    for ev in events:
        if ev.get("outcome") in ("stored", "updated"):
            print(f"[memory] {ev['outcome']}: {ev.get('fact')} ({ev.get('detail')})")
        elif ev.get("outcome") == "rejected":
            print(f"[memory] rejected at {ev.get('stage')}: {ev.get('fact')} ({ev.get('detail')})")


# --- proactive speech ---------------------------------------------------------

def judge_proactive(item, situation: dict) -> bool:
    """The model's call on an ambiguous proactive item (see backend/proactive.py)."""
    prompt = (
        f"ITEM\n"
        f"  kind    : {item.kind}\n"
        f"  urgency : {item.urgency}\n"
        f"  content : {item.summary}\n\n"
        f"SITUATION\n"
        + "\n".join(f"  {k}: {v}" for k, v in situation.items())
    )
    data = parse_json(ask(prompts.PROACTIVE_JUDGE, prompt, max_tokens=150))
    if not isinstance(data, dict):
        return False    # unparseable, stay quiet
    decision = bool(data.get("speak"))
    print(f"[proactive] judge: {'SPEAK' if decision else 'wait'} ({data.get('reason', '')})")
    return decision


def phrase_attention_check(item) -> str:
    """Short "you there?" line said before an important proactive item."""
    try:
        line = ask(
            prompts.ATTENTION_CHECK + _core_profile(),
            f"You're about to tell him about a {item.kind}.",
            model=config.CLAUDE_MODEL,
            max_tokens=60,
        ).strip()
        return line or "You there?"
    except Exception as e:
        print(f"[proactive] attention phrasing failed ({e}); using fallback")
        return "You there?"


def phrase_proactive(item) -> str:
    """Let SEVRIN word the item himself so it doesn't sound like a notification."""
    try:
        line = ask(
            prompts.PROACTIVE_LINE + _core_profile() + _memory_context(),
            f"Facts to convey ({item.kind}): {item.summary}",
            model=config.CLAUDE_MODEL,
            max_tokens=120,
        ).strip()
        return line or item.summary
    except Exception as e:
        print(f"[proactive] phrasing failed ({e}); using raw summary")
        return item.summary


def record_proactive(line: str):
    """Unprompted lines go into history too, so he doesn't repeat himself."""
    store.add_turn("assistant", line)
