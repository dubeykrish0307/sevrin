"""
SEVRIN — the reasoning layer.

Streams from Claude, batching sentences into "packets" of up to
PACKET_SENTENCE_LIMIT so speech starts as early as possible.

Now memory-backed:
  - conversation history is loaded from and written to SQLite (brain/store.py),
    so SEVRIN no longer forgets everything when the backend restarts.
  - verified facts about Krish are injected into the system prompt, so it
    actually knows him rather than re-learning every session.
  - after each turn, fact extraction runs in a BACKGROUND thread — it makes
    several verification API calls, and doing that inline would add seconds
    of latency to every reply. Memory formation must never slow speech.
"""
import os
import re
import threading

from anthropic import Anthropic

import config
from brain import store, extractor

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Optional hook the backend sets so memory events can reach the UI.
on_memory_event = None


def _split_sentences(buffer: str):
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    complete = parts[:-1]
    remainder = parts[-1] if parts else ""
    return complete, remainder


_CORE_PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "krish_profile.md"
)
_core_profile_cache = None


def _core_profile() -> str:
    """The static, hand-authored facts about Krish (data/krish_profile.md).
    Distinct from the dynamic, extracted-and-verified facts in
    _memory_context(): this is trusted because he wrote it directly, not
    because it survived the extraction pipeline. Cached in memory since the
    file rarely changes and there's no reason to hit disk every turn — edit
    the file and restart to pick up changes."""
    global _core_profile_cache
    if _core_profile_cache is None:
        try:
            with open(_CORE_PROFILE_PATH, encoding="utf-8") as f:
                content = f.read().strip()
            _core_profile_cache = f"\n\n{content}" if content else ""
        except Exception:
            _core_profile_cache = ""
    return _core_profile_cache


def _memory_context() -> str:
    """Renders verified facts for the system prompt. Only high-confidence
    active facts, grouped by subject, capped so a long memory never crowds
    out the actual conversation."""
    facts = store.active_facts(limit=60)
    if not facts:
        return ""
    by_subject = {}
    for f in facts:
        by_subject.setdefault(f["subject"], []).append(f["fact"])
    lines = []
    for subject, items in by_subject.items():
        lines.append(f"{subject}:")
        for it in items[:12]:
            lines.append(f"  - {it}")
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
        f"  What you had already said out loud: \"{spoken}\"" if spoken else
        "  You had barely started speaking.",
    ]
    if unspoken:
        block.append(f"  What you had NOT yet said: \"{unspoken}\"")
    else:
        block.append("  You had essentially finished your point.")
    block.append(
        "Handle this the way a composed person would: he has the floor now, so "
        "respond to what he actually said. Judge for yourself whether the "
        "unfinished part still matters — most of the time it doesn't and you "
        "simply drop it. If it genuinely does (a real risk, a wrong assumption "
        "he's about to act on, something he asked for and still needs), say so "
        "once, briefly, and let him decide. Do not announce that you were "
        "interrupted, do not apologise for it, and do not resume where you left "
        "off unless he asks."
    )
    return "\n".join(block)


def ask_streaming_packets(user_text: str, interruption: dict = None,
                          extra_context: str = None):
    """interruption, when present, describes a reply of SEVRIN's that Krish
    cut off: what he'd managed to say, what he hadn't reached yet, and what
    Krish said over him. It's handed to the model as context so SEVRIN can
    JUDGE what to do about it — comply and drop it, comply but flag that
    something mattered, or answer the new thing while keeping the unfinished
    point alive. Deliberately not rule-based; see SYSTEM_PROMPT."""
    turn_id = store.add_turn("user", user_text)

    # history now comes from disk, not a module-level list
    history = store.recent_turns(limit=config.MAX_HISTORY_TURNS)

    system_prompt = config.SYSTEM_PROMPT + _core_profile() + _memory_context()

    if interruption:
        system_prompt += _interruption_context(interruption)
    if extra_context:
        system_prompt += extra_context

    buffer = ""
    full_reply = ""
    pending = []

    with _client.messages.stream(
        model=config.CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            buffer += text
            full_reply += text
            complete, buffer = _split_sentences(buffer)
            for sentence in complete:
                if sentence.strip():
                    pending.append(sentence.strip())
            if len(pending) >= config.PACKET_SENTENCE_LIMIT:
                yield " ".join(pending)
                pending = []

    if buffer.strip():
        pending.append(buffer.strip())
    if pending:
        yield " ".join(pending)

    store.add_turn("assistant", full_reply)

    # Fact extraction runs off the hot path — several verification calls
    # would otherwise stall the next reply.
    threading.Thread(
        target=_extract_in_background, args=(user_text, turn_id), daemon=True
    ).start()


def _extract_in_background(user_text: str, turn_id: int):
    try:
        events = extractor.process_turn(user_text, source_turn_id=turn_id, on_event=on_memory_event)
        for ev in events:
            if ev.get("outcome") in ("stored", "updated"):
                print(f"[memory] {ev['outcome']}: {ev.get('fact')} ({ev.get('detail')})")
            elif ev.get("outcome") == "rejected":
                print(f"[memory] rejected at {ev.get('stage')}: {ev.get('fact')} — {ev.get('detail')}")
    except Exception as e:
        print(f"[memory] extraction failed: {e}")


# ---------------------------------------------------------------------------
# Proactive speech
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You decide whether an assistant should speak up right now,
unprompted, or keep waiting.

The assistant only reaches you for genuinely ambiguous cases — obvious ones
(he's mid-sentence, the item is urgent and the user is free) are already
handled mechanically before you see them. So assume the situation is a real
judgement call.

Weigh:
- How much the user would want this NOW versus in a few minutes.
- Whether the moment looks like a natural pause or a brief gap in something
  ongoing. A few seconds of silence usually means someone is still thinking,
  not that they've finished.
- How long the item has waited. Something repeatedly deferred should
  eventually be said or it's useless.
- Whether it's still actionable. Information the user can't act on is noise.

Default to waiting. Silence costs a short delay; a badly timed interruption
costs trust. But do not defer indefinitely — if an item has been considered
several times and still matters, say it.

Answer ONLY with JSON: {"speak": true|false, "reason": "<one short sentence>"}"""


def judge_proactive(item, situation: dict) -> bool:
    """Layer 3 of the proactive decision — see backend/proactive.py."""
    prompt = (
        f"ITEM\n"
        f"  kind    : {item.kind}\n"
        f"  urgency : {item.urgency}\n"
        f"  content : {item.summary}\n\n"
        f"SITUATION\n"
        + "\n".join(f"  {k}: {v}" for k, v in situation.items())
    )
    resp = _client.messages.create(
        model=extractor.VERIFY_MODEL,   # fast model; this is a narrow decision
        max_tokens=150,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data = extractor._json_from(text)
    if not isinstance(data, dict):
        return False        # unparseable -> stay quiet
    decision = bool(data.get("speak"))
    print(f"[proactive] judge: {'SPEAK' if decision else 'wait'} — {data.get('reason','')}")
    return decision


_ATTENTION_SYSTEM = """You are SEVRIN, about to interrupt Krish's attention
unprompted for something that needs a real response from him — not just
background chatter. Before saying what it's about, say ONE short line that
checks he's actually there and paying attention, in character: warm, direct,
maybe a little dry. The way a person checks in before launching into
something, not a notification chime.

Do NOT reveal what the actual matter is yet — that comes after he responds.
Vary your phrasing; don't say the exact same check-in every time.

Output only the line itself."""


def phrase_attention_check(item) -> str:
    """The line SEVRIN says to get his attention BEFORE delivering a
    time-sensitive proactive item — see backend/service.py's
    _speak_proactively for why this two-step exists."""
    try:
        resp = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=60,
            system=_ATTENTION_SYSTEM + _core_profile(),
            messages=[{"role": "user", "content":
                       f"You're about to tell him about a {item.kind}."}],
        )
        line = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        return line or "You there?"
    except Exception as e:
        print(f"[proactive] attention phrasing failed ({e}); using fallback")
        return "You there?"


_PHRASE_SYSTEM = """You are SEVRIN, speaking to Krish unprompted — he did not
ask you anything; you are the one starting this.

Turn the facts you're given into ONE short spoken line, in character:
understated, precise, no preamble. Lead with the thing that matters.

Because you're interrupting his attention unasked, be briefer than usual.
No "just letting you know", no "I wanted to mention", no apologising for
speaking. State it and stop.

Output only the line itself."""


def phrase_proactive(item) -> str:
    """SEVRIN phrases the item himself rather than reading a canned string,
    so unprompted remarks sound like him and not like a notification."""
    try:
        resp = _client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=120,
            system=_PHRASE_SYSTEM + _core_profile() + _memory_context(),
            messages=[{"role": "user", "content":
                       f"Facts to convey ({item.kind}): {item.summary}"}],
        )
        line = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        return line or item.summary
    except Exception as e:
        print(f"[proactive] phrasing failed ({e}); using raw summary")
        return item.summary


def record_proactive(line: str):
    """Log what SEVRIN said unprompted, so it's part of conversation history
    and he doesn't repeat himself or lose the thread."""
    store.add_turn("assistant", line)
