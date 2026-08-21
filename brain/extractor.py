"""
SEVRIN — autonomous fact extraction with multi-layer verification.

SEVRIN decides for itself what's worth remembering, but nothing reaches
long-term memory without surviving several independent checks. The design
principle: a memory system that confidently stores WRONG things about you
is worse than one that stores nothing. So every layer can veto.

    Layer 0  CHEAP FILTER   — skip turns that obviously carry no facts
                              (greetings, "what time is it"). Pure Python,
                              no API cost.

    Layer 1  EXTRACT        — Claude proposes candidate facts from the turn,
                              each with its own confidence and the verbatim
                              evidence it's based on.

    Layer 2  GROUNDING      — every candidate is re-checked against the
                              ACTUAL transcript by a separate call that did
                              not see layer 1's reasoning. Catches the model
                              inventing things the user never said. A fact
                              whose evidence isn't really in the text dies
                              here.

    Layer 3  CONTRADICTION  — the candidate is compared against existing
                              stored facts. If it conflicts, it's either a
                              genuine UPDATE (supersede the old one) or a
                              mistake (reject). This is what stops memory
                              rotting as things change.

    Layer 4  THRESHOLD      — combined confidence must clear a bar, and
                              volatile/low-value facts get dropped.

Everything each layer concluded is stored with the fact, so you can always
ask "why do you think that?" and get a real answer.
"""
import json
import re

from anthropic import Anthropic

import config
from brain import store

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

# Use a fast model for the verification passes — several calls per turn adds
# up, and these are narrow classification-style tasks, not open reasoning.
VERIFY_MODEL = "claude-haiku-4-5-20251001"

MIN_CONFIDENCE = getattr(config, "MEMORY_MIN_CONFIDENCE", 0.7)  # tunable in config.py


# ---------------- layer 0: cheap filter ----------------

_TRIVIAL = re.compile(
    r"^\s*(hi|hey|hello|yo|thanks|thank you|ok|okay|cool|nice|got it|sure|yes|no|yeah|nope"
    r"|good morning|good night|bye|test|testing)\b[\s!.?]*$",
    re.I,
)


def _worth_examining(user_text: str) -> bool:
    """Rejects turns that plainly can't contain a durable fact. Saves an API
    round-trip on every 'hey' and 'thanks'."""
    t = user_text.strip()
    if len(t) < 12:
        return False
    if _TRIVIAL.match(t):
        return False
    return True


def _json_from(text: str):
    """Models sometimes wrap JSON in prose or fences despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    end = max(text.rfind("]"), text.rfind("}"))
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def _ask(system: str, user: str, max_tokens: int = 800):
    resp = _client.messages.create(
        model=VERIFY_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


# ---------------- layer 1: extraction ----------------

_EXTRACT_SYSTEM = """You extract durable facts about a user from a single conversation turn.

A DURABLE fact is something still true weeks from now: preferences, tools they use,
projects they work on, people in their life, decisions they've made, constraints,
habits, skills, goals.

NOT durable (never extract these):
- transient state ("I'm tired", "I'm at the cafe right now")
- questions they asked
- things YOU (the assistant) said or suggested
- speculation, or anything they didn't clearly state
- one-off task requests ("summarize this")

Return ONLY a JSON array. Each item:
{"subject": "<short category, e.g. tooling|projects|people|preferences|schedule|skills>",
 "fact": "<one clear sentence, third person, e.g. 'Uses pnpm instead of npm'>",
 "confidence": <0.0-1.0 how certain you are they actually stated this>,
 "evidence": "<the VERBATIM span from their message that supports it>"}

If the turn contains no durable facts, return []. Be conservative — it is far better
to miss a fact than to invent one."""


def _extract_candidates(user_text: str):
    out = _ask(_EXTRACT_SYSTEM, f"User's message:\n\"\"\"\n{user_text}\n\"\"\"")
    data = _json_from(out)
    if not isinstance(data, list):
        return []
    clean = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("fact") or not item.get("evidence"):
            continue
        clean.append({
            "subject": (item.get("subject") or "general").strip().lower(),
            "fact": item["fact"].strip(),
            "confidence": float(item.get("confidence", 0.5)),
            "evidence": item["evidence"].strip(),
        })
    return clean


# ---------------- layer 2: grounding ----------------

_GROUND_SYSTEM = """You verify whether a claimed fact is genuinely supported by a message.

You will be given the user's ORIGINAL message and a CLAIM someone extracted from it.

Decide strictly:
- Is the claim's quoted evidence actually present in the message (allowing minor paraphrase)?
- Does the message genuinely support the claim, without inference or assumption?
- Is the claim about the USER (not about the assistant, not general world knowledge)?

Return ONLY JSON:
{"grounded": true|false, "reason": "<one short sentence>", "adjusted_confidence": <0.0-1.0>}

Be strict. If the claim requires ANY assumption beyond what's written, grounded is false."""


def _verify_grounding(user_text: str, candidate: dict):
    prompt = (
        f"ORIGINAL MESSAGE:\n\"\"\"\n{user_text}\n\"\"\"\n\n"
        f"CLAIM: {candidate['fact']}\n"
        f"CLAIMED EVIDENCE: {candidate['evidence']}"
    )
    data = _json_from(_ask(_GROUND_SYSTEM, prompt, max_tokens=300))
    if not isinstance(data, dict):
        return {"grounded": False, "reason": "verifier returned unparseable output", "adjusted_confidence": 0.0}
    return {
        "grounded": bool(data.get("grounded")),
        "reason": str(data.get("reason", ""))[:200],
        "adjusted_confidence": float(data.get("adjusted_confidence", 0.0)),
    }


# ---------------- layer 3: contradiction ----------------

_CONTRA_SYSTEM = """You compare a NEW fact about a user against FACTS ALREADY STORED about them.

Return ONLY JSON:
{"relation": "new" | "duplicate" | "update" | "conflict",
 "conflicts_with_id": <id or null>,
 "reason": "<one short sentence>"}

Definitions:
- "new": unrelated to anything stored.
- "duplicate": stored facts already say this; nothing to add.
- "update": genuinely supersedes a stored fact (their situation changed —
  e.g. switched tools, changed jobs). Give the id it replaces.
- "conflict": contradicts a stored fact but does NOT look like a real change
  (likely an extraction error). Give the id it conflicts with."""


def _check_contradictions(candidate: dict):
    existing = store.active_facts(limit=80)
    if not existing:
        return {"relation": "new", "conflicts_with_id": None, "reason": "no stored facts yet"}

    listing = "\n".join(f"[{f['id']}] ({f['subject']}) {f['fact']}" for f in existing)
    prompt = f"STORED FACTS:\n{listing}\n\nNEW FACT: ({candidate['subject']}) {candidate['fact']}"
    data = _json_from(_ask(_CONTRA_SYSTEM, prompt, max_tokens=300))
    if not isinstance(data, dict):
        return {"relation": "new", "conflicts_with_id": None, "reason": "checker unparseable; treated as new"}
    rel = data.get("relation", "new")
    if rel not in ("new", "duplicate", "update", "conflict"):
        rel = "new"
    cid = data.get("conflicts_with_id")
    return {
        "relation": rel,
        "conflicts_with_id": int(cid) if isinstance(cid, (int, float)) else None,
        "reason": str(data.get("reason", ""))[:200],
    }


# ---------------- orchestration ----------------

def process_turn(user_text: str, source_turn_id: int = None, on_event=None):
    """Runs the full pipeline for one user turn. Returns a list of dicts
    describing what happened to each candidate (stored / rejected / etc),
    which the UI can display so memory formation is visible, not magic.

    on_event(dict) is an optional callback for live UI updates."""
    results = []

    def emit(ev):
        results.append(ev)
        if on_event:
            try:
                on_event(ev)
            except Exception:
                pass

    if not _worth_examining(user_text):
        return results

    try:
        candidates = _extract_candidates(user_text)
    except Exception as e:
        emit({"stage": "extract", "outcome": "error", "detail": str(e)})
        return results

    for cand in candidates:
        checks = {"extract_confidence": cand["confidence"]}

        # layer 2
        try:
            ground = _verify_grounding(user_text, cand)
        except Exception as e:
            emit({"stage": "ground", "fact": cand["fact"], "outcome": "error", "detail": str(e)})
            continue
        checks["grounding"] = ground
        if not ground["grounded"]:
            emit({"stage": "ground", "fact": cand["fact"], "outcome": "rejected", "detail": ground["reason"]})
            continue

        # combine layer 1 + layer 2 confidence conservatively (take the lower)
        confidence = min(cand["confidence"], ground["adjusted_confidence"])
        checks["combined_confidence"] = confidence

        # layer 3
        try:
            contra = _check_contradictions(cand)
        except Exception as e:
            emit({"stage": "contradiction", "fact": cand["fact"], "outcome": "error", "detail": str(e)})
            continue
        checks["contradiction"] = contra

        if contra["relation"] == "duplicate":
            emit({"stage": "contradiction", "fact": cand["fact"], "outcome": "duplicate", "detail": contra["reason"]})
            continue
        if contra["relation"] == "conflict":
            emit({"stage": "contradiction", "fact": cand["fact"], "outcome": "rejected", "detail": contra["reason"]})
            continue

        # layer 4
        if confidence < MIN_CONFIDENCE:
            emit({"stage": "threshold", "fact": cand["fact"], "outcome": "rejected",
                  "detail": f"confidence {confidence:.2f} below {MIN_CONFIDENCE}"})
            continue

        new_id = store.add_fact(
            subject=cand["subject"], fact=cand["fact"], confidence=confidence,
            evidence=cand["evidence"], checks=checks, source_turn=source_turn_id,
        )

        if contra["relation"] == "update" and contra["conflicts_with_id"]:
            store.supersede_fact(contra["conflicts_with_id"], new_id)
            emit({"stage": "store", "fact": cand["fact"], "outcome": "updated",
                  "detail": f"replaced fact #{contra['conflicts_with_id']}", "id": new_id})
        else:
            emit({"stage": "store", "fact": cand["fact"], "outcome": "stored",
                  "detail": f"confidence {confidence:.2f}", "id": new_id})

    return results
