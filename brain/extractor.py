"""Decides what's worth remembering from a user turn, and checks it first.

Storing something wrong is worse than storing nothing, so every layer can
veto a candidate fact:

  0. cheap filter   skip greetings and short turns (no API call)
  1. extract        propose facts, each with the verbatim evidence for it
  2. grounding      a separate call checks the evidence is really in the message
  3. contradiction  compare with stored facts: new, duplicate, update or conflict
  4. threshold      combined confidence must reach MEMORY_MIN_CONFIDENCE

What each layer concluded is saved with the fact, so "why do you think that?"
has an answer.
"""
import re

import config
from brain import prompts, store
from brain.claude import ask, parse_json

_TRIVIAL = re.compile(
    r"^\s*(hi|hey|hello|yo|thanks|thank you|ok|okay|cool|nice|got it|sure|yes|no|yeah|nope"
    r"|good morning|good night|bye|test|testing)\b[\s!.?]*$",
    re.I,
)


def _worth_examining(user_text: str) -> bool:
    text = user_text.strip()
    return len(text) >= 12 and not _TRIVIAL.match(text)


def _extract_candidates(user_text: str):
    data = parse_json(ask(prompts.EXTRACT, f'User\'s message:\n"""\n{user_text}\n"""'))
    if not isinstance(data, list):
        return []
    candidates = []
    for item in data:
        if not isinstance(item, dict) or not item.get("fact") or not item.get("evidence"):
            continue
        candidates.append({
            "subject": (item.get("subject") or "general").strip().lower(),
            "fact": item["fact"].strip(),
            "confidence": float(item.get("confidence", 0.5)),
            "evidence": item["evidence"].strip(),
        })
    return candidates


def _verify_grounding(user_text: str, candidate: dict) -> dict:
    prompt = (
        f'ORIGINAL MESSAGE:\n"""\n{user_text}\n"""\n\n'
        f"CLAIM: {candidate['fact']}\n"
        f"CLAIMED EVIDENCE: {candidate['evidence']}"
    )
    data = parse_json(ask(prompts.GROUND, prompt, max_tokens=300))
    if not isinstance(data, dict):
        return {"grounded": False, "reason": "verifier returned unparseable output",
                "adjusted_confidence": 0.0}
    return {
        "grounded": bool(data.get("grounded")),
        "reason": str(data.get("reason", ""))[:200],
        "adjusted_confidence": float(data.get("adjusted_confidence", 0.0)),
    }


def _check_contradictions(candidate: dict) -> dict:
    existing = store.active_facts(limit=80)
    if not existing:
        return {"relation": "new", "conflicts_with_id": None, "reason": "no stored facts yet"}

    listing = "\n".join(f"[{f['id']}] ({f['subject']}) {f['fact']}" for f in existing)
    prompt = f"STORED FACTS:\n{listing}\n\nNEW FACT: ({candidate['subject']}) {candidate['fact']}"
    data = parse_json(ask(prompts.CONTRADICTION, prompt, max_tokens=300))
    if not isinstance(data, dict):
        return {"relation": "new", "conflicts_with_id": None, "reason": "checker unparseable; treated as new"}

    relation = data.get("relation", "new")
    if relation not in ("new", "duplicate", "update", "conflict"):
        relation = "new"
    cid = data.get("conflicts_with_id")
    return {
        "relation": relation,
        "conflicts_with_id": int(cid) if isinstance(cid, (int, float)) else None,
        "reason": str(data.get("reason", ""))[:200],
    }


def process_turn(user_text: str, source_turn_id: int = None, on_event=None):
    """Run every layer on one user turn. Returns one event dict per candidate
    (stored, updated, duplicate, rejected or error); on_event gets each one live."""
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
        fact = cand["fact"]
        checks = {"extract_confidence": cand["confidence"]}

        try:
            ground = _verify_grounding(user_text, cand)
        except Exception as e:
            emit({"stage": "ground", "fact": fact, "outcome": "error", "detail": str(e)})
            continue
        checks["grounding"] = ground
        if not ground["grounded"]:
            emit({"stage": "ground", "fact": fact, "outcome": "rejected", "detail": ground["reason"]})
            continue

        # be conservative: take the lower of the two confidences
        confidence = min(cand["confidence"], ground["adjusted_confidence"])
        checks["combined_confidence"] = confidence

        try:
            contra = _check_contradictions(cand)
        except Exception as e:
            emit({"stage": "contradiction", "fact": fact, "outcome": "error", "detail": str(e)})
            continue
        checks["contradiction"] = contra

        if contra["relation"] == "duplicate":
            emit({"stage": "contradiction", "fact": fact, "outcome": "duplicate", "detail": contra["reason"]})
            continue
        if contra["relation"] == "conflict":
            emit({"stage": "contradiction", "fact": fact, "outcome": "rejected", "detail": contra["reason"]})
            continue

        if confidence < config.MEMORY_MIN_CONFIDENCE:
            emit({"stage": "threshold", "fact": fact, "outcome": "rejected",
                  "detail": f"confidence {confidence:.2f} below {config.MEMORY_MIN_CONFIDENCE}"})
            continue

        new_id = store.add_fact(
            subject=cand["subject"], fact=fact, confidence=confidence,
            evidence=cand["evidence"], checks=checks, source_turn=source_turn_id,
        )
        if contra["relation"] == "update" and contra["conflicts_with_id"]:
            store.supersede_fact(contra["conflicts_with_id"], new_id)
            emit({"stage": "store", "fact": fact, "outcome": "updated",
                  "detail": f"replaced fact #{contra['conflicts_with_id']}", "id": new_id})
        else:
            emit({"stage": "store", "fact": fact, "outcome": "stored",
                  "detail": f"confidence {confidence:.2f}", "id": new_id})

    return results
