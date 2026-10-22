"""One Anthropic client for the whole app, plus helpers for one-shot calls."""
import json
import re

from anthropic import Anthropic

import config

client = Anthropic(api_key=config.ANTHROPIC_API_KEY)

_CODE_FENCE = re.compile(r"^```(?:json)?|```$", re.M)


def ask(system: str, user: str, model: str = config.VERIFY_MODEL, max_tokens: int = 800) -> str:
    """Single non-streaming call. Returns the reply text."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def parse_json(text: str):
    """Pull JSON out of a reply. Models sometimes wrap it in a code fence or
    add a sentence around it even when told not to. Returns None if nothing parses."""
    text = _CODE_FENCE.sub("", text.strip()).strip()
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    end = max(text.rfind("]"), text.rfind("}"))
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None
