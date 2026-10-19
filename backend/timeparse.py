"""
SEVRIN — parsing spoken time expressions.

Deterministic patterns first, model only as a fallback. A scheduling error
is a worse failure than a clumsy sentence: if he asks to be reminded at 6pm
and it fires at 6am, the feature is actively harmful. So the common shapes
are handled by explicit rules whose behaviour is verifiable, and only
genuinely unusual phrasings ("the Tuesday after next") reach the model —
whose output is then validated before it's trusted.

All times are LOCAL time.

Returns a dict:
    {"when": <epoch seconds>, "recurrence": <dict|None>, "subject": <str>}
or None if the text isn't a scheduling request at all.
"""
import re
from datetime import datetime, timedelta

# Speech-to-text writes numbers as words as often as digits — "five minute
# timer", not "5 minute timer". Without these, a perfectly clear request
# silently fails to parse.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "ninety": 90, "half": 0,   # "half an hour" handled below
}
_NUM_WORD_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))


# --- relative: "in 5 minutes" ---------------------------------------------
_UNIT_RE = r"second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|day|days"

# Accepts all of:
#   "in 5 minutes" / "in five minutes" / "in half an hour"
#   "for 5 minutes" / "for five minutes"      (timer phrasing)
#   "a five minute timer" / "5 minute timer"  (no preposition at all)
_REL = re.compile(
    r"\b(?:in|for)?\s*(?:(\d+)|(" + _NUM_WORD_RE + r"))\s*"
    r"(" + _UNIT_RE + r")\b",
    re.I,
)
_HALF_HOUR = re.compile(r"\bhalf\s+(?:an\s+)?hour\b", re.I)
_UNIT = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
}

# --- absolute clock time --------------------------------------------------
# Handles "at 6pm", "at 18:30", "at 6:30 pm" AND colon-less forms like
# "at 215" or "at 1430". That last group matters because speech-to-text
# renders "two fifteen" as "215" — a real reminder was silently missed
# because the pattern below originally required a colon or an am/pm marker.
_CLOCK = re.compile(
    r"\bat\s+(\d{1,2})(?::\s*(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b",
    re.I,
)
_CLOCK_COMPACT = re.compile(
    r"\bat\s+(\d{3,4})\s*(am|pm|a\.m\.|p\.m\.)?\b",
    re.I,
)


def _compact_clock(match):
    """Parse a colon-less time: 215 -> 2:15, 1430 -> 14:30.

    Returns (hour, minute, ampm) or None if the digits don't form a valid
    time — better to fall through than to invent one.
    """
    digits = match.group(1)
    ampm = (match.group(2) or "").lower().replace(".", "")
    if len(digits) == 3:
        hour, minute = int(digits[0]), int(digits[1:])
    else:
        hour, minute = int(digits[:2]), int(digits[2:])
    if minute > 59 or hour > 23:
        return None
    if ampm.startswith("p") and hour < 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0
    return hour, minute, ampm

# --- day words ------------------------------------------------------------
_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_TONIGHT = re.compile(r"\btonight\b", re.I)
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.I)

# --- recurrence -----------------------------------------------------------
_EVERY_DAY = re.compile(r"\bevery\s+day\b|\bdaily\b", re.I)
_EVERY_WEEKDAY = re.compile(r"\bevery\s+weekday\b|\bon\s+weekdays\b", re.I)
_EVERY_WEEKDAY_NAMED = re.compile(
    r"\bevery\s+(" + "|".join(_WEEKDAYS) + r")\b", re.I)

# --- vague times of day ---------------------------------------------------
_TIMES_OF_DAY = {
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (19, 0),
    "night": (21, 0),
    "noon": (12, 0),
    "midday": (12, 0),
}
_TOD_RE = re.compile(r"\b(" + "|".join(_TIMES_OF_DAY) + r")\b", re.I)

# does this even look like a scheduling request?
_TRIGGER = re.compile(
    r"\b(remind|reminder|remember|wake me|tell me|let me know|alert me|"
    r"nudge me|ping me|timer|alarm|countdown|set a timer|buzz me)\b", re.I)


def looks_like_schedule(text: str) -> bool:
    return bool(_TRIGGER.search(text))


def _clock_from(match, now, same_day=True, text=""):
    """Turn a matched clock expression into (hour, minute).

    The hard case is a bare hour with no am/pm ("at 7"). Testing caught four
    wrong answers here, all of the worst kind — a confidently returned but
    incorrect time. "tomorrow at 7" became 19:00, "every weekday at 9"
    became a 21:00 standup, "friday at 3" became 03:00.

    The rule that fixes it: the current clock only matters when the target is
    TODAY. For another day, or a recurring item, "at 7" means the ordinary
    reading of that hour, not whatever is next relative to right now.
    """
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower().replace(".", "")

    if ampm.startswith("p") and hour < 12:
        return hour + 12, minute
    if ampm.startswith("a"):
        return (0 if hour == 12 else hour), minute
    if hour >= 13:
        return hour, minute            # 24-hour clock, unambiguous

    # Bare hour. Look for an explicit hint first.
    if re.search(r"\b(morning|am)\b", text, re.I):
        return hour, minute
    if re.search(r"\b(evening|night|afternoon|tonight|pm)\b", text, re.I):
        return (hour + 12 if hour < 12 else hour), minute

    if not same_day:
        # A different day: interpret by everyday convention rather than by
        # what's next from now. 1-6 read as afternoon/evening, 7-12 as morning.
        return (hour + 12 if 1 <= hour <= 6 else hour), minute

    # Today: pick the next occurrence of that hour that hasn't passed.
    if hour < 12 and now.hour >= hour:
        pm = hour + 12
        if now.hour < pm:
            return pm, minute
    return hour, minute


def _extract_subject(text: str) -> str:
    """Strip the scheduling scaffolding and keep what it's actually about."""
    s = text
    s = re.sub(r"\b(can you|could you|please|hey jarvis|jarvis|set me|set|"
               r"a|an|for)\b", " ", s, flags=re.I)
    s = re.sub(r"\b(timer|alarm|countdown)\b", " ", s, flags=re.I)
    s = _HALF_HOUR.sub(" ", s)
    s = _TRIGGER.sub(" ", s)
    s = _REL.sub(" ", s)
    s = _CLOCK_COMPACT.sub(" ", s)
    s = _CLOCK.sub(" ", s)
    s = _EVERY_DAY.sub(" ", s)
    s = _EVERY_WEEKDAY.sub(" ", s)
    s = _EVERY_WEEKDAY_NAMED.sub(" ", s)
    s = _WEEKDAY_RE.sub(" ", s)
    s = _TOMORROW.sub(" ", s)
    s = _TONIGHT.sub(" ", s)
    s = _TOD_RE.sub(" ", s)
    s = re.sub(r"\b(me|to|that|about|the|at|on|in|i|have|has|a|an)\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" .,!?-")
    return s


def parse(text: str, now: datetime = None):
    """Deterministic parse. Returns a dict or None if no pattern matched."""
    now = now or datetime.now()
    subject = _extract_subject(text)

    # ---- recurring ----
    recurrence = None
    if _EVERY_WEEKDAY.search(text):
        recurrence = {"type": "weekdays"}
    elif _EVERY_DAY.search(text):
        recurrence = {"type": "daily"}
    else:
        m = _EVERY_WEEKDAY_NAMED.search(text)
        if m:
            recurrence = {"type": "weekly", "weekday": _WEEKDAYS[m.group(1).lower()]}

    # ---- time of day ----
    # Is the target today, or another day / recurring? That decides how a
    # bare hour is read.
    another_day = bool(
        _TOMORROW.search(text) or _WEEKDAY_RE.search(text) or recurrence is not None
    )

    hour = minute = None
    compact = _CLOCK_COMPACT.search(text)
    clock = _CLOCK.search(text)

    if compact:
        parsed_compact = _compact_clock(compact)
        if parsed_compact:
            hour, minute, had_ampm = parsed_compact
            # A bare compact time still needs am/pm resolution ("at 215").
            if not had_ampm and hour < 12:
                if another_day:
                    hour = hour + 12 if 1 <= hour <= 6 else hour
                elif now.hour >= hour and now.hour < hour + 12:
                    hour += 12
    elif clock:
        hour, minute = _clock_from(clock, now, same_day=not another_day, text=text)
    else:
        tod = _TOD_RE.search(text)
        if tod:
            hour, minute = _TIMES_OF_DAY[tod.group(1).lower()]

    # ---- relative offsets take priority; they're unambiguous ----
    if _HALF_HOUR.search(text) and not recurrence:
        when = now + timedelta(minutes=30)
        return {"when": when.timestamp(), "recurrence": None, "subject": subject}

    rel = _REL.search(text)
    if rel and not recurrence:
        if rel.group(1):
            amount = int(rel.group(1))
        else:
            amount = _NUMBER_WORDS.get((rel.group(2) or "").lower(), 1)
        unit = rel.group(3).lower()
        if amount <= 0:
            amount = 1
        when = now + timedelta(seconds=amount * _UNIT[unit])
        return {"when": when.timestamp(), "recurrence": None, "subject": subject}

    if recurrence is not None:
        if hour is None:
            hour, minute = 9, 0          # sensible default for a recurring item
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if recurrence["type"] == "weekly":
            target = _advance_to_weekday(target, recurrence["weekday"], now)
        elif recurrence["type"] == "weekdays":
            if target <= now:
                target += timedelta(days=1)
            while target.weekday() > 4:
                target += timedelta(days=1)
        else:  # daily
            if target <= now:
                target += timedelta(days=1)
        recurrence["hour"] = hour
        recurrence["minute"] = minute
        return {"when": target.timestamp(), "recurrence": recurrence, "subject": subject}

    # ---- one-off at a clock time / day ----
    if hour is not None:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        wd = _WEEKDAY_RE.search(text)
        if _TOMORROW.search(text):
            target += timedelta(days=1)
        elif wd:
            target = _advance_to_weekday(target, _WEEKDAYS[wd.group(1).lower()], now)
        elif target <= now:
            target += timedelta(days=1)     # that time has passed today
        return {"when": target.timestamp(), "recurrence": None, "subject": subject}

    # bare "tomorrow" with no time -> tomorrow morning
    if _TOMORROW.search(text):
        target = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return {"when": target.timestamp(), "recurrence": None, "subject": subject}

    return None


def _advance_to_weekday(target, weekday, now):
    days = (weekday - target.weekday()) % 7
    if days == 0 and target <= now:
        days = 7
    return target + timedelta(days=days)


def next_occurrence(recurrence: dict, after: datetime = None):
    """Given a recurrence spec, when does it next fire after `after`?"""
    after = after or datetime.now()
    hour = recurrence.get("hour", 9)
    minute = recurrence.get("minute", 0)
    target = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= after:
        target += timedelta(days=1)

    rtype = recurrence.get("type")
    if rtype == "weekdays":
        while target.weekday() > 4:
            target += timedelta(days=1)
    elif rtype == "weekly":
        wd = recurrence.get("weekday", 0)
        while target.weekday() != wd:
            target += timedelta(days=1)
    return target.timestamp()
