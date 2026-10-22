"""Parse spoken reminder requests ("remind me at 6pm to call mum").

Rules only, no model. A reminder that fires at 6am instead of 6pm is worse
than useless, so every case here is explicit and testable. All times are local.

parse() returns {"when": epoch, "recurrence": dict | None, "subject": str},
or None if no time could be found.
"""
import re
from datetime import datetime, timedelta

# STT writes numbers as words about as often as digits ("five minute timer").
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "ninety": 90, "half": 0,   # "half an hour" has its own pattern
}
_NUM_WORD_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

# relative: "in 5 minutes", "for five minutes", "a five minute timer"
_UNIT_RE = r"second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|day|days"
_REL = re.compile(
    r"\b(?:in|for)?\s*(?:(\d+)|(" + _NUM_WORD_RE + r"))\s*"
    r"(" + _UNIT_RE + r")\b",
    re.I,
)
_HALF_HOUR = re.compile(r"\bhalf\s+(?:an\s+)?hour\b", re.I)
_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600,
    "day": 86400, "days": 86400,
}

# clock times: "at 6pm", "at 18:30", "at 6:30 pm"
_CLOCK = re.compile(r"\bat\s+(\d{1,2})(?::\s*(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I)
# STT turns "two fifteen" into "215", so colon-less times need their own pattern
_CLOCK_COMPACT = re.compile(r"\bat\s+(\d{3,4})\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I)

_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_TONIGHT = re.compile(r"\btonight\b", re.I)
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WEEKDAY_RE = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")\b", re.I)

_EVERY_DAY = re.compile(r"\bevery\s+day\b|\bdaily\b", re.I)
_EVERY_WEEKDAY = re.compile(r"\bevery\s+weekday\b|\bon\s+weekdays\b", re.I)
_EVERY_NAMED_DAY = re.compile(r"\bevery\s+(" + "|".join(_WEEKDAYS) + r")\b", re.I)

_TIMES_OF_DAY = {
    "morning": (9, 0),
    "afternoon": (14, 0),
    "evening": (19, 0),
    "night": (21, 0),
    "noon": (12, 0),
    "midday": (12, 0),
}
_TOD_RE = re.compile(r"\b(" + "|".join(_TIMES_OF_DAY) + r")\b", re.I)

_AM_HINT = re.compile(r"\b(morning|am)\b", re.I)
_PM_HINT = re.compile(r"\b(evening|night|afternoon|tonight|pm)\b", re.I)

_TRIGGER = re.compile(
    r"\b(remind|reminder|remember|wake me|tell me|let me know|alert me|"
    r"nudge me|ping me|timer|alarm|countdown|set a timer|buzz me)\b", re.I)

# words stripped out when working out what the reminder is about
_FILLER = re.compile(r"\b(can you|could you|please|hey jarvis|jarvis|set me|set|a|an|for)\b", re.I)
_TIMER_WORDS = re.compile(r"\b(timer|alarm|countdown)\b", re.I)
_STOPWORDS = re.compile(r"\b(me|to|that|about|the|at|on|in|i|have|has|a|an)\b", re.I)


def looks_like_schedule(text: str) -> bool:
    return bool(_TRIGGER.search(text))


def _ampm_hour(hour: int, ampm: str) -> int:
    if ampm.startswith("p") and hour < 12:
        return hour + 12
    if ampm.startswith("a") and hour == 12:
        return 0
    return hour


def _compact_clock(match):
    """'215' -> (2, 15, ampm), '1430' -> (14, 30, ampm). None if it isn't a real time."""
    digits = match.group(1)
    ampm = (match.group(2) or "").lower().replace(".", "")
    if len(digits) == 3:
        hour, minute = int(digits[0]), int(digits[1:])
    else:
        hour, minute = int(digits[:2]), int(digits[2:])
    if minute > 59 or hour > 23:
        return None
    return _ampm_hour(hour, ampm), minute, ampm


def _clock_from(match, now, same_day=True, text=""):
    """(hour, minute) for a matched clock time.

    The tricky case is a bare hour like "at 7". Only for today does the
    current time matter (pick the next 7 that hasn't passed). For another day
    or a recurring reminder, use the everyday reading: 1-6 means pm, 7-12 am.
    Otherwise "tomorrow at 7" came out as 19:00 and "friday at 3" as 03:00.
    """
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower().replace(".", "")

    if ampm.startswith("p") and hour < 12:
        return hour + 12, minute
    if ampm.startswith("a"):
        return (0 if hour == 12 else hour), minute
    if hour >= 13:
        return hour, minute     # 24-hour clock

    if _AM_HINT.search(text):
        return hour, minute
    if _PM_HINT.search(text):
        return (hour + 12 if hour < 12 else hour), minute

    if not same_day:
        return (hour + 12 if 1 <= hour <= 6 else hour), minute

    if hour < 12 and now.hour >= hour:
        pm = hour + 12
        if now.hour < pm:
            return pm, minute
    return hour, minute


def _extract_subject(text: str) -> str:
    """Strip the scheduling words and keep what the reminder is about."""
    s = text
    for pattern in (_FILLER, _TIMER_WORDS, _HALF_HOUR, _TRIGGER, _REL, _CLOCK_COMPACT, _CLOCK,
                    _EVERY_DAY, _EVERY_WEEKDAY, _EVERY_NAMED_DAY, _WEEKDAY_RE,
                    _TOMORROW, _TONIGHT, _TOD_RE, _STOPWORDS):
        s = pattern.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" .,!?-")


def _recurrence_in(text: str):
    if _EVERY_WEEKDAY.search(text):
        return {"type": "weekdays"}
    if _EVERY_DAY.search(text):
        return {"type": "daily"}
    m = _EVERY_NAMED_DAY.search(text)
    if m:
        return {"type": "weekly", "weekday": _WEEKDAYS[m.group(1).lower()]}
    return None


def parse(text: str, now: datetime = None):
    now = now or datetime.now()
    subject = _extract_subject(text)
    recurrence = _recurrence_in(text)

    # whether the target is today decides how a bare hour is read
    another_day = bool(_TOMORROW.search(text) or _WEEKDAY_RE.search(text) or recurrence is not None)

    hour = minute = None
    compact = _CLOCK_COMPACT.search(text)
    clock = _CLOCK.search(text)
    if compact:
        parsed = _compact_clock(compact)
        if parsed:
            hour, minute, ampm = parsed
            if not ampm and hour < 12:
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

    # relative offsets are unambiguous, so they win
    if _HALF_HOUR.search(text) and not recurrence:
        when = now + timedelta(minutes=30)
        return {"when": when.timestamp(), "recurrence": None, "subject": subject}

    rel = _REL.search(text)
    if rel and not recurrence:
        if rel.group(1):
            amount = int(rel.group(1))
        else:
            amount = _NUMBER_WORDS.get((rel.group(2) or "").lower(), 1)
        amount = max(amount, 1)
        when = now + timedelta(seconds=amount * _UNIT_SECONDS[rel.group(3).lower()])
        return {"when": when.timestamp(), "recurrence": None, "subject": subject}

    if recurrence is not None:
        if hour is None:
            hour, minute = 9, 0
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if recurrence["type"] == "weekly":
            target = _advance_to_weekday(target, recurrence["weekday"], now)
        elif recurrence["type"] == "weekdays":
            if target <= now:
                target += timedelta(days=1)
            while target.weekday() > 4:
                target += timedelta(days=1)
        elif target <= now:   # daily
            target += timedelta(days=1)
        recurrence["hour"] = hour
        recurrence["minute"] = minute
        return {"when": target.timestamp(), "recurrence": recurrence, "subject": subject}

    if hour is not None:
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        weekday = _WEEKDAY_RE.search(text)
        if _TOMORROW.search(text):
            target += timedelta(days=1)
        elif weekday:
            target = _advance_to_weekday(target, _WEEKDAYS[weekday.group(1).lower()], now)
        elif target <= now:
            target += timedelta(days=1)
        return {"when": target.timestamp(), "recurrence": None, "subject": subject}

    # "tomorrow" with no time means tomorrow morning
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
    """Epoch time of the next firing of a recurring schedule after `after`."""
    after = after or datetime.now()
    target = after.replace(hour=recurrence.get("hour", 9), minute=recurrence.get("minute", 0),
                           second=0, microsecond=0)
    if target <= after:
        target += timedelta(days=1)

    rtype = recurrence.get("type")
    if rtype == "weekdays":
        while target.weekday() > 4:
            target += timedelta(days=1)
    elif rtype == "weekly":
        weekday = recurrence.get("weekday", 0)
        while target.weekday() != weekday:
            target += timedelta(days=1)
    return target.timestamp()
