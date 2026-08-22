"""
SEVRIN — simple timer reminders.

Deliberately minimal: this exists so proactive speech can be tested
end-to-end with something low-stakes ("remind me in 30 seconds") before
calendar integration depends on it. A real scheduler with persistence and
recurrence comes next.

Reminders live in memory only for now, so they don't survive a restart.
That's fine for a timer; it would not be fine for "remind me tomorrow",
which is exactly why the persistent scheduler is the next piece.
"""
import re
import threading
import time

# "remind me in 5 minutes to call mum" / "in 30 seconds, check the build"
_PATTERN = re.compile(
    r"\b(?:remind me|remember|tell me|let me know)\b.*?\bin\s+"
    r"(\d+)\s*(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours)\b"
    r"(?:\s*(?:to|that|about)?\s*(.*))?$",
    re.I,
)

_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600,
}


def parse(text: str):
    """Returns (delay_seconds, subject) or None if this isn't a reminder."""
    m = _PATTERN.search(text.strip())
    if not m:
        return None
    amount = int(m.group(1))
    unit = m.group(2).lower()
    subject = (m.group(3) or "").strip(" .,!?")
    delay = amount * _UNIT_SECONDS.get(unit, 60)
    return delay, subject


class ReminderTimers:
    """Holds reminders until they're due, then hands them to the proactive
    queue — which then decides WHEN to actually say them.

    Note the separation: this decides when a reminder becomes RELEVANT; the
    proactive queue decides when it's APPROPRIATE to speak. An earlier draft
    submitted straight to the queue at scheduling time, which would have
    fired the reminder immediately instead of when it was due.
    """

    def __init__(self, queue):
        self.queue = queue
        self._pending = []
        self._stop = threading.Event()
        self._thread = None

    def schedule(self, delay_seconds, subject):
        due = time.time() + delay_seconds
        self._pending.append({"due": due, "subject": subject})
        return due

    def schedule_from_text(self, text):
        """If `text` is a reminder request, schedule it and return
        (delay, subject). Otherwise None, so the caller handles it normally."""
        parsed = parse(text)
        if not parsed:
            return None
        delay, subject = parsed
        self.schedule(delay, subject)
        print(f"[reminders] scheduled in {delay}s: {subject or '(no subject)'}")
        return delay, subject

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            now = time.time()
            due = [r for r in self._pending if r["due"] <= now]
            for r in due:
                self._pending.remove(r)
                subject = r["subject"]
                summary = (f"the reminder he asked for: {subject}" if subject
                           else "the reminder he asked for, with no subject given")
                self.queue.submit(
                    kind="reminder",
                    summary=summary,
                    # He asked for this AT a time; late delivery defeats it.
                    urgency="critical",
                    # Stale after a few minutes — better dropped than wrong.
                    deadline=r["due"] + 300,
                )
            self._stop.wait(1.0)
