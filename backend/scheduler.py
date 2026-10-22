"""Reminders that survive a restart.

Schedules are rows in SQLite. The loop sleeps until the next one is due
(waking early when a new one is added) instead of polling, so reminders fire
on time. Anything that fell due while the backend was off is handled on
startup: up to 15 minutes late it still fires, older than that it's skipped,
and recurring ones move on to their next occurrence.
"""
import json
import threading
import time
from datetime import datetime

from backend import timeparse
from brain import store

MISSED_GRACE_SECONDS = 15 * 60


class Scheduler:
    def __init__(self, proactive_queue):
        self.queue = proactive_queue
        self._stop = threading.Event()
        self._wake = threading.Event()   # set when a schedule is added
        self._thread = None

    def schedule_from_text(self, text: str):
        """If text is a reminder request, save it and return
        (when_epoch, subject, recurrence). Otherwise None."""
        if not timeparse.looks_like_schedule(text):
            return None
        parsed = timeparse.parse(text)
        if not parsed:
            return None

        when = parsed["when"]
        if when <= time.time() - 5:
            # firing instantly would just be confusing
            print(f"[scheduler] parsed a time in the past; ignoring ({text!r})")
            return None

        store.add_schedule(
            subject=parsed["subject"],
            next_due=when,
            recurrence=parsed["recurrence"],
            source_text=text,
        )
        self._wake.set()

        rec = parsed["recurrence"]["type"] if parsed["recurrence"] else "once"
        print(f"[scheduler] scheduled '{parsed['subject']}' for "
              f"{datetime.fromtimestamp(when):%a %H:%M} ({rec})")
        return when, parsed["subject"], parsed["recurrence"]

    def upcoming(self, limit=5):
        return store.active_schedules()[:limit]

    def start(self):
        self._stop.clear()
        self._catch_up()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[scheduler] running ({len(store.active_schedules())} active schedule(s))")

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _catch_up(self):
        """Deal with anything that came due while the backend was down."""
        now = time.time()
        for row in store.due_schedules(now):
            overdue = now - row["next_due"]
            recurrence = _load_recurrence(row)
            if overdue <= MISSED_GRACE_SECONDS:
                mins = int(overdue // 60)
                late = f" (about {mins} minute(s) late)" if mins >= 1 else ""
                self._fire(row, recurrence, late_note=late)
            else:
                print(f"[scheduler] skipping long-overdue '{row['subject']}' "
                      f"({int(overdue // 60)} min late)")
                self._advance(row, recurrence)

    def _run(self):
        while not self._stop.is_set():
            next_due = store.next_due_time()
            if next_due is None:
                self._wake.wait(timeout=30)
                self._wake.clear()
                continue

            delay = next_due - time.time()
            if delay > 0:
                self._wake.wait(timeout=min(delay, 30))
                self._wake.clear()
                continue

            for row in store.due_schedules():
                self._fire(row, _load_recurrence(row))

    def _fire(self, row, recurrence, late_note=""):
        subject = row["subject"]
        summary = (f"the reminder he asked for: {subject}{late_note}" if subject
                   else f"the reminder he asked for{late_note}")
        self.queue.submit(
            kind="reminder",
            summary=summary,
            urgency="critical",   # he picked the time, late is pointless
            deadline=time.time() + MISSED_GRACE_SECONDS,
            context={"schedule_id": row["id"]},
        )
        self._advance(row, recurrence)

    def _advance(self, row, recurrence):
        """Move a recurring schedule to its next time, or close a one-off."""
        if recurrence:
            next_due = timeparse.next_occurrence(recurrence)
            store.mark_fired(row["id"], next_due=next_due)
            print(f"[scheduler] '{row['subject']}' recurs next at "
                  f"{datetime.fromtimestamp(next_due):%a %H:%M}")
        else:
            store.mark_fired(row["id"])


def _load_recurrence(row):
    if not row.get("recurrence"):
        return None
    try:
        return json.loads(row["recurrence"])
    except ValueError:
        return None
