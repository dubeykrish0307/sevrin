"""
SEVRIN — persistent scheduler.

Replaces the in-memory timers, which lost everything on restart. Schedules
live in SQLite, so "remind me at 6pm" survives the backend going down and
coming back.

Three things this handles that the timers didn't:

  PERSISTENCE  Schedules are rows, not objects. Restarting reloads them.

  DRIFT        The old loop polled once a second, and the proactive queue
                polled again on top of it, so a 30s reminder could land ~1s
                late and the errors compounded. This sleeps until the exact
                moment the next item is due (waking early if something new
                is scheduled), so firing is precise regardless of how long
                the wait is.

  MISSED ITEMS  If the backend was off when something was due, it's caught
                on startup. Recently missed items still fire, flagged as
                late — a reminder five minutes overdue is usually still
                wanted. Long-overdue ones are skipped rather than delivered
                as confusing history, and recurring ones simply advance to
                their next occurrence.
"""
import threading
import time
from datetime import datetime

from brain import store
from backend import timeparse

# How overdue a missed item can be and still be worth saying.
MISSED_GRACE_SECONDS = 15 * 60


class Scheduler:
    def __init__(self, proactive_queue):
        self.queue = proactive_queue
        self._stop = threading.Event()
        self._wake = threading.Event()   # set when a new schedule is added
        self._thread = None

    # ---------------- creating schedules ----------------

    def schedule_from_text(self, text: str):
        """If `text` is a scheduling request, persist it and return
        (when_epoch, subject, recurrence). Otherwise None."""
        if not timeparse.looks_like_schedule(text):
            return None
        parsed = timeparse.parse(text)
        if not parsed:
            return None

        when = parsed["when"]
        # Guard against a parse that lands in the past — better to refuse
        # than to fire instantly and confuse him.
        if when <= time.time() - 5:
            print(f"[scheduler] parsed a time in the past; ignoring ({text!r})")
            return None

        store.add_schedule(
            subject=parsed["subject"],
            next_due=when,
            recurrence=parsed["recurrence"],
            source_text=text,
        )
        self._wake.set()   # recompute the sleep window immediately

        when_str = datetime.fromtimestamp(when).strftime("%a %H:%M")
        rec = parsed["recurrence"]["type"] if parsed["recurrence"] else "once"
        print(f"[scheduler] scheduled '{parsed['subject']}' for {when_str} ({rec})")
        return when, parsed["subject"], parsed["recurrence"]

    def upcoming(self, limit=5):
        return store.active_schedules()[:limit]

    # ---------------- the loop ----------------

    def start(self):
        self._stop.clear()
        self._catch_up()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        n = len(store.active_schedules())
        print(f"[scheduler] running ({n} active schedule(s))")

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _catch_up(self):
        """Handle anything that fell due while the backend was down."""
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
            now = time.time()
            nxt = store.next_due_time()

            if nxt is None:
                # nothing scheduled — wait to be woken by a new one
                self._wake.wait(timeout=30)
                self._wake.clear()
                continue

            delay = nxt - now
            if delay > 0:
                # Sleep until exactly when it's due, but wake early if a new
                # schedule is added. This is what removes the polling drift.
                self._wake.wait(timeout=min(delay, 30))
                self._wake.clear()
                continue

            for row in store.due_schedules():
                self._fire(row, _load_recurrence(row))

    # ---------------- firing ----------------

    def _fire(self, row, recurrence, late_note=""):
        subject = row["subject"]
        summary = (f"the reminder he asked for: {subject}{late_note}"
                   if subject else f"the reminder he asked for{late_note}")
        self.queue.submit(
            kind="reminder",
            summary=summary,
            # He chose the time, so delivering it late defeats the point.
            urgency="critical",
            deadline=time.time() + MISSED_GRACE_SECONDS,
            context={"schedule_id": row["id"]},
        )
        self._advance(row, recurrence)

    def _advance(self, row, recurrence):
        """Reschedule a recurring item, or close out a one-shot."""
        if recurrence:
            nxt = timeparse.next_occurrence(recurrence)
            store.mark_fired(row["id"], next_due=nxt)
            print(f"[scheduler] '{row['subject']}' recurs next at "
                  f"{datetime.fromtimestamp(nxt):%a %H:%M}")
        else:
            store.mark_fired(row["id"])


def _load_recurrence(row):
    import json
    if not row.get("recurrence"):
        return None
    try:
        return json.loads(row["recurrence"])
    except Exception:
        return None
