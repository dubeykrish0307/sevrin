"""Queue for things SEVRIN says without being asked (reminders, alerts, ...).

Speaking at the wrong moment is worse than not speaking, so each item goes
through three layers before it's said:

  1. expiry     past its deadline -> drop it
  2. structure  fixed rules on the current state: never talk over him or
                over a reply, speak critical items as soon as he's free,
                speak anything once he's been quiet for a while
  3. judgement  only the ambiguous middle (he only just went quiet) goes to
                the model

Most decisions never reach layer 3, which keeps them fast and predictable.
"""
import threading
import time
import uuid

# Urgency is set by whoever submits the item; the model doesn't guess it.
CRITICAL = "critical"   # actionable now, e.g. a reminder he set
TIMELY = "timely"       # matters soon, a minute's delay is fine
WHENEVER = "whenever"   # no time pressure, e.g. a task finished

_URGENCY_RANK = {CRITICAL: 3, TIMELY: 2, WHENEVER: 1}

MIN_GAP_BETWEEN_PROACTIVE = 20.0   # seconds between unprompted remarks
QUIET_PERIOD_SECONDS = 25.0        # idle this long = clearly free


class ProactiveItem:
    def __init__(self, kind, summary, urgency=TIMELY, deadline=None, context=None):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind              # 'reminder' | 'task_complete' | 'alert' | 'observation'
        self.summary = summary        # the facts; SEVRIN words it himself
        self.urgency = urgency
        self.deadline = deadline      # epoch seconds, after which it's pointless
        self.context = context or {}
        self.created = time.time()
        self.attempts = 0             # times the judge has been asked about it

    @property
    def age(self):
        return time.time() - self.created

    @property
    def expired(self):
        return self.deadline is not None and time.time() > self.deadline

    def __repr__(self):
        return f"<{self.kind}:{self.urgency} {self.summary[:40]!r}>"


class ProactiveQueue:
    """Holds pending items and decides when each one gets spoken.

    state_fn() returns the assistant's current state ('idle', 'listening',
    'thinking' or 'speaking'), speak_fn(item) says it, and judge_fn(item,
    situation) is the model call for ambiguous cases.
    """

    def __init__(self, state_fn, speak_fn, judge_fn=None):
        self._items = []
        self._lock = threading.Lock()
        self._state_fn = state_fn
        self._speak_fn = speak_fn
        self._judge_fn = judge_fn
        self._stop = threading.Event()
        self._thread = None
        self._last_spoke = 0.0
        self._last_user_activity = time.time()

    def submit(self, kind, summary, urgency=TIMELY, deadline=None, context=None):
        item = ProactiveItem(kind, summary, urgency, deadline, context)
        with self._lock:
            self._items.append(item)
        print(f"[proactive] queued {item}")
        return item.id

    def pending(self):
        with self._lock:
            return list(self._items)

    def note_user_activity(self):
        """Call whenever Krish speaks or types."""
        self._last_user_activity = time.time()

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
            try:
                self._tick()
            except Exception as e:
                print(f"[proactive] delivery error: {e}")
            self._stop.wait(1.0)

    def _tick(self):
        with self._lock:
            fresh = []
            for item in self._items:
                if item.expired:
                    print(f"[proactive] dropped (expired) {item}")
                else:
                    fresh.append(item)
            self._items = fresh
            if not self._items:
                return
            # most urgent first, then oldest
            self._items.sort(key=lambda i: (-_URGENCY_RANK.get(i.urgency, 1), i.created))
            item = self._items[0]

        decision, reason = self._should_speak(item)
        if decision != "speak":
            return

        with self._lock:
            if item not in self._items:
                return
            self._items.remove(item)

        print(f"[proactive] speaking: {item} ({reason})")
        self._last_spoke = time.time()
        try:
            self._speak_fn(item)
        except Exception as e:
            print(f"[proactive] failed to speak: {e}")

    def _should_speak(self, item):
        """Returns ('speak' | 'wait', reason)."""
        state = self._state_fn()
        idle_for = time.time() - self._last_user_activity
        since_spoke = time.time() - self._last_spoke

        # layer 2: fixed rules
        if state == "speaking":
            return "wait", "already speaking"
        if state == "listening":
            # even critical items wait for him to finish the sentence
            return "wait", "he's mid-utterance"
        if state == "thinking":
            return "wait", "mid-reply"
        if since_spoke < MIN_GAP_BETWEEN_PROACTIVE:
            return "wait", "spoke recently"
        if item.urgency == CRITICAL:
            return "speak", "critical, and he's free"
        if idle_for > QUIET_PERIOD_SECONDS:
            return "speak", f"idle {int(idle_for)}s"

        # layer 3: he's idle, but only just. Ask the model.
        item.attempts += 1
        if self._judge_fn is None:
            return "wait", "no judge; waiting for a quieter moment"

        situation = {
            "state": state,
            "seconds_since_he_spoke": round(idle_for, 1),
            "seconds_since_last_proactive": round(since_spoke, 1),
            "item_age_seconds": round(item.age, 1),
            "times_considered": item.attempts,
            "other_pending": len(self.pending()) - 1,
        }
        try:
            speak = self._judge_fn(item, situation)
        except Exception as e:
            print(f"[proactive] judge failed ({e}); holding")
            return "wait", "judge error"
        return ("speak", "judged appropriate") if speak else ("wait", "judged too soon")
