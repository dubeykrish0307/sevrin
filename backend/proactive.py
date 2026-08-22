"""
SEVRIN — proactive speech.

Everything SEVRIN wants to say WITHOUT being asked goes through here: a
reminder falling due, a long task finishing, something worth flagging.

THE JUDGEMENT PROBLEM
Speaking up at the wrong moment is worse than not speaking at all. Cutting
across someone mid-sentence to say "your build finished" is the behaviour
that makes assistants annoying enough to switch off. But staying silent
about a meeting starting in two minutes is a real failure too.

So judgement runs in layers, and — deliberately — most decisions never
reach the model:

  Layer 1  EXPIRY     Past its deadline? Drop it. A reminder for a meeting
                      that already started is noise, not help.

  Layer 2  STRUCTURE  Deterministic rules based on what's happening RIGHT
                      NOW. If he's mid-utterance, wait. If nothing is going
                      on, speak. These need no reasoning and no latency, and
                      they're precise because they're mechanical.

  Layer 3  JUDGEMENT  Only the genuinely ambiguous middle — a conversation
                      just ended, or something's been waiting a while — gets
                      an actual decision from the model, with the queue and
                      the current situation as context.

Layer 3 is where "let him judge" lives. Layers 1 and 2 exist so that
judgement is applied to real dilemmas rather than to every trivial case,
which is what keeps it precise.
"""
import queue
import threading
import time
import uuid

# --- urgency levels, set by whatever creates the item ---------------------
# These are structural facts about the item, not opinions, which is why the
# caller supplies them rather than the model inferring them.
CRITICAL = "critical"   # time-sensitive and actionable NOW (meeting in 2 min)
TIMELY = "timely"       # matters soon, but a minute's delay is fine
WHENEVER = "whenever"   # no time pressure at all (a task finished)

_URGENCY_RANK = {CRITICAL: 3, TIMELY: 2, WHENEVER: 1}


class ProactiveItem:
    def __init__(self, kind, summary, urgency=TIMELY, deadline=None, context=None):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind              # 'reminder' | 'task_complete' | 'alert' | 'observation'
        self.summary = summary        # the FACTS; SEVRIN phrases it himself
        self.urgency = urgency
        self.deadline = deadline      # epoch seconds after which it's pointless
        self.context = context or {}
        self.created = time.time()
        self.attempts = 0             # how many times we've considered speaking it

    @property
    def age(self):
        return time.time() - self.created

    @property
    def expired(self):
        return self.deadline is not None and time.time() > self.deadline

    def __repr__(self):
        return f"<{self.kind}:{self.urgency} {self.summary[:40]!r}>"


class ProactiveQueue:
    """Holds pending items and decides when they get spoken.

    The backend owns one of these. It needs two things wired in: a way to
    ask what SEVRIN is currently doing (state_fn) and a way to actually
    speak (speak_fn).
    """

    def __init__(self, state_fn, speak_fn, judge_fn=None):
        self._items = []
        self._lock = threading.Lock()
        self._state_fn = state_fn      # () -> 'idle'|'listening'|'thinking'|'speaking'
        self._speak_fn = speak_fn      # (item) -> None
        self._judge_fn = judge_fn      # (item, situation) -> bool  (the model)
        self._stop = threading.Event()
        self._thread = None
        self._last_spoke = 0.0
        self._last_user_activity = time.time()

    # ---------------- submission ----------------

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
        """Called whenever Krish speaks or types, so the queue knows how
        recently he was engaged."""
        self._last_user_activity = time.time()

    # ---------------- delivery loop ----------------

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
            if not self._items:
                return
            # LAYER 1 — expiry. Drop anything that's no longer useful.
            fresh = []
            for it in self._items:
                if it.expired:
                    print(f"[proactive] dropped (expired) {it}")
                else:
                    fresh.append(it)
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
            if item in self._items:
                self._items.remove(item)
            else:
                return   # something else took it

        print(f"[proactive] speaking: {item} ({reason})")
        self._last_spoke = time.time()
        try:
            self._speak_fn(item)
        except Exception as e:
            print(f"[proactive] failed to speak: {e}")

    # ---------------- the judgement ----------------

    def _should_speak(self, item):
        """Returns (decision, reason) where decision is 'speak' | 'wait'."""
        state = self._state_fn()
        idle_for = time.time() - self._last_user_activity
        since_spoke = time.time() - self._last_spoke

        # --- LAYER 2: structural rules, no model involved ---

        # Never talk over himself.
        if state == "speaking":
            return "wait", "already speaking"

        # Never cut across Krish mid-utterance. Even critical items wait the
        # couple of seconds it takes him to finish a sentence — interrupting
        # a person mid-word is never the right call.
        if state == "listening":
            return "wait", "he's mid-utterance"

        # A reply is being generated; let it land first.
        if state == "thinking":
            return "wait", "mid-reply"

        # Don't stack proactive remarks back to back.
        if since_spoke < MIN_GAP_BETWEEN_PROACTIVE:
            return "wait", "spoke recently"

        # From here he's idle. Critical items go immediately — that's what
        # critical means, and it's a structural fact set at submission, not
        # something to deliberate over.
        if item.urgency == CRITICAL:
            return "speak", "critical, and he's free"

        # Genuinely quiet for a while: safe for anything.
        if idle_for > QUIET_PERIOD_SECONDS:
            return "speak", f"idle {int(idle_for)}s"

        # --- LAYER 3: the ambiguous middle goes to the model ---
        # He's idle but only just — a conversation may have only now ended.
        # This is where a real judgement call is warranted.
        item.attempts += 1
        if self._judge_fn is None:
            # No model available: be conservative and wait for genuine quiet.
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


# How long to leave between proactive remarks, so they don't pile up.
MIN_GAP_BETWEEN_PROACTIVE = 20.0
# After this long with no interaction, he's clearly free.
QUIET_PERIOD_SECONDS = 25.0
