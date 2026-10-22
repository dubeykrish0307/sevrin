"""SQLite storage (data/sevrin.db): conversation turns, verified facts and
scheduled reminders. Plain sqlite3, no ORM."""
import json
import os
import sqlite3
import threading
import time

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "sevrin.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    role       TEXT NOT NULL,           -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts    REAL NOT NULL,
    updated_ts    REAL NOT NULL,
    subject       TEXT NOT NULL,        -- e.g. 'tooling', 'schedule'
    fact          TEXT NOT NULL,        -- one sentence
    confidence    REAL NOT NULL,        -- 0..1 after verification
    status        TEXT NOT NULL,        -- 'active' | 'superseded' | 'rejected'
    source_turn   INTEGER,              -- conversations.id
    evidence      TEXT,                 -- the quote that justified it
    checks        TEXT,                 -- JSON, what each verification layer said
    superseded_by INTEGER               -- facts.id of the replacement
);
CREATE INDEX IF NOT EXISTS idx_facts_status  ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);

CREATE TABLE IF NOT EXISTS schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts  REAL NOT NULL,
    subject     TEXT NOT NULL,
    next_due    REAL NOT NULL,          -- epoch seconds
    recurrence  TEXT,                   -- JSON, NULL for one-off
    status      TEXT NOT NULL,          -- 'active' | 'done' | 'cancelled'
    last_fired  REAL,
    source_text TEXT                    -- what was actually said, for debugging
);
CREATE INDEX IF NOT EXISTS idx_sched_due    ON schedules(next_due);
CREATE INDEX IF NOT EXISTS idx_sched_status ON schedules(status);
"""

_local = threading.local()


def _conn():
    # sqlite connections can't be shared across threads, and the voice loop,
    # websocket server and scheduler all run on different ones
    if not hasattr(_local, "conn"):
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        _local.conn = conn
    return _local.conn


def _write(sql: str, params=()) -> int:
    conn = _conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid


def _rows(sql: str, params=()):
    return [dict(r) for r in _conn().execute(sql, params).fetchall()]


# --- conversations ------------------------------------------------------------

def add_turn(role: str, content: str, session_id: str = None) -> int:
    return _write(
        "INSERT INTO conversations (ts, role, content, session_id) VALUES (?, ?, ?, ?)",
        (time.time(), role, content, session_id),
    )


def recent_turns(limit: int = 20):
    """Last `limit` turns, oldest first, in the shape the Messages API wants."""
    rows = _rows("SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?", (limit,))
    return list(reversed(rows))


# --- facts --------------------------------------------------------------------

def add_fact(subject: str, fact: str, confidence: float, evidence: str,
             checks: dict, source_turn: int = None) -> int:
    now = time.time()
    return _write(
        """INSERT INTO facts
           (created_ts, updated_ts, subject, fact, confidence, status, source_turn, evidence, checks)
           VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
        (now, now, subject, fact, confidence, source_turn, evidence, json.dumps(checks)),
    )


def active_facts(limit: int = 200):
    return _rows(
        "SELECT * FROM facts WHERE status = 'active' ORDER BY confidence DESC, updated_ts DESC LIMIT ?",
        (limit,),
    )


def supersede_fact(old_id: int, new_id: int):
    # keep the old row so there's a history of what SEVRIN used to believe
    _write(
        "UPDATE facts SET status = 'superseded', superseded_by = ?, updated_ts = ? WHERE id = ?",
        (new_id, time.time(), old_id),
    )


def stats():
    conn = _conn()
    turns = conn.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"]
    facts = conn.execute("SELECT COUNT(*) n FROM facts WHERE status='active'").fetchone()["n"]
    return {"turns": turns, "active_facts": facts}


# --- schedules ----------------------------------------------------------------

def add_schedule(subject: str, next_due: float, recurrence: dict = None,
                 source_text: str = None) -> int:
    return _write(
        """INSERT INTO schedules (created_ts, subject, next_due, recurrence, status, source_text)
           VALUES (?, ?, ?, ?, 'active', ?)""",
        (time.time(), subject, next_due, json.dumps(recurrence) if recurrence else None, source_text),
    )


def due_schedules(now: float = None):
    now = time.time() if now is None else now
    return _rows(
        "SELECT * FROM schedules WHERE status = 'active' AND next_due <= ? ORDER BY next_due",
        (now,),
    )


def active_schedules():
    return _rows("SELECT * FROM schedules WHERE status = 'active' ORDER BY next_due")


def next_due_time():
    """Epoch time of the soonest active schedule, or None."""
    row = _conn().execute("SELECT MIN(next_due) AS t FROM schedules WHERE status = 'active'").fetchone()
    return row["t"]


def mark_fired(schedule_id: int, next_due: float = None):
    """Pass next_due to move a recurring schedule on; leave it out to close a one-off."""
    if next_due is None:
        _write("UPDATE schedules SET status='done', last_fired=? WHERE id=?", (time.time(), schedule_id))
    else:
        _write("UPDATE schedules SET next_due=?, last_fired=? WHERE id=?",
               (next_due, time.time(), schedule_id))
