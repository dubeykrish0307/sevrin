"""
SEVRIN — persistent storage layer.

SQLite lives in data/sevrin.db. Two things are stored here:

  conversations  — every turn, so SEVRIN no longer forgets everything on
                   restart (previously brain/llm.py kept history in a
                   module-level list that died with the process).

  facts          — durable things learned about Krish, kept SEPARATE from
                   raw chat log. Facts carry a confidence score and a
                   verification status, because a memory system that
                   confidently stores wrong things is worse than one that
                   stores nothing. See brain/extractor.py for how facts get
                   proposed and verified before they land here.

Plain sqlite3 from the stdlib — no ORM, no extra dependency.
"""
import json
import os
import sqlite3
import threading
import time

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "sevrin.db")

_local = threading.local()


def _conn():
    """One connection per thread — the backend runs the voice loop and the
    websocket server on different threads, and sqlite connections are not
    safe to share across them."""
    if not hasattr(_local, "conn"):
        os.makedirs(DB_DIR, exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _init(_local.conn)
    return _local.conn


def _init(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS conversations (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         REAL NOT NULL,
        role       TEXT NOT NULL,           -- 'user' | 'assistant'
        content    TEXT NOT NULL,
        session_id TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts);

    CREATE TABLE IF NOT EXISTS facts (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts   REAL NOT NULL,
        updated_ts   REAL NOT NULL,
        subject      TEXT NOT NULL,         -- what the fact is about, e.g. 'tooling', 'schedule'
        fact         TEXT NOT NULL,         -- the fact itself, one clear sentence
        confidence   REAL NOT NULL,         -- 0..1 after verification
        status       TEXT NOT NULL,         -- 'active' | 'superseded' | 'rejected'
        source_turn  INTEGER,               -- conversations.id it came from
        evidence     TEXT,                  -- the quote/context that justified it
        checks       TEXT,                  -- JSON: what each verification layer concluded
        superseded_by INTEGER               -- facts.id that replaced this one
    );

    CREATE INDEX IF NOT EXISTS idx_facts_status  ON facts(status);
    CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
    """)
    conn.commit()


# ---------------- conversations ----------------

def add_turn(role: str, content: str, session_id: str = None) -> int:
    c = _conn()
    cur = c.execute(
        "INSERT INTO conversations (ts, role, content, session_id) VALUES (?, ?, ?, ?)",
        (time.time(), role, content, session_id),
    )
    c.commit()
    return cur.lastrowid


def recent_turns(limit: int = 20):
    """Most recent turns, oldest-first (ready to hand to the model)."""
    c = _conn()
    rows = c.execute(
        "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def search_turns(keyword: str, limit: int = 10):
    c = _conn()
    rows = c.execute(
        "SELECT role, content, ts FROM conversations WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
        (f"%{keyword}%", limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------- facts ----------------

def add_fact(subject: str, fact: str, confidence: float, evidence: str,
             checks: dict, source_turn: int = None) -> int:
    now = time.time()
    c = _conn()
    cur = c.execute(
        """INSERT INTO facts
           (created_ts, updated_ts, subject, fact, confidence, status, source_turn, evidence, checks)
           VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)""",
        (now, now, subject, fact, confidence, source_turn, evidence, json.dumps(checks)),
    )
    c.commit()
    return cur.lastrowid


def active_facts(limit: int = 200):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM facts WHERE status = 'active' ORDER BY confidence DESC, updated_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def facts_about(subject: str):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM facts WHERE status = 'active' AND subject = ? ORDER BY confidence DESC",
        (subject,),
    ).fetchall()
    return [dict(r) for r in rows]


def supersede_fact(old_id: int, new_id: int):
    """Marks a fact as replaced rather than deleting it — history is useful
    when debugging why SEVRIN believes something."""
    c = _conn()
    c.execute(
        "UPDATE facts SET status = 'superseded', superseded_by = ?, updated_ts = ? WHERE id = ?",
        (new_id, time.time(), old_id),
    )
    c.commit()


def reject_fact(fact_id: int, reason: str = ""):
    c = _conn()
    row = c.execute("SELECT checks FROM facts WHERE id = ?", (fact_id,)).fetchone()
    checks = json.loads(row["checks"]) if row and row["checks"] else {}
    checks["rejected_reason"] = reason
    c.execute(
        "UPDATE facts SET status = 'rejected', updated_ts = ?, checks = ? WHERE id = ?",
        (time.time(), json.dumps(checks), fact_id),
    )
    c.commit()


def delete_fact(fact_id: int):
    c = _conn()
    c.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    c.commit()


def stats():
    c = _conn()
    turns = c.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"]
    facts = c.execute("SELECT COUNT(*) n FROM facts WHERE status='active'").fetchone()["n"]
    return {"turns": turns, "active_facts": facts}
