"""Session memory with rolling summary compaction (ported from MVP store.py).

SQLite-backed. SessionStore keeps per-session history and compacts old
turns into a rolling summary so the model "remembers" without unbounded
context. Tables live in the same experiments/cortexedge.db as the
logger's `requests` table.
"""

import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, created_ts TEXT, updated_ts TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, role TEXT, content TEXT, ts TEXT
);
"""


def _conn() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.database_path)
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    def __init__(self, summarize_fn=None, compact_every: int = 4, keep_raw: int = 4):
        self.summarize_fn = summarize_fn
        self.compact_every = compact_every
        self.keep_raw = keep_raw

    def get_or_create(self, session_id: str | None = None) -> str:
        session_id = session_id or f"ses-{uuid.uuid4().hex[:12]}"
        con = _conn()
        try:
            row = con.execute(
                "SELECT id FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO sessions (id, created_ts, updated_ts, summary) "
                    "VALUES (?,?,?,?)",
                    (session_id, _now(), _now(), ""),
                )
                con.commit()
        finally:
            con.close()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        con = _conn()
        try:
            con.execute(
                "INSERT INTO messages (session_id, role, content, ts) VALUES (?,?,?,?)",
                (session_id, role, content, _now()),
            )
            con.execute(
                "UPDATE sessions SET updated_ts=? WHERE id=?", (_now(), session_id)
            )
            con.commit()
        finally:
            con.close()

    def user_turn_count(self, session_id: str) -> int:
        con = _conn()
        try:
            return con.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'",
                (session_id,),
            ).fetchone()[0]
        finally:
            con.close()

    def get_summary(self, session_id: str) -> str:
        con = _conn()
        try:
            row = con.execute(
                "SELECT summary FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            return row[0] or "" if row else ""
        finally:
            con.close()

    def set_summary(self, session_id: str, summary: str):
        con = _conn()
        try:
            con.execute(
                "UPDATE sessions SET summary=?, updated_ts=? WHERE id=?",
                (summary, _now(), session_id),
            )
            con.commit()
        finally:
            con.close()

    def last_messages(self, session_id: str, limit: int) -> list[dict]:
        con = _conn()
        try:
            rows = con.execute(
                "SELECT role, content FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
        finally:
            con.close()

    def maybe_compact(self, session_id: str) -> str:
        if self.summarize_fn is None:
            return self.get_summary(session_id)
        if self.user_turn_count(session_id) < self.compact_every:
            return self.get_summary(session_id)
        history = self.last_messages(session_id, limit=1000)
        text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        summary = self.summarize_fn(text)
        self.set_summary(session_id, summary)
        return summary

    def build_context(self, session_id: str) -> tuple[list[dict], str]:
        summary = self.maybe_compact(session_id)
        messages = []
        if summary:
            messages.append({"role": "system", "content": f"Prior context: {summary}"})
        messages.extend(self.last_messages(session_id, limit=self.keep_raw * 2))
        return messages, summary

    def clear(self, session_id: str):
        con = _conn()
        try:
            con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            con.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            con.commit()
        finally:
            con.close()