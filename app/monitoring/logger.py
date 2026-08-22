"""SQLite experiment logger (PLAN 15.11) with privacy gating (PLAN 16.7).

Policy behaviour:
  public     -> full prompt + response + metrics logged
  private    -> prompt/response replaced by hashes; metrics logged
  restricted -> aggregate metrics only (no hashes, no metadata)
  ephemeral  -> nothing written to disk at all
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from app.config import settings
from app.schemas import InferenceResult, SystemState

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT, prompt_hash TEXT, policy TEXT,
    route TEXT, reason TEXT, latency_ms REAL, ttft_ms REAL,
    tokens_per_second REAL, completion_tokens INTEGER,
    free_ram_mb REAL, cpu_percent REAL, temperature_c REAL,
    network_available INTEGER, quality_score REAL, metadata_json TEXT,
    prompt TEXT, response TEXT
);"""


def _connect() -> sqlite3.Connection:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.database_path)
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_request(
    prompt: str,
    policy: str,
    route: str,
    reason: str,
    result: InferenceResult,
    state: SystemState,
    quality_score: float | None = None,
    metadata: dict | None = None,
):
    """Record one routed request; content stored per privacy policy."""
    if policy == "ephemeral":
        return

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    store_prompt = prompt if policy == "public" else None
    store_response = result.text if policy == "public" else None
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata and policy == "public" else None

    con = _connect()
    try:
        con.execute(
            """INSERT INTO requests
            (timestamp, prompt_hash, policy, route, reason, latency_ms, ttft_ms,
             tokens_per_second, completion_tokens, free_ram_mb, cpu_percent,
             temperature_c, network_available, quality_score, metadata_json,
             prompt, response)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), prompt_hash, policy, route, reason,
                result.latency_ms, result.ttft_ms, result.tokens_per_second,
                result.completion_tokens,
                state.free_ram_mb, state.cpu_percent, state.cpu_temperature_c,
                int(state.network_available), quality_score, metadata_json,
                store_prompt, store_response,
            ),
        )
        con.commit()
    finally:
        con.close()


def recent_requests(limit: int = 20, include_text: bool = False) -> list[dict]:
    cols = [
        "timestamp", "prompt_hash", "policy", "route", "reason", "latency_ms",
        "ttft_ms", "tokens_per_second", "completion_tokens", "free_ram_mb",
        "cpu_percent", "temperature_c", "network_available",
    ]
    if include_text:
        cols += ["prompt", "response"]
    con = _connect()
    try:
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM requests ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def summary_stats() -> dict:
    con = _connect()
    try:
        row = con.execute(
            "SELECT COUNT(*), AVG(latency_ms), AVG(tokens_per_second), "
            "AVG(ttft_ms) FROM requests"
        ).fetchone()
        by_route = con.execute(
            "SELECT route, COUNT(*) FROM requests GROUP BY route"
        ).fetchall()
        return {
            "total": row[0] or 0,
            "avg_latency_ms": round(row[1], 1) if row[1] else None,
            "avg_tokens_per_second": round(row[2], 1) if row[2] else None,
            "avg_ttft_ms": round(row[3], 1) if row[3] else None,
            "by_route": {r: c for r, c in by_route},
        }
    finally:
        con.close()
