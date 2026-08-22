"""Privacy-gated logging tests (PLAN 16.7)."""

from typing import cast

from app.monitoring import logger as dblog
from app.schemas import InferenceResult, Route, SystemState


def make_result(route: str = "small"):
    return InferenceResult(
        text="the answer", route=cast(Route, route), latency_ms=10.0,
        ttft_ms=4.0, completion_tokens=20, tokens_per_second=2.0,
    )


def make_state():
    return SystemState(
        free_ram_mb=3000, cpu_percent=10, cpu_temperature_c=50,
        network_available=True,
    )


def test_public_logs_full_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(dblog, "settings", type("S", (), {"database_path": tmp_path / "t.db"}))
    dblog.log_request("secret query", "public", "small", "reason",
                      make_result(), make_state())
    rows = dblog.recent_requests(limit=10, include_text=True)
    assert rows[0]["prompt"] == "secret query"
    assert rows[0]["response"] == "the answer"


def test_private_hashes_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(dblog, "settings", type("S", (), {"database_path": tmp_path / "t.db"}))
    dblog.log_request("my private note", "private", "small", "reason",
                      make_result(), make_state())
    rows = dblog.recent_requests(limit=10, include_text=True)
    assert rows[0]["prompt"] is None
    assert rows[0]["response"] is None
    assert rows[0]["prompt_hash"] != "my private note"


def test_restricted_stores_metrics_only(monkeypatch, tmp_path):
    monkeypatch.setattr(dblog, "settings", type("S", (), {"database_path": tmp_path / "t.db"}))
    dblog.log_request("restricted text", "restricted", "medium", "reason",
                      make_result(), make_state())
    rows = dblog.recent_requests(limit=10, include_text=False)
    assert rows[0]["route"] == "medium"
    assert rows[0]["latency_ms"] == 10.0


def test_ephemeral_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(dblog, "settings", type("S", (), {"database_path": tmp_path / "t.db"}))
    dblog.log_request("ephemeral text", "ephemeral", "small", "reason",
                      make_result(), make_state())
    assert dblog.summary_stats()["total"] == 0