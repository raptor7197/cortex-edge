"""Tests for BLUR routing, store, and search modules.

Run: pytest mvp/tests/ -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import blur
import store


class TestBlur:
    def test_select_low_budget_fast_route(self):
        blur.set_enabled({"small", "medium", "moe"})
        route, reason = blur.select_route(latency_budget_s=0.8, quality_priority=0.5)
        assert route == "small"
        assert "BLUR" in reason

    def test_select_high_quality_priority(self):
        blur.set_enabled({"small", "medium", "moe"})
        route, _ = blur.select_route(latency_budget_s=5.0, quality_priority=0.9)
        assert route == "medium"

    def test_fallback_when_none_feasible(self):
        blur.set_enabled({"small", "medium"})
        route, _ = blur.select_route(latency_budget_s=0.5, quality_priority=0.0)
        assert route in {"small", "medium"}

    def test_no_routes_raises(self):
        blur.set_enabled(set())
        with pytest.raises(RuntimeError):
            blur.select_route()


class TestStore:
    def test_log_and_read_response(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
        store.log_response(
            session_id="s1", route="small", model="m",
            prompt="hello", response="hi",
            latency_ms=10, ttft_ms=5, tokens_per_second=2.0,
            completion_tokens=1, policy="blur:small",
        )
        rows = store.recent_responses(limit=5)
        assert len(rows) == 1
        assert rows[0]["route"] == "small"
        assert rows[0]["prompt"] == "hello"

    def test_session_context_with_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
        calls = {"n": 0}

        def fake_summarize(text):
            calls["n"] += 1
            return "SUMMARY"

        ss = store.SessionStore(summarize_fn=fake_summarize, compact_every=2, keep_raw=2)
        sid = ss.get_or_create()
        ss.add_message(sid, "user", "q1")
        ss.add_message(sid, "assistant", "a1")
        messages, summary = ss.build_context(sid)
        assert calls["n"] == 0  # not enough turns yet
        ss.add_message(sid, "user", "q2")
        ss.add_message(sid, "assistant", "a2")
        messages, summary = ss.build_context(sid)
        assert calls["n"] == 1  # compacted after 2 user turns
        assert summary == "SUMMARY"
        assert messages[0] == {"role": "system", "content": "Prior context: SUMMARY"}
