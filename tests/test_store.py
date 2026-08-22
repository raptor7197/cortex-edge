"""Session memory tests (ported from MVP test_core.py)."""

from app.memory.session_store import SessionStore


def test_session_context_with_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.memory.session_store.settings.database_path",
        tmp_path / "test.db",
    )
    calls = {"n": 0}

    def fake_summarize(text):
        calls["n"] += 1
        return "SUMMARY"

    ss = SessionStore(summarize_fn=fake_summarize, compact_every=2, keep_raw=2)
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


def test_get_or_create_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.memory.session_store.settings.database_path",
        tmp_path / "db.sqlite",
    )
    ss = SessionStore()
    sid = ss.get_or_create("fixed-id")
    assert ss.get_or_create("fixed-id") == sid


def test_clear_session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.memory.session_store.settings.database_path",
        tmp_path / "db.sqlite",
    )
    ss = SessionStore()
    sid = ss.get_or_create()
    ss.add_message(sid, "user", "hi")
    assert ss.user_turn_count(sid) == 1
    ss.clear(sid)
    assert ss.user_turn_count(sid) == 0