"""Pooled bridge client tests.

These run a real HTTP server (stdlib) that speaks the OpenAI Chat
Completions shape the `@pooled/cli serve` bridge speaks, so the client
code path — connection reuse, SSE parsing, tool calls, the circuit
breaker — is exercised for real instead of being mocked out.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.config import settings
from app.inference import pooled_client


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "pooled", "object": "model"}]})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
            ]
            for c in chunks:
                data = f"data: {json.dumps(c)}\n\n".encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if payload.get("tools"):
            self._json({
                "model": "pooled",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write_file",
                                         "arguments": '{"path":"a.py","content":"x"}'},
                        }],
                    }
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 5},
            })
            return
        self._json({
            "model": "pooled",
            "choices": [{"message": {"role": "assistant", "content": "Hello from the room"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        })


@pytest.fixture()
def bridge():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_base = settings.pooled_base_url
    old_enabled = settings.pooled_enabled
    settings.pooled_base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    settings.pooled_enabled = True
    pooled_client.breaker.record_success()  # reset breaker state
    yield settings.pooled_base_url
    settings.pooled_base_url = old_base
    settings.pooled_enabled = old_enabled
    pooled_client.breaker.record_success()
    server.shutdown()
    server.server_close()


def test_status_reports_reachable(bridge):
    info = pooled_client.status()
    assert info["reachable"] is True
    assert info["models"] == ["pooled"]
    assert pooled_client.available() is True


def test_run_pooled_returns_inference_result(bridge):
    result = pooled_client.run_pooled("hi", max_tokens=16)
    assert result.text == "Hello from the room"
    assert result.route == "pooled"
    assert result.prompt_tokens == 9
    assert result.completion_tokens == 4
    assert result.tokens_per_second is not None
    assert result.metadata["backend"] == "pooled"


def test_stream_pooled_yields_deltas_then_done(bridge):
    events = list(pooled_client.stream_pooled([{"role": "user", "content": "hi"}]))
    deltas = [e["delta"] for e in events if "delta" in e]
    assert "".join(deltas) == "Hello"
    final = events[-1]
    assert final["done"] is True
    assert final["text"] == "Hello"
    assert final["completion_tokens"] == 2
    assert "error" not in final


def test_run_pooled_tools_returns_tool_calls(bridge):
    message = pooled_client.run_pooled_tools([{"role": "user", "content": "write a.py"}])
    assert message["tool_calls"][0]["function"]["name"] == "write_file"


def test_unreachable_bridge_is_reported_not_raised():
    old_port = settings.pooled_base_url
    old_enabled = settings.pooled_enabled
    settings.pooled_base_url = "http://127.0.0.1:9/v1"  # discard port
    settings.pooled_enabled = True
    pooled_client.breaker.record_success()
    info = pooled_client.status()
    assert info["reachable"] is False
    assert "error" in info
    with pytest.raises(pooled_client.PooledUnavailable):
        pooled_client.run_pooled("hi", max_tokens=4)
    settings.pooled_base_url = old_port
    settings.pooled_enabled = old_enabled
    pooled_client.breaker.record_success()


def test_disabled_bridge_refuses(bridge):
    settings.pooled_enabled = False
    assert pooled_client.available() is False
    with pytest.raises(pooled_client.PooledUnavailable):
        pooled_client.run_pooled("hi", max_tokens=4)
    settings.pooled_enabled = True
