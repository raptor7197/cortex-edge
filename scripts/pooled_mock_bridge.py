#!/usr/bin/env python3
"""Stand-in for the `@pooled/cli serve` bridge.

The real bridge joins a browser room and re-exposes the room's model over
OpenAI Chat Completions. A room needs other people's browser tabs, so for
local testing this stand-in speaks exactly the same surface and forwards
to a local Ollama model instead:

    /v1/models              model list (so the pooled route becomes visible)
    /v1/chat/completions    non-streaming + SSE streaming + tool calls

CortexEdge cannot tell the difference, which is what makes it a useful
harness for the `pooled` route: start this, then

    POOLED_BASE_URL=http://127.0.0.1:8080/v1 ... uvicorn app.api.server:app

and the router will offer `pooled` as a real route.

    python scripts/pooled_mock_bridge.py --port 8080 --upstream-model gemma2:2b
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OLLAMA = "http://127.0.0.1:11434"


def upstream_base() -> str:
    from app.inference import llm_client

    return llm_client.ollama_base()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_model = "gemma2:2b"

    def log_message(self, *args):
        pass

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send_json({
                "object": "list",
                "data": [{"id": "pooled", "object": "model", "owned_by": "pooled-mock"}],
            })
        elif self.path.rstrip("/").endswith("/health"):
            self._send_json({"status": "ok", "mock": True})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        upstream = f"{upstream_base()}/v1/chat/completions"
        body = {**body, "model": self.upstream_model}
        try:
            if body.get("stream"):
                r = requests.post(upstream, json=body, stream=True, timeout=600)
                self.send_response(r.status_code)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                if r.status_code != 200:
                    self.wfile.write(r.content)
                    return
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                return
            r = requests.post(upstream, json=body, timeout=600)
            self._send_json(r.json(), r.status_code)
        except Exception as exc:
            self._send_json({"error": {"message": str(exc), "type": "mock_bridge"}}, 502)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream-model", default="gemma2:2b")
    args = ap.parse_args()
    Handler.upstream_model = args.upstream_model
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"pooled mock bridge on http://{args.host}:{args.port}/v1 "
        f"-> {upstream_base()} ({args.upstream_model})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
