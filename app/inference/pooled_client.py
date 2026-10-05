"""Pooled bridge client — peer-to-peer LLM inference from browser tabs.

`pooled` (https://github.com/Nehanth/pooled) runs one open model across the
GPUs of every device in a *room*; each tab holds a slice of the layers and the
hidden state hops between tabs over WebRTC. A room can serve its model to any
OpenAI-compatible client with:

    npx @pooled/cli serve "https://pooled.run/r/<code>#k=..."   # 127.0.0.1:8080
      -> OpenAI     http://127.0.0.1:8080/v1   (chat/completions, responses)
      -> Anthropic  http://127.0.0.1:8080      (messages)

CortexEdge treats that bridge as the **`pooled` route**: a route that is
*remote* (like cloud) for privacy purposes, but that runs an open model on
hardware you and your peers own, with no per-token cost. Capabilities borrowed
from the room, per PLAN §16.1:

  * big-model quality (27B/35B MoE) without local RAM — quality like cloud
  * tool calling + Code mode (the room's agent loop) — see `run_pooled_tools`
  * speculative decoding + batched prefill on the room's GPUs (latency)

Everything here is plain OpenAI Chat Completions over one reused HTTP
connection, so the same client also works against llama.cpp, vLLM, LiteLLM or
any other OpenAI-compatible endpoint (used by the unit tests).
"""

from __future__ import annotations

import json
import time
from typing import Iterator, cast

import requests
from requests.adapters import HTTPAdapter

from app.config import settings
from app.schemas import InferenceResult, Route


class PooledUnavailable(RuntimeError):
    """The bridge is not reachable (not started, room closed, token wrong)."""


class _Breaker:
    """Same shape as the cloud circuit breaker: skip a dead bridge fast."""

    def __init__(self, max_failures: int = 3, cooldown_s: int = 60):
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at >= self.cooldown_s:
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.max_failures:
            self.opened_at = time.time()

    def record_success(self):
        self.failures = 0
        self.opened_at = None


breaker = _Breaker()

_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """One pooled connection for the process: keeps the TCP connection (and
    the bridge's own upstream WebRTC state) warm across requests."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _SESSION = s
    return _SESSION


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if settings.pooled_api_key:
        h["Authorization"] = f"Bearer {settings.pooled_api_key}"
    return h


def base_url() -> str:
    return settings.pooled_base_url.rstrip("/")


def chat_url() -> str:
    return f"{base_url()}/chat/completions"


def status() -> dict:
    """Probe the bridge. Never raises; returns a JSON-able dict for /health."""
    info: dict = {
        "enabled": bool(settings.pooled_enabled),
        "base_url": base_url(),
        "model": settings.pooled_model,
        "room_link": bool(settings.pooled_room_link),
        "reachable": False,
        "models": [],
        "breaker_open": breaker.open,
    }
    if not settings.pooled_enabled:
        return info
    try:
        r = session().get(
            f"{base_url()}/models",
            headers=_headers(),
            timeout=settings.pooled_probe_timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        info["models"] = [m.get("id") for m in data.get("data", [])]
        info["reachable"] = True
    except Exception as exc:  # not started / room gone / wrong token
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def available() -> bool:
    """True when the pooled route can be offered to the router."""
    if not settings.pooled_enabled or breaker.open:
        return False
    return bool(status()["reachable"])


def _payload(
    messages: list[dict],
    max_tokens: int,
    stream: bool,
    tools: list[dict] | None = None,
) -> dict:
    body: dict = {
        "model": settings.pooled_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    return body


def run_pooled(
    prompt: str,
    max_tokens: int = 256,
    tools: list[dict] | None = None,
) -> InferenceResult:
    """Non-streaming pooled completion (benchmarks + /query)."""
    if not settings.pooled_enabled:
        raise PooledUnavailable("Pooled route is disabled (POOLED_ENABLED=0)")
    if breaker.open:
        raise PooledUnavailable("Pooled circuit breaker open (cooldown)")

    start = time.perf_counter()
    try:
        r = session().post(
            chat_url(),
            json=_payload([{"role": "user", "content": prompt}], max_tokens, False, tools),
            headers=_headers(),
            timeout=settings.pooled_timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        breaker.record_success()
    except Exception as exc:
        breaker.record_failure()
        raise PooledUnavailable(f"Pooled inference failed: {exc}") from exc

    message = data["choices"][0]["message"]
    text = message.get("content") or ""
    usage = data.get("usage", {}) or {}
    elapsed_ms = (time.perf_counter() - start) * 1000
    completion = usage.get("completion_tokens")
    metadata = {
        "backend": "pooled",
        "model": data.get("model", settings.pooled_model),
        "room": bool(settings.pooled_room_link),
    }
    if message.get("tool_calls"):
        metadata["tool_calls"] = message["tool_calls"]
    return InferenceResult(
        text=text,
        route=cast(Route, "pooled"),
        latency_ms=elapsed_ms,
        ttft_ms=None,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=completion,
        tokens_per_second=(
            round(completion / (elapsed_ms / 1000), 1)
            if completion and elapsed_ms
            else None
        ),
        metadata=metadata,
    )


def stream_pooled(
    messages: list[dict],
    max_tokens: int = 1024,
    tools: list[dict] | None = None,
) -> Iterator[dict]:
    """Stream a pooled completion.

    Yields {"delta": str, "reasoning": str|None} per chunk, then one final
    {"done": True, text, prompt_tokens, completion_tokens, ttft_ms,
    latency_ms, tool_calls}. Same contract as
    app.inference.llm_client.stream_chat, so /query/stream can use either.
    """
    if not settings.pooled_enabled:
        raise PooledUnavailable("Pooled route is disabled (POOLED_ENABLED=0)")
    if breaker.open:
        raise PooledUnavailable("Pooled circuit breaker open (cooldown)")

    start = time.perf_counter()
    ttft_ms: float | None = None
    text = ""
    reasoning = ""
    tool_calls: list[dict] = []
    prompt_tokens = completion_tokens = 0
    error: str | None = None

    try:
        r = session().post(
            chat_url(),
            json=_payload(messages, max_tokens, True, tools),
            headers=_headers(),
            timeout=settings.pooled_timeout_s,
            stream=True,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                prompt_tokens = chunk["usage"].get("prompt_tokens") or prompt_tokens
                completion_tokens = chunk["usage"].get("completion_tokens") or completion_tokens
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta_obj = choices[0].get("delta") or {}
            delta = delta_obj.get("content") or ""
            think = delta_obj.get("reasoning_content") or ""
            if (delta or think) and ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000
            for call in delta_obj.get("tool_calls") or []:
                tool_calls.append(call)
            if think:
                reasoning += think
                yield {"delta": "", "reasoning": think}
            if delta:
                text += delta
                yield {"delta": delta}
        breaker.record_success()
    except Exception as exc:
        breaker.record_failure()
        error = f"{type(exc).__name__}: {exc}"

    done: dict = {
        "done": True,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": round(ttft_ms or 0, 1),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    if not completion_tokens and text:
        # Some OpenAI-compatible bridges omit `usage` on the stream (the real
        # pooled bridge does send it). Flag the estimate instead of reporting
        # a misleading 0 tokens.
        done["completion_tokens"] = max(1, len(text) // 4)
        done["usage_estimated"] = True
    if reasoning:
        done["reasoning"] = reasoning
    if tool_calls:
        done["tool_calls"] = tool_calls
    if error is not None:
        done["error"] = error
    yield done


# --- borrowed capability: the room's agent loop (Code mode) ----------------

DEFAULT_CODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the room's Code-mode workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_pooled_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 1024,
) -> dict:
    """One agent step through the bridge's tool-calling path.

    Returns the raw assistant message: {content, tool_calls, reasoning_content}.
    The caller executes the tools and appends results, exactly as with any
    OpenAI-compatible agent loop. Pooled grammar-constrains calls to the
    declared schema, so calls always parse (down to the 1.7B room model).
    """
    if not settings.pooled_enabled:
        raise PooledUnavailable("Pooled route is disabled (POOLED_ENABLED=0)")
    r = None
    try:
        r = session().post(
            chat_url(),
            json=_payload(
                messages,
                max_tokens,
                False,
                tools if tools is not None else DEFAULT_CODE_TOOLS,
            ),
            headers=_headers(),
            timeout=settings.pooled_timeout_s,
        )
        r.raise_for_status()
        breaker.record_success()
        return r.json()["choices"][0]["message"]
    except Exception as exc:
        breaker.record_failure()
        detail = f" ({r.text[:200]})" if r is not None else ""
        raise PooledUnavailable(f"Pooled tool call failed: {exc}{detail}") from exc
