"""Local + cloud inference clients (PLAN 15.8, improved per 16.2).

Two local backends:
  - "ollama": Ollama native /api/chat — reliable token counts and
    simple streaming (NDJSON). Default on dev machines.
  - "openai": OpenAI-compatible endpoints (llama.cpp llama-server on
    ports 8101/8102 as per PLAN) — the Raspberry Pi deployment path.

Streaming path measures TTFT (time-to-first-token) which the
non-streaming endpoint cannot report (PLAN 16.2). The cloud client has
a circuit breaker: after N consecutive failures the route is disabled
for a cooldown period (PLAN Phase 8).

Latency work (docs/LOCAL_LATENCY.md): one pooled HTTP session for the
whole process, `keep_alive=-1` so the model is never unloaded between
requests (a cold load is 4-6 s on CPU, dwarfing generation), a trimmed
KV cache (`num_ctx`), explicit thread count and startup warm-up.

The peer-to-peer `pooled` route lives in app/inference/pooled_client.py.
"""

import json
import os
import socket
import struct
import time
from typing import cast

import requests
from requests.adapters import HTTPAdapter

from app.config import keep_alive_value, settings
from app.schemas import InferenceResult, Route

LOCAL_MODELS = {
    "small": settings.model_small,
    "medium": settings.model_medium,
    "moe": settings.model_moe,
    "large": settings.model_large,
}

_SESSION: requests.Session | None = None
_OLLAMA_BASE: str | None = None


def session() -> requests.Session:
    """Process-wide HTTP session (keep-alive: no TCP handshake per request)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _SESSION = s
    return _SESSION


def default_gateway() -> str | None:
    """Default gateway of this container/host (the machine running Ollama)."""
    try:
        with open("/proc/net/route", encoding="utf-8") as fh:
            for line in fh.readlines()[1:]:
                fields = line.split()
                if len(fields) > 2 and fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except Exception:
        pass
    env_gw = os.environ.get("HERMES_HOST_IP") or os.environ.get("OLLAMA_FALLBACK_HOST")
    return env_gw or None


def ollama_base() -> str:
    """Resolve the Ollama base URL once.

    `OLLAMA_API_URL=auto` probes, in order: $OLLAMA_HOST, localhost:11434,
    the default gateway on :11434 (the dev host when CortexEdge runs in a
    container). The first one that answers /api/tags wins.
    """
    global _OLLAMA_BASE
    if _OLLAMA_BASE is not None:
        return _OLLAMA_BASE

    configured = (settings.ollama_api_url or "").strip()
    if configured and configured.lower() != "auto":
        _OLLAMA_BASE = configured.rstrip("/")
        return _OLLAMA_BASE

    candidates: list[str] = []
    env_host = os.environ.get("OLLAMA_HOST", "").strip()
    if env_host and "://" in env_host:
        candidates.append(env_host.rstrip("/"))
    candidates += ["http://127.0.0.1:11434", "http://localhost:11434"]
    gw = default_gateway()
    if gw:
        candidates.append(f"http://{gw}:11434")

    for cand in candidates:
        try:
            r = session().get(f"{cand}/api/tags", timeout=1.5)
            if r.status_code == 200:
                _OLLAMA_BASE = cand
                return _OLLAMA_BASE
        except Exception:
            continue
    _OLLAMA_BASE = candidates[-1] if len(candidates) > 1 else "http://127.0.0.1:11434"
    return _OLLAMA_BASE


def local_model_name(route: str) -> str:
    if route in LOCAL_MODELS:
        return LOCAL_MODELS[route]
    if route in {"local_rag", "cache"}:
        return LOCAL_MODELS[settings.rag_generation_route]
    return route


def ollama_options(max_tokens: int) -> dict:
    """Options that cut latency on CPU (docs/LOCAL_LATENCY.md)."""
    opts: dict = {
        "num_predict": max_tokens,
        "temperature": 0.2,
        "num_ctx": settings.ollama_num_ctx,
        "num_batch": settings.ollama_num_batch,
    }
    if settings.ollama_num_thread > 0:
        opts["num_thread"] = settings.ollama_num_thread
    return opts


def ollama_tags(timeout: float = 3.0) -> set[str]:
    """Names of models the backend has locally (empty when unreachable)."""
    try:
        r = session().get(f"{ollama_base()}/api/tags", timeout=timeout)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


def warmup(route: str, timeout: int = 300) -> bool:
    """Preload a tier's model with keep_alive=-1 so the first real request
    does not pay the 4-6 s cold-load cost."""
    if settings.backend != "ollama":
        return False
    try:
        r = session().post(
            f"{ollama_base()}/api/chat",
            json={
                "model": local_model_name(route),
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": keep_alive_value(),
                # same options as a real request, so warm-up leaves the model
                # loaded in exactly the configuration later requests use
                # (a differing num_ctx makes Ollama reload it)
                "options": ollama_options(1),
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def resync_ollama_base() -> str:
    """Re-probe after a backend restart (used by /health?refresh=1)."""
    global _OLLAMA_BASE
    _OLLAMA_BASE = None
    return ollama_base()


class CloudCircuitBreaker:
    """Disables the cloud route after repeated failures (PLAN Phase 8)."""

    def __init__(self, max_failures: int, cooldown_s: int):
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


circuit_breaker = CloudCircuitBreaker(
    settings.cloud_max_failures, settings.cloud_cooldown_s
)


def _post_json(url: str, payload: dict, timeout: int, headers: dict | None = None):
    r = session().post(url, json=payload, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


def run_local(route: str, prompt: str, max_tokens: int = 256) -> InferenceResult:
    """Non-streaming local inference (used by benchmarks)."""
    start = time.perf_counter()
    if settings.backend == "ollama":
        model = local_model_name(route)
        r = _post_json(
            f"{ollama_base()}/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": keep_alive_value(),
                "options": ollama_options(max_tokens),
            },
            settings.request_timeout_s,
        )
        data = r.json()
        text = data["message"]["content"]
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        metadata = {
            "backend": "ollama",
            "model": model,
            "server": ollama_base(),
            "load_ms": round(data.get("load_duration", 0) / 1e6, 1),
            "prompt_eval_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
            "eval_ms": round(data.get("eval_duration", 0) / 1e6, 1),
        }
    else:
        r = _post_json(
            _local_url(route),
            {
                "model": route,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
            },
            settings.request_timeout_s,
        )
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        metadata = {"backend": "llama.cpp", "model": route}

    elapsed_ms = (time.perf_counter() - start) * 1000
    tps = (
        completion_tokens / (elapsed_ms / 1000)
        if completion_tokens and elapsed_ms
        else None
    )
    return InferenceResult(
        text=text,
        route=cast(Route, route),
        latency_ms=elapsed_ms,
        ttft_ms=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=round(tps, 1) if tps is not None else None,
        metadata=metadata,
    )


def _local_url(route: str) -> str:
    if route == "small":
        return settings.small_model_url
    if route == "medium":
        return settings.medium_model_url
    raise ValueError(f"No llama.cpp endpoint configured for route {route}")


def stream_chat(
    route: str,
    messages: list[dict],
    max_tokens: int = 1024,
):
    """Stream tokens from the local backend.

    Yields dicts: {"delta": str} per token, then a final
    {"done": True, "prompt_tokens": n, "completion_tokens": m,
     "ttft_ms": t, "latency_ms": l}.
    """
    start = time.perf_counter()
    ttft_ms = None
    text = ""
    prompt_tokens = completion_tokens = 0
    error: str | None = None
    try:
        if settings.backend == "ollama":
            r = _post_json(
                f"{ollama_base()}/api/chat",
                {
                    "model": local_model_name(route),
                    "messages": messages,
                    "stream": True,
                    "keep_alive": keep_alive_value(),
                    "options": ollama_options(max_tokens),
                },
                timeout=180,
            )
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    text += delta
                    yield {"delta": delta}
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count") or 0
                    completion_tokens = chunk.get("eval_count") or 0
                    break
        else:
            r = _post_json(
                _local_url(route),
                {
                    "model": route,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
                timeout=180,
            )
            for line in r.iter_lines(decode_unicode=True):
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    text += delta
                    yield {"delta": delta}
                usage = data.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens") or 0
                    completion_tokens = usage.get("completion_tokens") or 0
    except Exception as exc:
        # Never swallow the terminal event: flag the error in `done` so
        # callers can distinguish a clean end from an incomplete stream.
        error = f"{type(exc).__name__}: {exc}"
    done_event: dict = {
        "done": True,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": round(ttft_ms or 0, 1),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }
    if error is not None:
        done_event["error"] = error
    yield done_event


def run_cloud(prompt: str, max_tokens: int = 256) -> InferenceResult:
    """Cloud fallback with circuit breaker (OpenAI-compatible endpoint)."""
    if not settings.cloud_model_url or not settings.cloud_api_key:
        raise RuntimeError("Cloud endpoint is not configured")
    if circuit_breaker.open:
        raise RuntimeError("Cloud circuit breaker open (cooldown)")

    start = time.perf_counter()
    try:
        r = _post_json(
            settings.cloud_model_url,
            {
                "model": settings.cloud_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            settings.request_timeout_s,
            headers={"Authorization": f"Bearer {settings.cloud_api_key}"},
        )
        data = r.json()
        usage = data.get("usage", {})
        circuit_breaker.record_success()
        completion = usage.get("completion_tokens")
        elapsed_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(
            text=data["choices"][0]["message"]["content"],
            route="cloud",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion,
            tokens_per_second=(
                round(completion / (elapsed_ms / 1000), 1)
                if completion and elapsed_ms
                else None
            ),
            metadata={"backend": "cloud", "model": settings.cloud_model},
        )
    except Exception as exc:
        circuit_breaker.record_failure()
        raise RuntimeError(f"Cloud inference failed: {exc}") from exc
