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
"""

import json
import time
from typing import cast

import requests

from app.config import settings
from app.schemas import InferenceResult, Route

LOCAL_MODELS = {
    "small": settings.model_small,
    "medium": settings.model_medium,
    "moe": settings.model_moe,
    "large": settings.model_large,
}


def local_model_name(route: str) -> str:
    if route in LOCAL_MODELS:
        return LOCAL_MODELS[route]
    if route in {"local_rag", "cache"}:
        return LOCAL_MODELS[settings.rag_generation_route]
    return route


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
    r = requests.post(url, json=payload, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


def run_local(route: str, prompt: str, max_tokens: int = 256) -> InferenceResult:
    """Non-streaming local inference (used by benchmarks)."""
    start = time.perf_counter()
    if settings.backend == "ollama":
        r = _post_json(
            f"{settings.ollama_api_url}/api/chat",
            {
                "model": local_model_name(route),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.2},
            },
            settings.request_timeout_s,
        )
        data = r.json()
        text = data["message"]["content"]
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        metadata = {"backend": "ollama", "model": local_model_name(route)}
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
                f"{settings.ollama_api_url}/api/chat",
                {
                    "model": local_model_name(route),
                    "messages": messages,
                    "stream": True,
                    "options": {"num_predict": max_tokens, "temperature": 0.2},
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
            prompt_tokens=usage.get("prompt_tokens"),            completion_tokens=completion,
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
