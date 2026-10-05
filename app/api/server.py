"""CortexEdge orchestrator (PLAN 15.12 + MVP streaming/sessions/search).

Pipeline: cache lookup -> system state -> feature extraction -> route
selection (rule | cost | blur | learned) -> execution (small/medium/
RAG/pooled/cloud) with fallback -> privacy-gated logging -> response.

Endpoints:
  GET  /health                    system state + routes + db status
  GET  /routes                    route table with cost estimates
  POST /query                     non-streaming (benchmarks/scripts)
  POST /query/stream              NDJSON streaming with TTFT + tokens
  GET  /pooled/status             pooled (P2P browser) bridge status
  POST /pooled/code               one agent step through a pooled room
"""

import sys
import threading
import time
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.cache.semantic_cache import SemanticCache
from app.config import keep_alive_value, settings
from app.inference import pooled_client
from app.inference.llm_client import (
    LOCAL_MODELS,
    local_model_name,
    ollama_base,
    ollama_tags,
    run_cloud,
    run_local,
    stream_chat,
    warmup,
)
from app.memory.session_store import SessionStore
from app.monitoring import logger as dblog
from app.monitoring.system_state import read_system_state
from app.rag.retriever import Retriever, build_rag_prompt
from app.router import blur, cost, rule_router
from app.router.cost import ROUTE_COSTS, select_min_cost
from app.router.features import extract_features
from app.router.learned_router import LearnedRouter
from app.schemas import InferenceResult, QueryRequest, Route
from app.tools import web_search

LOCAL_TIERS = {"small", "medium", "moe", "large"}
OFFLOAD_TIERS = {"pooled", "cloud"}

# Degradation order when a local tier fails (BUG 6): each route falls back
# to progressively cheaper tiers before an offload route is considered.
LOCAL_FALLBACK_CHAIN: dict[str, list[str]] = {
    "large": ["moe", "medium", "small"],
    "moe": ["medium", "small"],
    "medium": ["small"],
    "small": [],
}
# Preference order for local fallback when an offload route fails.
OFFLOAD_FALLBACK_ORDER = ["large", "moe", "medium", "small"]


def _pooled_usable(policy: str = "public", offline_only: bool = False) -> bool:
    """Peer-to-peer inference leaves the device, so it obeys the same
    policy gate as cloud (PLAN §10)."""
    if not pooled_client.available():
        return False
    return policy == "public" and not offline_only


def _cloud_configured() -> bool:
    return bool(settings.cloud_model_url and settings.cloud_api_key)


def _available_models() -> set[str]:
    """Routes whose Ollama model is actually pulled on the backend."""
    if settings.backend != "ollama":
        return {"small", "medium"}
    tags = ollama_tags()
    return {route for route, model in LOCAL_MODELS.items() if model in tags}


def refresh_enabled() -> set[str]:
    """Routes this machine can serve right now (local tiers + offload)."""
    available = _available_models()
    enabled = set(available) or {"small"}
    if pooled_client.available():
        enabled.add("pooled")
    if _cloud_configured():
        enabled.add("cloud")
    blur.set_enabled(enabled)
    cost.set_enabled(enabled)
    return enabled


def summarize_fn(text: str) -> str:
    """Compaction: the medium model summarizes prior context."""
    try:
        import requests

        payload = {
            "model": settings.model_medium,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a context compactor. Summarize the "
                        "conversation below into a dense paragraph "
                        "preserving key facts, decisions and questions."
                    ),
                },
                {"role": "user", "content": text[:6000]},
            ],
            "stream": False,
            "keep_alive": keep_alive_value(),
            "options": {
                "num_predict": 200,
                "temperature": 0.3,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        if settings.backend == "ollama":
            r = requests.post(
                f"{ollama_base()}/api/chat", json=payload, timeout=60
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        return ""
    except Exception:
        return ""


def _warmup_models(routes: list[str]):
    """Preload tier models so the first real request has no cold-load cost."""
    for route in routes:
        try:
            ok = warmup(route)
            print(f"[warmup] {route} -> {local_model_name(route)}: {'ready' if ok else 'skipped'}",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[warmup] {route} failed: {exc}", file=sys.stderr, flush=True)


app = FastAPI(title="CortexEdge", version="0.3.0")
app.state.enabled_routes = refresh_enabled()
app.state.cache = SemanticCache()
app.state.retriever = None
app.state.sessions = SessionStore(summarize_fn=summarize_fn)
app.state.learned = None
app.state.last_state = None


@app.on_event("startup")
def startup():
    try:
        app.state.retriever = Retriever()
    except Exception as exc:
        print(f"[startup] RAG retriever unavailable: {exc}", file=sys.stderr)
    joblib_path = Path(__file__).resolve().parents[2] / "experiments" / "router.joblib"
    if joblib_path.exists():
        try:
            app.state.learned = LearnedRouter(joblib_path)
        except Exception as exc:
            print(f"[startup] learned router load failed: {exc}", file=sys.stderr)
    if settings.ollama_warmup:
        warm_routes = [
            r.strip()
            for r in settings.ollama_warmup_routes.split(",")
            if r.strip() in LOCAL_TIERS
        ]
        threading.Thread(
            target=_warmup_models,
            args=(sorted(set(warm_routes) & app.state.enabled_routes),),
            daemon=True,
        ).start()


@app.get("/health")
def health(refresh: bool = False):
    if refresh:
        app.state.enabled_routes = refresh_enabled()
    state = read_system_state()
    return {
        "status": "ok",
        "system": state.model_dump(),
        "routes": {
            "small": settings.model_small,
            "medium": settings.model_medium,
            "moe": settings.model_moe,
            "large": settings.model_large,
        },
        "enabled_routes": sorted(app.state.enabled_routes),
        "ollama": {
            "base_url": ollama_base(),
            "reachable": bool(ollama_tags()),
            "keep_alive": settings.ollama_keep_alive,
            "num_ctx": settings.ollama_num_ctx,
        },
        "pooled": pooled_client.status(),
        "cloud": {"configured": _cloud_configured(), "model": settings.cloud_model},
        "db": {"path": str(settings.database_path), "ok": True},
        "learned_router": app.state.learned is not None,
        "rag": app.state.retriever is not None,
    }


@app.get("/routes")
def routes():
    """Route table with the cost estimates the cost router uses."""
    return {
        "enabled": sorted(app.state.enabled_routes),
        "routes": [
            {
                "name": c.name,
                "model": c.model,
                "latency_ms": c.latency_ms,
                "energy_j": c.energy_j,
                "memory_mb": c.memory_mb,
                "cloud_cost_per_1k": c.cloud_cost,
                "privacy_risk": c.privacy_risk,
                "quality": c.quality,
            }
            for c in ROUTE_COSTS
        ],
    }


@app.get("/pooled/status")
def pooled_status():
    return pooled_client.status()


@app.post("/pooled/code")
def pooled_code(payload: dict):
    """One agent step on the room's model (pooled Code mode / tool calling).

    Body: {"messages": [...], "tools": [...] (optional), "max_tokens": 1024}
    Returns the raw assistant message so the caller can run the tool loop.
    """
    if not _pooled_usable():
        raise HTTPException(status_code=503, detail="Pooled room is not available")
    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages[] is required")
    try:
        return pooled_client.run_pooled_tools(
            messages,
            tools=payload.get("tools"),
            max_tokens=int(payload.get("max_tokens", 1024)),
        )
    except pooled_client.PooledUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def effective_policy(req: QueryRequest) -> str:
    if req.policy == "restricted" or req.policy == "ephemeral":
        return req.policy
    if req.private or req.offline_only:
        return "private"
    return req.policy


def recompute_enabled(req: QueryRequest) -> set[str]:
    """Enabled routes restricted by the request's privacy policy."""
    enabled = app.state.enabled_routes
    if effective_policy(req) != "public" or req.offline_only:
        return {r for r in enabled if r in LOCAL_TIERS or r == "local_rag"}
    return enabled


def resolve_route(req: QueryRequest, features, state) -> tuple[Route, str]:
    """Route selection: policy constraints always apply; the chosen
    router (rule | cost | blur | learned) picks the tier inside them."""
    if req.use_documents:
        return "local_rag", "Document-grounded request"

    enabled = recompute_enabled(req)

    if req.model != "auto":
        if req.model not in enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Route '{req.model}' not available on this machine",
            )
        return req.model, f"explicit:{req.model}"

    policy = effective_policy(req)
    local_only = policy in {"private", "restricted", "ephemeral"}

    if req.router == "cost":
        target = req.quality_priority if req.quality_priority > 0 else settings.quality_target
        best = select_min_cost(quality_target=target, enabled=enabled)
        if best.name == "cloud" and local_only:
            best = select_min_cost(
                quality_target=target,
                enabled={r for r in enabled if r in LOCAL_TIERS},
            )
        reason = (
            f"COST: quality_target={target} -> {best.name} "
            f"(latency {best.latency_ms}ms, energy {best.energy_j}J, "
            f"quality {best.quality}, cloud cost ${best.cloud_cost})"
        )
        return cast(Route, best.name), reason

    if req.router == "learned":
        learned = app.state.learned
        if learned is None or not learned.available:
            decision = rule_router.select_route(features, state, req, enabled)
            return decision.route, decision.reason + " (learned router unavailable)"
        predicted = learned.predict(features, state)
        if local_only and predicted in OFFLOAD_TIERS:
            predicted = "medium"
        if predicted not in enabled and predicted != "cloud":
            predicted = "medium"
        return cast(Route, predicted), f"LEARNED: {predicted}"

    if req.router == "blur":
        route, reason = blur.select_route(
            latency_budget_s=req.latency_budget_s,
            quality_priority=req.quality_priority,
        )
        if local_only and route in OFFLOAD_TIERS:
            route, reason = blur.select_route(
                latency_budget_s=req.latency_budget_s, quality_priority=0.0
            )
        return cast(Route, route), reason

    # default: rule router (PLAN 15.6)
    decision = rule_router.select_route(features, state, req, enabled)
    return decision.route, decision.reason


def _run_offload(route: str, prompt: str, req: QueryRequest) -> InferenceResult:
    if route == "pooled":
        tools = pooled_client.DEFAULT_CODE_TOOLS if req.pooled_tools else None
        return pooled_client.run_pooled(
            prompt, max_tokens=req.max_tokens, tools=tools
        )
    return run_cloud(prompt, max_tokens=req.max_tokens)


def execute_route(route: Route, prompt: str, req: QueryRequest) -> InferenceResult:
    """Run one route with the configured fallback chain."""
    try:
        if route == "local_rag":
            retriever = app.state.retriever
            if retriever is None:
                raise RuntimeError("RAG index is not available (run `make ingest`)")
            passages = retriever.search(req.text)
            result = run_local(settings.rag_generation_route, build_rag_prompt(req.text, passages))
            result.route = "local_rag"
            result.metadata["sources"] = [
                {"source": p["source"], "chunk": p["chunk"], "score": p["score"]}
                for p in passages
            ]
            return result
        if route in OFFLOAD_TIERS:
            return _run_offload(route, prompt, req)
        if route in LOCAL_TIERS:
            return run_local(route, prompt, max_tokens=req.max_tokens)
        raise RuntimeError(f"Unsupported route: {route}")
    except Exception as exc:
        # Full cascade: walk local tiers downward; an offload route only
        # when policy and availability allow it; 503 with the original
        # error otherwise.
        policy = effective_policy(req)
        local_only = policy in {"private", "restricted", "ephemeral"}
        enabled = recompute_enabled(req)
        network = state_has_network() if not local_only else False

        generation_tier = (
            settings.rag_generation_route if route == "local_rag" else route
        )
        seen: set[str] = set()
        chain: list[str] = []
        if route in OFFLOAD_TIERS:
            # The chosen offload route failed -> the other offload route
            # first, then every local tier from best to cheapest.
            chain.extend(t for t in ("pooled", "cloud") if t != route)
            chain.extend(t for t in OFFLOAD_FALLBACK_ORDER if t in LOCAL_TIERS)
        elif route == "local_rag":
            chain.append(generation_tier)  # answer without RAG context
            chain.extend(LOCAL_FALLBACK_CHAIN.get(generation_tier, []))
        else:
            chain.extend(LOCAL_FALLBACK_CHAIN.get(route, []))

        for tier in chain:
            if tier in seen:
                continue
            seen.add(tier)
            if tier in OFFLOAD_TIERS:
                if local_only or not network or tier not in enabled:
                    continue
                try:
                    return _run_offload(tier, prompt, req)
                except Exception as offload_exc:
                    exc = offload_exc
                continue
            if tier not in LOCAL_TIERS:
                continue
            try:
                return run_local(tier, prompt, max_tokens=req.max_tokens)
            except Exception as tier_exc:
                exc = tier_exc

        raise HTTPException(status_code=503, detail=str(exc))


def state_has_network() -> bool:
    return read_system_state().network_available


def build_messages(req: QueryRequest, route: Route, policy: str) -> tuple[list[dict], str | None]:
    """Assemble chat messages: session context + optional web context."""
    session_id = None
    extra = ""
    if req.use_search and policy == "public" and not req.offline_only:
        try:
            results = web_search.search_web(req.text)
            extra = web_search.build_search_context(results)
        except Exception:
            extra = ""

    if policy in {"restricted", "ephemeral"}:
        messages = [{"role": "user", "content": req.text}]
        if extra:
            messages.append({"role": "system", "content": extra})
        return messages, None

    sessions: SessionStore = app.state.sessions
    session_id = req.session_id or sessions.get_or_create()

    messages, _ = sessions.build_context(session_id)
    # Drop trailing unanswered user turns left by earlier failed requests so
    # the context never contains duplicate/stale user messages.
    while messages and messages[-1]["role"] == "user":
        messages.pop()
    messages.append({"role": "user", "content": req.text})
    if extra:
        messages.append({"role": "system", "content": extra})

    # Persist exactly once, for explicit AND auto-created sessions.
    try:
        sessions.add_message(session_id, "user", req.text)
    except Exception:
        pass  # logging/storage failure must not break inference
    return messages, session_id


def _log(req: QueryRequest, policy: str, route: Route, reason: str,
         result: InferenceResult, session_id: str | None):
    dblog.log_request(
        prompt=req.text,
        policy=policy,
        route=route,
        reason=reason,
        result=result,
        state=read_system_state(),
        metadata={
            "session_id": session_id,
            "use_search": req.use_search,
            "router": req.router,
            **result.metadata,
        },
    )


@app.get("/db/stats")
def db_stats():
    return dblog.summary_stats()


@app.post("/query")
def query(req: QueryRequest):
    """Non-streaming query (used by benchmarks and scripts)."""
    state = read_system_state()
    policy = effective_policy(req)

    cache: SemanticCache = app.state.cache
    cached = cache.lookup(
        req.text, policy, req.private, req.offline_only,
        route=req.router,
    )
    if cached is not None:
        return {
            "route": "cache",
            "response": cached,
            "latency_ms": 0.0,
            "system": state.model_dump(),
        }

    features = extract_features(req.text, req.use_documents)
    route, reason = resolve_route(req, features, state)
    messages, session_id = build_messages(req, route, policy)
    prompt = messages[-1]["content"] if messages else req.text

    result = execute_route(route, prompt, req)

    if policy == "public":
        cache.exact_put(req.text, result.text, route=req.router)
        cache.semantic_put(req.text, result.text)

    if session_id:
        app.state.sessions.add_message(session_id, "assistant", result.text)

    _log(req, policy, result.route, reason, result, session_id)
    return {
        "route": result.route,
        "model": local_model_name(result.route),
        "reason": reason,
        "response": result.text,
        "latency_ms": round(result.latency_ms, 1),
        "ttft_ms": result.ttft_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "tokens_per_second": result.tokens_per_second,
        "session_id": session_id,
        "metadata": result.metadata,
        "system": state.model_dump(),
    }


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Stream tokens as NDJSON, with sessions, search and BLUR/cost routing."""

    def generate():
        try:
            state = read_system_state()
            policy = effective_policy(req)

            # Cache hits served on the streaming path too (BUG 8).
            cached = app.state.cache.lookup(
                req.text, policy, req.private, req.offline_only,
                route=req.router,
            )
            if cached is not None:
                for chunk in _split_chars(cached):
                    yield f'{{"delta": {_json(chunk)}}}\n'
                yield (
                    _json_obj({
                        "done": True,
                        "route": "cache",
                        "model": "cache",
                        "reason": "exact/semantic cache hit",
                        "session_id": req.session_id,
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "tokens_per_second": None,
                    })
                    + "\n"
                )
                return

            features = extract_features(req.text, req.use_documents)
            route, reason = resolve_route(req, features, state)
            messages, session_id = build_messages(req, route, policy)

            if route in OFFLOAD_TIERS:
                yield from _stream_offload(req, route, reason, session_id, policy, messages)
                return

            start = time.perf_counter()
            text = ""
            final = None
            gen_route = route if route != "local_rag" else settings.rag_generation_route
            messages_for_model = messages
            passages: list[dict] | None = None
            if route == "local_rag":
                retriever = app.state.retriever
                if retriever is None:
                    raise RuntimeError("RAG index is not available (run `make ingest`)")
                passages = retriever.search(req.text)
                assert passages is not None
                messages_for_model = [
                    {"role": "user", "content": build_rag_prompt(req.text, passages)}
                ]
            try:
                for event in stream_chat(
                    gen_route,
                    messages_for_model,
                    max_tokens=req.max_tokens,
                ):
                    if event.get("done"):
                        final = event
                        break
                    delta = event.get("delta", "")
                    if delta:
                        text += delta
                        yield f'{{"delta": {_json(delta)}}}\n'
            except Exception as exc:
                yield f'{{"error": {_json(str(exc))}}}\n'
                return

            if not final:
                yield '{"error": "stream ended without completion"}\n'
                return
            if final.get("error"):
                yield f'{{"error": {_json(final["error"])}}}\n'
                return
            metadata: dict = {"streaming": True, "reason": reason}
            if route == "local_rag" and passages is not None:
                metadata["sources"] = [
                    {"source": p["source"], "chunk": p["chunk"], "score": p["score"]}
                    for p in passages
                ]
            result = InferenceResult(
                text=text,
                route=route,
                latency_ms=final["latency_ms"],
                ttft_ms=final["ttft_ms"],
                prompt_tokens=final["prompt_tokens"],
                completion_tokens=final["completion_tokens"],
                tokens_per_second=(
                    round(final["completion_tokens"] / (final["latency_ms"] / 1000), 1)
                    if final["latency_ms"] and final["completion_tokens"]
                    else 0
                ),
                metadata=metadata,
            )

            if policy == "public":
                app.state.cache.exact_put(req.text, text, route=req.router)
                app.state.cache.semantic_put(req.text, text)
            if session_id:
                app.state.sessions.add_message(session_id, "assistant", text)
            _log(req, policy, route, reason, result, session_id)
            yield _done(req, route, reason, session_id, result) + "\n"
        except Exception as exc:
            yield f'{{"error": {_json(str(exc))}}}\n'

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _stream_offload(req, route, reason, session_id, policy, messages):
    """Streaming for pooled (native SSE) and cloud (buffered fallback)."""
    if route == "pooled":
        tools = pooled_client.DEFAULT_CODE_TOOLS if req.pooled_tools else None
        text = ""
        final = None
        try:
            for event in pooled_client.stream_pooled(
                messages, max_tokens=req.max_tokens, tools=tools
            ):
                if event.get("done"):
                    final = event
                    break
                delta = event.get("delta", "")
                if delta:
                    text += delta
                    yield f'{{"delta": {_json(delta)}}}\n'
        except pooled_client.PooledUnavailable as exc:
            # room gone mid-request -> finish on a local tier
            try:
                result = execute_route("medium", req.text, req)
                for chunk in _split_chars(result.text):
                    yield f'{{"delta": {_json(chunk)}}}\n'
                yield _done(req, result.route, f"pooled unavailable ({exc})", session_id, result) + "\n"
            except Exception as fallback_exc:
                yield f'{{"error": {_json(str(fallback_exc))}}}\n'
            return
        if final is None or final.get("error"):
            err = (final or {}).get("error", "stream ended without completion")
            yield f'{{"error": {_json(err)}}}\n'
            return
        metadata = {"streaming": True, "reason": reason, "backend": "pooled"}
        if final.get("tool_calls"):
            metadata["tool_calls"] = final["tool_calls"]
        if final.get("reasoning"):
            metadata["reasoning_tokens"] = len(final["reasoning"].split())
        result = InferenceResult(
            text=text,
            route="pooled",
            latency_ms=final["latency_ms"],
            ttft_ms=final["ttft_ms"],
            prompt_tokens=final["prompt_tokens"],
            completion_tokens=final["completion_tokens"],
            tokens_per_second=(
                round(final["completion_tokens"] / (final["latency_ms"] / 1000), 1)
                if final["latency_ms"] and final["completion_tokens"]
                else 0
            ),
            metadata=metadata,
        )
        if policy == "public":
            app.state.cache.exact_put(req.text, text, route=req.router)
            app.state.cache.semantic_put(req.text, text)
        if session_id:
            app.state.sessions.add_message(session_id, "assistant", text)
        _log(req, policy, route, reason, result, session_id)
        yield _done(req, route, reason, session_id, result) + "\n"
        return

    # cloud streaming unsupported -> fall back to non-stream
    result = run_cloud(req.text, max_tokens=req.max_tokens)
    for ch in _split_chars(result.text):
        yield f'{{"delta": {_json(ch)}}}\n'
    yield _done(req, route, reason, session_id, result) + "\n"
    _log(req, policy, route, reason, result, session_id)


def _json(s: str) -> str:
    import json

    return json.dumps(s)


def _json_obj(obj: dict) -> str:
    import json

    return json.dumps(obj)


def _split_chars(text: str, n: int = 64) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)] or [""]


def _done(req: QueryRequest, route: Route, reason: str,
          session_id: str | None, result: InferenceResult) -> str:
    import json as _json

    return _json.dumps(
        {
            "done": True,
            "route": result.route or route,
            "model": local_model_name(result.route or route),
            "reason": reason,
            "session_id": session_id,
            "ttft_ms": result.ttft_ms,
            "latency_ms": round(result.latency_ms, 1),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "tokens_per_second": result.tokens_per_second,
        }
    )
