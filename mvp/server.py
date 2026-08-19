import json
import sys
import time
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import blur
import search
import store
from store import SessionStore

OLLAMA_API = "http://localhost:11434/api/chat"

MODELS = {
    "small": "qwen2.5:0.5b",
    "medium": "gemma2:2b",
    "moe": "qwen2.5:1.5b",  # placeholder for true MoE (OLMoE/A2.7B need llama.cpp)
    "large": "qwen3:4b",
}


def _available_models() -> set[str]:
    """Routes whose Ollama model is actually pulled locally."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        tags = {m["name"] for m in r.json().get("models", [])}
        return {route for route, m in MODELS.items() if m in tags}
    except Exception:
        return set()


# enable routes that are pulled on this machine (updated at startup)
def refresh_enabled():
    available = _available_models()
    enabled = available or {"small"}
    blur.set_enabled(enabled)
    return enabled


app = FastAPI(title="CortexEdge MVP", version="0.2.0")
app.state.enabled_routes = refresh_enabled()


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    model: Literal["small", "medium", "moe", "large", "auto"] = "auto"
    max_tokens: int = Field(1024, ge=1, le=4096)
    session_id: str | None = None
    latency_budget_s: float = Field(2.0, ge=0.5, le=30.0)
    quality_priority: float = Field(0.5, ge=0.0, le=1.0)
    use_search: bool = False


def summarize_fn(text: str) -> str:
    """Compaction: medium model summarizes prior context."""
    try:
        r = requests.post(
            OLLAMA_API,
            json={
                "model": MODELS["medium"],
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
                "options": {"num_predict": 200, "temperature": 0.3},
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return ""


sessions = SessionStore(summarize_fn=summarize_fn)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {k: {"ollama_name": v} for k, v in MODELS.items()},
        "enabled_routes": sorted(app.state.enabled_routes),
        "db": {"responses": store.DB_PATH.name, "ok": True},
    }


@app.post("/query")
def query(req: Query):
    """Non-streaming query (used by benchmarks)."""
    route, reason = resolve_route(req)
    result = run_inference(route, req)
    return {
        "route": route,
        "model": result["model"],
        "reason": reason,
        "response": result["text"],
        "latency_s": round(result["latency_ms"] / 1000, 3),
        "ttft_ms": result["ttft_ms"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "tokens_per_second": result["tokens_per_second"],
    }


@app.post("/query/stream")
def query_stream(req: Query):
    """Stream tokens as they are generated (NDJSON), with memory + BLUR."""

    def generate():
        try:
            route, reason = resolve_route(req)
            session_id = req.session_id or sessions.get_or_create()
            if req.session_id:
                sessions.add_message(session_id, "user", req.text)

            # optional web context
            extra = ""
            if req.use_search:
                results = search.search_web(req.text)
                extra = search.build_search_context(results)

            messages, _ = sessions.build_context(session_id)
            if messages and messages[-1]["role"] == "user":
                messages[-1] = {"role": "user", "content": req.text}
            else:
                messages.append({"role": "user", "content": req.text})
            if extra:
                messages.append(
                    {"role": "system", "content": extra}
                )

            payload = {
                "model": MODELS[route],
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": req.max_tokens,
                    "temperature": 0.2,
                },
            }
            start = time.perf_counter()
            ttft_ms = None
            text = ""
            prompt_tokens = completion_tokens = 0
            try:
                r = requests.post(OLLAMA_API, json=payload, stream=True, timeout=180)
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        text += delta
                        yield f"{json.dumps({'delta': delta})}\n"
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count") or 0
                        completion_tokens = chunk.get("eval_count") or 0
                        break
            except Exception as exc:
                yield json.dumps({"error": str(exc)}) + "\n"
                return

            latency_ms = (time.perf_counter() - start) * 1000
            tps = completion_tokens / (latency_ms / 1000) if completion_tokens else 0

            if req.session_id:
                sessions.add_message(session_id, "assistant", text)

            store.log_response(
                session_id=session_id,
                route=route,
                model=MODELS[route],
                prompt=req.text,
                response=text,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                tokens_per_second=round(tps, 1),
                completion_tokens=completion_tokens,
                policy=f"blur:{route}",
                metadata={"reason": reason, "use_search": req.use_search},
            )
            yield json.dumps(
                {
                    "done": True,
                    "route": route,
                    "model": MODELS[route],
                    "reason": reason,
                    "session_id": session_id,
                    "ttft_ms": round(ttft_ms or 0, 1),
                    "latency_ms": round(latency_ms, 1),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": round(tps, 1),
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def resolve_route(req: Query) -> tuple[str, str]:
    if req.model == "auto":
        route, reason = blur.select_route(
            latency_budget_s=req.latency_budget_s,
            quality_priority=req.quality_priority,
        )
    else:
        route = req.model
        reason = f"explicit:{route}"
    return route, reason


def run_inference(route: str, req: Query) -> dict:
    """Non-streaming inference via native Ollama API (reliable token counts)."""
    messages = [{"role": "user", "content": req.text}]
    start = time.perf_counter()
    try:
        r = requests.post(
            OLLAMA_API,
            json={
                "model": MODELS[route],
                "messages": messages,
                "stream": False,
                "options": {"num_predict": req.max_tokens, "temperature": 0.2},
            },
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    latency_ms = (time.perf_counter() - start) * 1000
    text = data["message"]["content"]
    completion = data.get("eval_count") or 0
    tps = completion / (latency_ms / 1000) if latency_ms else 0
    return {
        "text": text,
        "model": MODELS[route],
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "prompt_tokens": data.get("prompt_eval_count") or 0,
        "completion_tokens": completion,
        "tokens_per_second": round(tps, 1),
    }
