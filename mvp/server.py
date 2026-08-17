import json
import time
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_BASE = "http://localhost:11434/v1/chat/completions"

MODELS = {
    "small": "qwen2.5:0.5b",
    "medium": "llama3.2:1b",
    "large": "qwen3:4b",
}

app = FastAPI(title="CortexEdge MVP", version="0.1.0")


class Query(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    model: Literal["small", "medium", "large"] = "small"
    max_tokens: int = Field(256, ge=1, le=2048)


@app.get("/health")
def health():
    return {"status": "ok", "models": {k: {"ollama_name": v} for k, v in MODELS.items()}}


@app.post("/query")
def query(req: Query):
    route = req.model
    model_name = MODELS[route]
    start = time.perf_counter()
    try:
        r = requests.post(
            OLLAMA_BASE,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": req.text}],
                "max_tokens": req.max_tokens,
                "temperature": 0.2,
                "stream": False,
            },
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    elapsed_s = time.perf_counter() - start

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0
    tps = completion_tokens / elapsed_s if elapsed_s else 0

    return {
        "route": route,
        "model": model_name,
        "response": text,
        "latency_s": round(elapsed_s, 3),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(tps, 1),
        "usage": usage,
    }