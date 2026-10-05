#!/usr/bin/env python3
"""Before/after response-time measurement for the local models.

Three phases, same prompt, through the app's own client code:

  cold   model not resident (keep_alive=0 first): pays the load cost
  warm   model resident (OLLAMA_KEEP_ALIVE=-1): default thread/context
  tuned  resident + OLLAMA_NUM_THREAD / NUM_CTX / NUM_BATCH from settings

Reports time-to-first-token (streaming) and total latency, and writes
experiments/results/response_time.json.

    python scripts/bench_end_to_end.py --route small --tokens 64
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import keep_alive_value, settings  # noqa: E402
from app.inference import llm_client  # noqa: E402

PROMPT = "Explain in about sixty words why memory bandwidth limits LLM decode speed."
RESULTS_DIR = Path("experiments") / "results"


def unload(model: str):
    llm_client.session().post(
        f"{llm_client.ollama_base()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "x"}],
            "stream": False,
            "options": {"num_predict": 0},
            "keep_alive": 0,
        },
        timeout=60,
    )


def stream_once(route: str, model: str, tokens: int, options: dict | None) -> dict:
    """One streaming run with the app's own code path and explicit options."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "keep_alive": keep_alive_value(),
        "options": {
            "num_predict": tokens,
            "temperature": 0.2,
            **(options or {}),
        },
    }
    start = time.perf_counter()
    ttft = None
    r = llm_client.session().post(
        f"{llm_client.ollama_base()}/api/chat", json=payload, timeout=600, stream=True
    )
    r.raise_for_status()
    final = {}
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        chunk = json.loads(line)
        if ttft is None and chunk.get("message", {}).get("content"):
            ttft = (time.perf_counter() - start) * 1000
        if chunk.get("done"):
            final = chunk
            break
    total = (time.perf_counter() - start) * 1000
    return {
        "ttft_ms": round(ttft or 0, 1),
        "total_ms": round(total, 1),
        "load_ms": round((final.get("load_duration") or 0) / 1e6, 1),
        "prompt_eval_ms": round((final.get("prompt_eval_duration") or 0) / 1e6, 1),
        "eval_ms": round((final.get("eval_duration") or 0) / 1e6, 1),
        "eval_count": final.get("eval_count") or 0,
        "tok_per_s": round(
            (final.get("eval_count") or 0) / ((final.get("eval_duration") or 1) / 1e9), 1
        ),
    }


def phase(label: str, route: str, model: str, tokens: int, options: dict | None,
          repeats: int, cold: bool = False) -> dict:
    if cold:
        unload(model)
    elif options:
        # Switching llama.cpp options (ctx/threads) makes Ollama reload the
        # model, so pay that reload outside the measured samples: every
        # phase is measured with its options already resident.
        try:
            stream_once(route, model, 4, options)
        except Exception:
            pass
    samples = []
    for _ in range(repeats):
        try:
            samples.append(stream_once(route, model, tokens, options))
        except Exception as exc:
            print(f"  {label}: {exc}")
    if not samples:
        return {}
    row = {
        "phase": label,
        "route": route,
        "model": model,
        "options": options or {},
        "repeats": len(samples),
        "ttft_ms": statistics.median(s["ttft_ms"] for s in samples),
        "total_ms": statistics.median(s["total_ms"] for s in samples),
        "load_ms": statistics.median(s["load_ms"] for s in samples),
        "prompt_eval_ms": statistics.median(s["prompt_eval_ms"] for s in samples),
        "tok_per_s": statistics.median(s["tok_per_s"] for s in samples),
        "eval_count": int(statistics.median(s["eval_count"] for s in samples)),
    }
    print(
        f"  {label:<8} {route:<6} ttft={row['ttft_ms']:>7.0f}ms "
        f"total={row['total_ms']:>7.0f}ms load={row['load_ms']:>7.0f}ms "
        f"prefill={row['prompt_eval_ms']:>5.0f}ms decode={row['tok_per_s']:>5.1f} tok/s"
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", nargs="+", default=["small", "medium"])
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    tuned = {
        "num_ctx": settings.ollama_num_ctx,
        "num_batch": settings.ollama_num_batch,
        "num_thread": settings.ollama_num_thread or 12,
    }
    rows = []
    print(f"ollama: {llm_client.ollama_base()}  keep_alive={keep_alive_value()!r}")
    for route in args.routes:
        model = llm_client.local_model_name(route)
        rows.append(phase("cold", route, model, args.tokens, None, 1, cold=True))
        rows.append(phase("warm", route, model, args.tokens, None, args.repeats))
        rows.append(phase("tuned", route, model, args.tokens, tuned, args.repeats))
    rows = [r for r in rows if r]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "response_time.json"
    out.write_text(json.dumps({
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama_base": llm_client.ollama_base(),
        "prompt": PROMPT,
        "max_tokens": args.tokens,
        "tuned_options": tuned,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    for route in args.routes:
        by = {r["phase"]: r for r in rows if r["route"] == route}
        if "cold" in by and "tuned" in by:
            saved = by["cold"]["total_ms"] - by["tuned"]["total_ms"]
            pct = 100 * saved / by["cold"]["total_ms"] if by["cold"]["total_ms"] else 0
            gain = (
                by["tuned"]["tok_per_s"] / by["warm"]["tok_per_s"]
                if by.get("warm", {}).get("tok_per_s")
                else 0
            )
            print(
                f"{route}: cold {by['cold']['total_ms']:.0f}ms -> tuned "
                f"{by['tuned']['total_ms']:.0f}ms ({pct:.0f}% faster); "
                f"decode {gain:.2f}x vs warm default"
            )


if __name__ == "__main__":
    main()
