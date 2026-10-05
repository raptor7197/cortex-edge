#!/usr/bin/env python3
"""Measure and tune local inference latency (docs/LOCAL_LATENCY.md).

Runs the same prompt against every configured local tier under several
llama.cpp option sets and reports, per request:

  load_ms            model load (0 when the model is already resident)
  prompt_eval_ms     prefill
  eval_ms            decode
  tok_per_s          decode tokens per second
  total_ms           wall time seen by the client

Writes experiments/results/latency.json and prints a table sorted by
decode speed, so the fastest option set per tier is obvious.

    python scripts/bench_latency.py [--models small medium] [--repeats 2]
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

PROMPT = (
    "Explain what a CPU cache is and why it matters for performance, "
    "in about 60 words."
)

# Option sets to compare. `num_ctx` shrinks the KV cache, `num_thread`
# controls CPU parallelism, `num_batch` the prefill batch size.
VARIANTS = [
    ("baseline-ctx4096", {"num_ctx": 4096, "num_batch": 512}),
    ("ctx2048", {"num_ctx": 2048, "num_batch": 512}),
    ("ctx1024", {"num_ctx": 1024, "num_batch": 512}),
    ("ctx2048-threads4", {"num_ctx": 2048, "num_batch": 512, "num_thread": 4}),
    ("ctx2048-threads6", {"num_ctx": 2048, "num_batch": 512, "num_thread": 6}),
    ("ctx2048-threads8", {"num_ctx": 2048, "num_batch": 512, "num_thread": 8}),
    ("ctx2048-threads10", {"num_ctx": 2048, "num_batch": 512, "num_thread": 10}),
    ("ctx2048-threads12", {"num_ctx": 2048, "num_batch": 512, "num_thread": 12}),
    ("ctx2048-threads16", {"num_ctx": 2048, "num_batch": 512, "num_thread": 16}),
    ("ctx2048-batch1024", {"num_ctx": 2048, "num_batch": 1024}),
    ("ctx2048-mlock", {"num_ctx": 2048, "num_batch": 512, "use_mlock": True}),
    ("ctx2048-nopredict", {"num_ctx": 2048, "num_batch": 512, "num_predict": 0}),
]

RESULTS_DIR = Path("experiments") / "results"


def one_request(model: str, options: dict, max_tokens: int, keep_alive) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {**options, "num_predict": options.get("num_predict", max_tokens),
                    "temperature": 0.2},
    }
    start = time.perf_counter()
    r = llm_client.session().post(
        f"{llm_client.ollama_base()}/api/chat", json=payload, timeout=600
    )
    r.raise_for_status()
    wall_ms = (time.perf_counter() - start) * 1000
    d = r.json()
    eval_count = d.get("eval_count") or 0
    eval_ms = (d.get("eval_duration") or 0) / 1e6
    return {
        "wall_ms": round(wall_ms, 1),
        "load_ms": round((d.get("load_duration") or 0) / 1e6, 1),
        "prompt_eval_ms": round((d.get("prompt_eval_duration") or 0) / 1e6, 1),
        "eval_ms": round(eval_ms, 1),
        "eval_count": eval_count,
        "tok_per_s": round(eval_count / (eval_ms / 1000), 1) if eval_ms and eval_count else 0.0,
        "answer_chars": len(d["message"]["content"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small", "medium"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="run only these variant names")
    ap.add_argument("--tag", default="latency",
                    help="output file suffix: experiments/results/<tag>.json")
    args = ap.parse_args()

    keep_alive = keep_alive_value()
    results: list[dict] = []
    print(f"ollama: {llm_client.ollama_base()}  keep_alive={keep_alive!r}")

    for route in args.models:
        model = llm_client.local_model_name(route)
        for name, options in VARIANTS:
            if args.variants and name not in args.variants:
                continue
            # 1 warm-up request (pays the load), then measured repeats
            try:
                one_request(model, options, args.max_tokens, keep_alive)
            except Exception as exc:
                print(f"  {route}/{name}: warmup failed: {exc}")
                continue
            samples = []
            for _ in range(args.repeats):
                try:
                    samples.append(one_request(model, options, args.max_tokens, keep_alive))
                except Exception as exc:
                    print(f"  {route}/{name}: {exc}")
            if not samples:
                continue
            row = {
                "route": route,
                "model": model,
                "variant": name,
                "options": options,
                "load_ms": statistics.median(s["load_ms"] for s in samples),
                "prompt_eval_ms": statistics.median(s["prompt_eval_ms"] for s in samples),
                "eval_ms": statistics.median(s["eval_ms"] for s in samples),
                "tok_per_s": round(
                    statistics.median(s["tok_per_s"] for s in samples), 1
                ),
                "wall_ms": statistics.median(s["wall_ms"] for s in samples),
                "eval_count": int(statistics.median(s["eval_count"] for s in samples)),
            }
            results.append(row)
            print(
                f"  {row['route']:<6} {name:<20} load={row['load_ms']:>7.0f}ms "
                f"prefill={row['prompt_eval_ms']:>6.0f}ms decode={row['tok_per_s']:>5.1f} tok/s "
                f"wall={row['wall_ms']:>7.0f}ms"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ollama_base": llm_client.ollama_base(),
        "keep_alive": keep_alive,
        "prompt": PROMPT,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    best: dict[str, dict] = {}
    for row in results:
        cur = best.get(row["route"])
        if cur is None or row["tok_per_s"] > cur["tok_per_s"]:
            best[row["route"]] = row
    print("\nfastest option set per tier:")
    for route, row in sorted(best.items()):
        print(f"  {route:<6} {row['variant']:<20} {row['tok_per_s']:>5.1f} tok/s  {row['options']}")


if __name__ == "__main__":
    main()
