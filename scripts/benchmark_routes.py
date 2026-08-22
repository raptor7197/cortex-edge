import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Per-route benchmark -> experiments/results/route_benchmark.csv
(PLAN 15.13). Measures latency, TTFT (via streaming), token rate,
RSS, system state for every prompt x route.

Usage: python scripts/benchmark_routes.py [--limit N] [--repeats R]
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import psutil

from app.config import settings
from app.inference import llm_client
from app.inference.llm_client import run_cloud
from app.monitoring.system_state import read_system_state

ROUTES = ["small", "medium", "cloud"]
RESULTS_DIR = Path("experiments") / "results"


def run_with_ttft(route: str, prompt: str, max_tokens: int = 128) -> tuple[float, float | None, float, str]:
    """Non-cloud runs use the streaming path to measure TTFT (PLAN 16.2)."""
    if route == "cloud":
        result = run_cloud(prompt, max_tokens=max_tokens)
        return result.latency_ms, result.ttft_ms, result.tokens_per_second or 0.0, result.text
    events = list(
        llm_client.stream_chat(
            route, [{"role": "user", "content": prompt}], max_tokens=max_tokens
        )
    )
    final = events[-1]
    if final.get("error"):
        raise RuntimeError(f"stream failed: {final['error']}")
    return final["latency_ms"], final["ttft_ms"], _tps(final), final["text"]


def _tps(final: dict) -> float:
    if final["latency_ms"] and final["completion_tokens"]:
        return round(final["completion_tokens"] / (final["latency_ms"] / 1000), 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max prompts to run")
    parser.add_argument("--repeats", type=int, default=3, help="repeats per prompt x route")
    parser.add_argument("--routes", nargs="+", default=ROUTES)
    args = parser.parse_args()

    prompts = [
        l.strip()
        for l in Path("datasets/prompts.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ][: args.limit or None]
    if not prompts:
        print("No prompts found in datasets/prompts.txt")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "route_benchmark.csv"
    process = psutil.Process()
    header = ["prompt_id", "task", "route", "latency_ms", "ttft_ms",
              "tokens_per_second", "rss_mb", "free_ram_mb", "temperature_c",
              "network", "response"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for pid, line in enumerate(prompts):
            task, _, prompt = line.partition("|")
            task = task.strip() or "generic"
            for route in args.routes:
                samples = []
                for _ in range(args.repeats):
                    try:
                        latency, ttft, tps, response = run_with_ttft(route, prompt.strip())
                        samples.append((latency, ttft, tps, response))
                    except Exception as exc:
                        print(f"[{pid}] {task} {route}: {exc}")
                    time.sleep(1)
                if not samples:
                    continue
                latency = statistics.median(s[0] for s in samples)
                ttft_values = [s[1] for s in samples if s[1]]
                ttft = statistics.median(ttft_values) if ttft_values else None
                tps_values = [s[2] for s in samples if s[2]]
                tps = statistics.median(tps_values) if tps_values else 0.0
                response = samples[0][3].replace("\n", " ")
                state = read_system_state()
                writer.writerow({
                    "prompt_id": pid, "task": task, "route": route,
                    "latency_ms": round(latency, 1),
                    "ttft_ms": round(ttft, 1) if ttft is not None else None,
                    "tokens_per_second": tps,
                    "rss_mb": round(process.memory_info().rss / (1024 * 1024), 1),
                    "free_ram_mb": round(state.free_ram_mb, 1),
                    "temperature_c": state.cpu_temperature_c,
                    "network": int(state.network_available),
                    "response": response,
                })
                f.flush()
                print(f"[{pid}] {task:<12} {route:<8} {latency:>8.0f}ms ttft={ttft or 0:>6.0f}ms tps={tps}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
