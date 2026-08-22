import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Build the routing training dataset (PLAN Phase 5).

For every prompt x route: measure latency/TTFT/token rate, then label
the best route using quality-constrained, resource-aware cost
selection (app.router.cost, PLAN 16.1): among routes whose quality is
within tolerance of the best observed quality, pick the lowest
weighted system cost.

Output: datasets/routing_training.csv with prompt features + device
state + best_route labels. Feeds scripts/train_router.py.

Usage: python scripts/build_dataset.py [--limit N] [--repeats R]
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from app.inference import llm_client
from app.monitoring.system_state import read_system_state
from app.router.cost import DEFAULT_WEIGHTS, route_cost_for, weighted_cost
from app.router.features import QueryFeatures, extract_features

ROUTES = ["small", "medium", "cloud"]
OUT = Path("datasets") / "routing_training.csv"

# quality proxy per route, from measured benchmarks (mvp/benchmark.py)
QUALITY_ESTIMATE = {"small": 0.35, "medium": 0.80, "cloud": 0.95}
QUALITY_TOLERANCE = 0.15


def run_route(route: str, prompt: str) -> tuple[float, float, float]:
    if route == "cloud":
        result = llm_client.run_cloud(prompt, max_tokens=128)
        return result.latency_ms, result.ttft_ms or 0.0, result.tokens_per_second or 0.0
    events = list(llm_client.stream_chat(route, [{"role": "user", "content": prompt}], max_tokens=128))
    final = events[-1]
    tps = final["completion_tokens"] / (final["latency_ms"] / 1000) if final["latency_ms"] and final["completion_tokens"] else 0
    return final["latency_ms"], final["ttft_ms"], tps


def best_route_label(feats: QueryFeatures, latencies: dict[str, float],
                     energy_j: dict[str, float], quality: dict[str, float],
                     network: bool) -> str:
    """PLAN best-route rule: within quality tolerance of the best,
    pick the lowest weighted cost."""
    best_q = max(quality.values())
    routes_present = [r for r in ROUTES if r in quality]
    feasible = [r for r in routes_present if quality[r] >= best_q - QUALITY_TOLERANCE]
    if not feasible:
        feasible = [max(quality, key=lambda r: quality[r] if r in quality else -1.0)]

    def cost_for(route: str) -> float:
        profile = route_cost_for(route)
        if profile is None:
            return 1e9
        return weighted_cost(
            profile.__class__(
                name=profile.name, model=profile.model,
                latency_ms=latencies[route], energy_j=energy_j.get(route, 0.0),
                memory_mb=profile.memory_mb, cloud_cost=profile.cloud_cost,
                privacy_risk=profile.privacy_risk, quality=quality[route],
            ),
            DEFAULT_WEIGHTS,
            quality_target=best_q - QUALITY_TOLERANCE,
        )

    return min(feasible, key=cost_for)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-cloud", action="store_true",
                        help="skip cloud route (no key configured)")
    args = parser.parse_args()

    prompts = [
        l.strip()
        for l in Path("datasets/prompts.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ][: args.limit or None]
    if not prompts:
        print("No prompts in datasets/prompts.txt")
        return
    routes = [r for r in ROUTES if not (args.skip_cloud and r == "cloud")]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id", "task", "prompt", "token_count", "sentence_count",
                "has_code", "has_math", "reasoning_score", "multi_part_score",
                "requires_documents", "complexity_score",
                *[f"latency_{r}" for r in routes],
                *[f"ttft_{r}" for r in routes],
                *[f"tps_{r}" for r in routes],
                "free_ram_mb", "cpu_percent", "cpu_temperature_c", "network_available",
                "best_route",
            ],
        )
        writer.writeheader()
        for pid, line in enumerate(prompts):
            task, _, prompt_text = line.partition("|")
            task = task.strip() or "generic"
            feats = extract_features(prompt_text)
            latencies, ttfts, tps, quality = {}, {}, {}, {}
            for route in routes:
                samples = []
                for _ in range(args.repeats):
                    try:
                        samples.append(run_route(route, prompt_text.strip()))
                    except Exception as exc:
                        print(f"[{pid}] {route}: {exc}")
                    time.sleep(1)
                if not samples:
                    latencies[route] = 1e6
                    continue
                latencies[route] = statistics.median(s[0] for s in samples)
                ttfts[route] = statistics.median(s[1] for s in samples)
                tps[route] = statistics.median(s[2] for s in samples)
                quality[route] = QUALITY_ESTIMATE[route]
                print(f"[{pid}] {task:<12} {route:<8} {latencies[route]:>8.0f}ms")

            state = read_system_state()
            label = best_route_label(
                feats, latencies, {}, quality, state.network_available
            ) if len(quality) > 1 else routes[0]
            row = feats.as_dict() | {
                "prompt_id": pid, "task": task, "prompt": prompt_text.strip(),
                **{f"latency_{r}": round(latencies.get(r, 0), 1) for r in routes},
                **{f"ttft_{r}": round(ttfts.get(r, 0), 1) for r in routes},
                **{f"tps_{r}": round(tps.get(r, 0), 1) for r in routes},
                "free_ram_mb": round(state.free_ram_mb, 1),
                "cpu_percent": state.cpu_percent,
                "cpu_temperature_c": state.cpu_temperature_c,
                "network_available": int(state.network_available),
                "best_route": label,
            }
            writer.writerow(row)
            f.flush()
    print(f"\nWrote {OUT} with best-route labels "
          f"(quality tolerance {QUALITY_TOLERANCE})")


if __name__ == "__main__":
    main()
