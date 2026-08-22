"""Quality-constrained, resource-aware cost selection (PLAN 16.1).

This is the core research contribution: for each candidate route we
compute a weighted cost (latency, energy, memory, cloud cost, privacy
risk) and reward response quality. We select the minimum-cost route
that satisfies the quality target, falling back to best-quality if
none is feasible.

Route estimates come from measured benchmark values (mvp/benchmark.py);
tune after running `scripts/benchmark_routes.py`.
"""

from dataclasses import dataclass


@dataclass
class RouteCost:
    name: str
    model: str
    latency_ms: float
    energy_j: float
    memory_mb: float
    cloud_cost: float = 0.0          # $ per 1k queries (0 for local)
    privacy_risk: float = 0.0        # 0 local ... 1 cloud
    quality: float = 0.5             # 0..1 relative quality estimate
    is_local_rag: bool = False


# Estimates measured on the dev laptop (16-core CPU, no dGPU); the
# latency figures include generation for typical ~50-token answers.
ROUTE_COSTS = [
    RouteCost("small", "qwen2.5:0.5b", latency_ms=700, energy_j=30, memory_mb=500, quality=0.35),
    RouteCost("medium", "gemma2:2b", latency_ms=3000, energy_j=90, memory_mb=1700, quality=0.80),
    RouteCost("moe", "qwen2.5:1.5b", latency_ms=1200, energy_j=50, memory_mb=1000, quality=0.60),
    RouteCost("large", "qwen3:4b", latency_ms=8000, energy_j=200, memory_mb=2600, quality=0.90),
    RouteCost(
        "cloud", "openai/gpt-4o-mini",
        latency_ms=2500, energy_j=5, memory_mb=0,
        cloud_cost=0.15, privacy_risk=1.0, quality=0.95,
    ),
]

DEFAULT_WEIGHTS = {
    "latency": 1e-4,      # per ms
    "energy": 1e-3,       # per joule
    "memory": 1e-4,       # per MB (residence cost)
    "cloud": 5.0,         # per $ (cloud spend penalty)
    "privacy": 5.0,       # per unit privacy risk
    "quality_penalty": 10.0,  # per unit of quality deficit
}

# routes that can actually be used on this machine (set at startup)
ENABLED = {"small", "medium"}


def set_enabled(routes: set[str]):
    global ENABLED
    ENABLED = routes


def weighted_cost(c: RouteCost, w: dict, quality_target: float) -> float:
    """Cost = weighted resource sum, penalised if quality below target."""
    cost = (
        w["latency"] * c.latency_ms
        + w["energy"] * c.energy_j
        + w["memory"] * c.memory_mb
        + w["cloud"] * c.cloud_cost
        + w["privacy"] * c.privacy_risk
    )
    if c.quality < quality_target:
        cost += w["quality_penalty"] * (quality_target - c.quality)
    return cost


def select_min_cost(
    quality_target: float,
    weights: dict | None = None,
    enabled: set[str] | None = None,
) -> RouteCost:
    """Minimum weighted cost among routes meeting the quality target."""
    w = weights or DEFAULT_WEIGHTS
    pool = [c for c in ROUTE_COSTS if c.name in (enabled or ENABLED)]
    if not pool:
        raise RuntimeError("No routes enabled")
    feasible = [c for c in pool if c.quality >= quality_target]
    if feasible:
        return min(feasible, key=lambda c: weighted_cost(c, w, quality_target))
    return max(pool, key=lambda c: c.quality)  # fall back to best quality


def route_cost_for(name: str) -> RouteCost | None:
    for c in ROUTE_COSTS:
        if c.name == name:
            return c
    return None
