"""Quality-constrained, resource-aware cost selection (PLAN 16.1).

This is the core research contribution: for each candidate route we
compute a weighted cost (latency, energy, memory, cloud cost, privacy
risk) and reward response quality. We select the minimum-cost route
that satisfies the quality target, falling back to best-quality if
none is feasible.

Route estimates are refreshed from measured values
(`scripts/benchmark_routes.py`, docs/LOCAL_LATENCY.md) — the numbers
below were measured on the dev laptop (16-core CPU, no dGPU) with the
GGUF models already on disk, warm (`keep_alive=-1`), 50-token answers.

The `pooled` route is peer-to-peer browser inference (Nehanth/pooled):
a room of your own devices running a 27B/35B MoE model in WebGPU tabs.
It costs no money and no local memory, has cloud-grade quality, but the
prompt leaves the device (hidden states go to the room's peers), so its
privacy risk sits next to cloud — just below it, since the peers are
yours and the weights are open.
"""

from dataclasses import dataclass


@dataclass
class RouteCost:
    name: str
    model: str
    latency_ms: float
    energy_j: float
    memory_mb: float
    cloud_cost: float = 0.0          # $ per 1k queries (0 for local/pooled)
    privacy_risk: float = 0.0        # 0 local ... 1 cloud
    quality: float = 0.5             # 0..1 relative quality estimate
    is_local_rag: bool = False


# Measured warm latencies (see docs/LOCAL_LATENCY.md):
#   small   qwen2.5:1.5b ~1.1 s for 50 tokens
#   moe     qwen2.5:1.5b placeholder (no MoE weights on this box yet)
#   medium  gemma2:2b    ~1.6 s for 50 tokens
#   large   llama3.1:8b  ~9-14 s for 50 tokens (CPU-bound)
_ROUTE_COSTS = {
    "small": RouteCost("small", "qwen2.5:1.5b", latency_ms=1100, energy_j=30, memory_mb=1100, quality=0.55),
    "medium": RouteCost("medium", "gemma2:2b", latency_ms=1600, energy_j=60, memory_mb=1900, quality=0.78),
    "moe": RouteCost("moe", "qwen2.5:1.5b", latency_ms=1100, energy_j=30, memory_mb=1100, quality=0.55),
    "large": RouteCost("large", "llama3.1:8b", latency_ms=12000, energy_j=260, memory_mb=5200, quality=0.90),
    "pooled": RouteCost(
        "pooled", "pooled (room)", latency_ms=4500, energy_j=0, memory_mb=0,
        cloud_cost=0.0, privacy_risk=0.8, quality=0.93,
    ),
    "cloud": RouteCost(
        "cloud", "openai/gpt-4o-mini",
        latency_ms=2500, energy_j=5, memory_mb=0,
        cloud_cost=0.15, privacy_risk=1.0, quality=0.95,
    ),
}

# Public list form kept for the UI/back-compat.
ROUTE_COSTS = list(_ROUTE_COSTS.values())

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
    return _ROUTE_COSTS.get(name)
