"""BLUR — Budget-Latency-aware Utility Routing.

Selects a route minimizing cost while satisfying a latency budget and a
quality floor. Route profiles come from measured benchmark values
(est_latency_s, quality) — tune after running `benchmark.py`.
"""

from dataclasses import dataclass


@dataclass
class RouteProfile:
    name: str
    model: str
    est_latency_s: float
    quality: float  # 0..1 relative quality estimate


ROUTE_PROFILES = {
    "small": RouteProfile("small", "qwen2.5:0.5b", 0.7, 0.35),
    "medium": RouteProfile("medium", "gemma2:2b", 3.0, 0.80),
    "moe": RouteProfile("moe", "qwen2.5:1.5b", 1.2, 0.60),
    "large": RouteProfile("large", "qwen3:4b", 8.0, 0.90),
}

# routes that can actually be used (large is slow; set by availability)
ENABLED = {"small", "medium"}


def set_enabled(routes: set[str]):
    global ENABLED
    ENABLED = routes


def select_route(
    latency_budget_s: float = 2.0,
    quality_priority: float = 0.5,
) -> tuple[str, str]:
    """Return (route_name, reason). Lower budget -> fast routes; higher
    quality_priority -> quality routes. Falls back to fastest route when
    the budget is too tight for anything else."""
    candidates = [p for name, p in ROUTE_PROFILES.items() if name in ENABLED]
    if not candidates:
        raise RuntimeError("No routes enabled")

    budget = max(latency_budget_s, 0.5)

    feasible = [p for p in candidates if p.est_latency_s <= budget]
    pool = feasible or [min(candidates, key=lambda p: p.est_latency_s)]

    # utility = quality, weighted against latency pressure
    latency_weight = 1.0 - quality_priority  # higher priority -> less latency weight
    def utility(p: RouteProfile) -> float:
        return p.quality - latency_weight * (p.est_latency_s / budget)

    best = max(pool, key=utility)
    reason = (
        f"BLUR: budget={budget}s quality_priority={quality_priority} "
        f"-> {best.name} (est {best.est_latency_s}s, quality {best.quality})"
    )
    return best.name, reason


def profile_for(route: str) -> RouteProfile | None:
    return ROUTE_PROFILES.get(route)
