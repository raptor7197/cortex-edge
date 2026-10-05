"""BLUR — Budget-Latency-aware Utility Routing (ported from the MVP).

Interactive budget-driven selection: pick the route with the highest
utility (quality, weighted against latency pressure) among routes that
fit the user's latency budget. Used by the UI's "auto" mode and as an
alternative to the rule router.

`pooled` (peer-to-peer browser inference) appears here as a high-quality
route whose "latency" is time-to-first-token plus the room's queue:
one round-trip per device per token, so it fits interactive budgets
only when the room is on a fast link — see docs/POOLED_INTEGRATION.md.
"""

from dataclasses import dataclass


@dataclass
class RouteProfile:
    name: str
    model: str
    est_latency_s: float
    quality: float  # 0..1 relative quality estimate


ROUTE_PROFILES = {
    "small": RouteProfile("small", "qwen2.5:1.5b", 1.1, 0.55),
    "medium": RouteProfile("medium", "gemma2:2b", 1.6, 0.78),
    "moe": RouteProfile("moe", "qwen2.5:1.5b", 1.1, 0.55),
    "large": RouteProfile("large", "llama3.1:8b", 12.0, 0.90),
    "pooled": RouteProfile("pooled", "pooled (room)", 2.5, 0.93),
}

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

    latency_weight = 1.0 - quality_priority

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
