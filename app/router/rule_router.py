"""Transparent rule-based router baseline (PLAN 15.6 + section 7).

Deterministic, explainable routing: policy -> resource pressure ->
complexity thresholds. Every decision carries a human-readable reason.

Order of preference when work leaves the device: `pooled` (a room of
your own devices, free, open weights) before `cloud` (paid, someone
else's hardware). Both are gated by policy — private/restricted/
ephemeral traffic never leaves the device.
"""

from dataclasses import dataclass

from app.config import settings
from app.router.features import QueryFeatures
from app.schemas import QueryRequest, Route, SystemState


@dataclass
class RoutingDecision:
    route: Route
    reason: str


def _offload_route(enabled: set[str] | None) -> Route | None:
    """Best route that leaves the device, if any is usable."""
    if enabled is None:
        return "cloud"
    if "pooled" in enabled:
        return "pooled"
    if "cloud" in enabled:
        return "cloud"
    return None


def select_route(
    features: QueryFeatures,
    state: SystemState,
    request: QueryRequest,
    enabled: set[str] | None = None,
) -> RoutingDecision:
    """Rule logic per PLAN section 7. Returns the route + a reason string.

    `enabled` is the set of routes this machine can actually serve; when
    None the historical cloud-only behaviour is kept (useful in tests).
    """
    private_mode = request.private or request.offline_only

    if request.use_documents or features.requires_documents:
        return RoutingDecision("local_rag", "Document-grounded request")

    if private_mode:
        route = "small" if features.complexity_score <= 0.45 else "medium"
        return RoutingDecision(route, "Private/offline policy prohibits cloud")

    offload = _offload_route(enabled)

    too_hot = (
        state.cpu_temperature_c is not None
        and state.cpu_temperature_c > settings.max_cpu_temperature_c
    )
    low_memory = state.free_ram_mb < settings.minimum_free_ram_mb

    if too_hot or low_memory:
        if state.network_available and offload is not None and features.complexity_score > 0.35:
            return RoutingDecision(
                offload, f"Local resource pressure -> offload to {offload}"
            )
        return RoutingDecision("small", "Resource pressure with no offload route")

    if features.complexity_score < 0.35:
        return RoutingDecision("small", "Low predicted complexity")
    if features.complexity_score < 0.70:
        return RoutingDecision("medium", "Moderate predicted complexity")
    if state.network_available and offload is not None:
        return RoutingDecision(
            offload, f"High complexity -> offload to {offload} (quality route)"
        )
    return RoutingDecision("medium", "High complexity but offline")
