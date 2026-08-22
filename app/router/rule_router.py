"""Transparent rule-based router baseline (PLAN 15.6 + section 7).

Deterministic, explainable routing: policy -> resource pressure ->
complexity thresholds. Every decision carries a human-readable reason.
"""

from dataclasses import dataclass

from app.config import settings
from app.router.features import QueryFeatures
from app.schemas import QueryRequest, Route, SystemState


@dataclass
class RoutingDecision:
    route: Route
    reason: str


def select_route(
    features: QueryFeatures,
    state: SystemState,
    request: QueryRequest,
) -> RoutingDecision:
    """Rule logic per PLAN section 7. Returns the route + a reason string."""
    private_mode = request.private or request.offline_only

    if request.use_documents or features.requires_documents:
        return RoutingDecision("local_rag", "Document-grounded request")

    if private_mode:
        route = "small" if features.complexity_score <= 0.45 else "medium"
        return RoutingDecision(route, "Private/offline policy prohibits cloud")

    too_hot = (
        state.cpu_temperature_c is not None
        and state.cpu_temperature_c > settings.max_cpu_temperature_c
    )
    low_memory = state.free_ram_mb < settings.minimum_free_ram_mb

    if too_hot or low_memory:
        if state.network_available and features.complexity_score > 0.35:
            return RoutingDecision("cloud", "Local resource pressure")
        return RoutingDecision("small", "Resource pressure with no cloud")

    if features.complexity_score < 0.35:
        return RoutingDecision("small", "Low predicted complexity")
    if features.complexity_score < 0.70:
        return RoutingDecision("medium", "Moderate predicted complexity")
    if state.network_available:
        return RoutingDecision("cloud", "High complexity and network available")
    return RoutingDecision("medium", "High complexity but offline")
