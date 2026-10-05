"""Routing tests for the `pooled` (peer-to-peer browser inference) route.

Covers the three things that must hold for the integration to be safe:
  1. pooled is never chosen for private/restricted/ephemeral traffic,
  2. pooled is cheaper than paid cloud (so it wins whenever both are
     plausible) and is preferred for complex queries when a room exists,
  3. pooled disappears from the route table when no bridge is reachable.
"""

import pytest

from app.api import server
from app.router import blur, cost, rule_router
from app.router.features import extract_features
from app.schemas import QueryRequest, SystemState

HARD_PROMPT = (
    "Derive step by step why merge sort is O(n log n); then write a python "
    "function that implements it; compare its performance with quicksort "
    "for nearly sorted input. Why is quicksort faster in practice? Explain "
    "the worst case mathematically. Analyse the memory overhead of each "
    "algorithm and summarise your findings in three bullet points for a "
    "code review."
)


def _state(network=True, free_mb=8000):
    return SystemState(
        free_ram_mb=free_mb,
        cpu_percent=10.0,
        cpu_temperature_c=45.0,
        network_available=network,
        loaded_models=[],
    )


def test_hard_prompt_is_high_complexity():
    """The router fixtures below depend on this staying a complex query."""
    assert extract_features(HARD_PROMPT).complexity_score >= 0.70


def test_rule_router_offloads_to_pooled_before_cloud():
    req = QueryRequest(text=HARD_PROMPT, router="rule")
    enabled = {"small", "medium", "large", "pooled", "cloud"}
    decision = rule_router.select_route(extract_features(HARD_PROMPT), _state(), req, enabled)
    assert decision.route == "pooled"
    assert "offload" in decision.reason


def test_rule_router_falls_back_to_cloud_without_room():
    req = QueryRequest(text=HARD_PROMPT, router="rule")
    decision = rule_router.select_route(
        extract_features(HARD_PROMPT), _state(), req, {"small", "medium", "cloud"}
    )
    assert decision.route == "cloud"


def test_rule_router_stays_local_when_offline():
    req = QueryRequest(text=HARD_PROMPT, router="rule")
    decision = rule_router.select_route(
        extract_features(HARD_PROMPT), _state(network=False), req, {"small", "medium"}
    )
    assert decision.route == "medium"


def test_cost_router_prices_pooled_below_paid_cloud():
    """The whole point of the route: cloud-grade quality, no per-token $."""
    w = cost.DEFAULT_WEIGHTS
    pooled = cost.route_cost_for("pooled")
    cloud = cost.route_cost_for("cloud")
    assert cost.weighted_cost(pooled, w, 0.9) < cost.weighted_cost(cloud, w, 0.9)


def test_cost_router_avoids_paid_cloud_when_pooled_is_feasible():
    best = cost.select_min_cost(
        quality_target=0.9, enabled={"small", "medium", "large", "pooled", "cloud"}
    )
    assert best.name in {"pooled", "large"}  # never the paid route


def test_cost_router_keeps_highest_quality_when_target_unreachable():
    best = cost.select_min_cost(quality_target=0.99, enabled={"small", "medium", "pooled", "cloud"})
    assert best.name == "cloud"  # quality 0.95 is the best available


def test_blur_includes_pooled_when_enabled():
    blur.set_enabled({"small", "medium", "pooled"})
    try:
        route, reason = blur.select_route(latency_budget_s=4.0, quality_priority=0.9)
        assert route == "pooled"
        assert "BLUR" in reason
    finally:
        blur.set_enabled({"small", "medium"})


def test_pooled_requires_public_policy(monkeypatch):
    monkeypatch.setattr(server.pooled_client, "available", lambda: True)
    assert server._pooled_usable("public") is True
    for policy in ("private", "restricted", "ephemeral"):
        assert server._pooled_usable(policy) is False
    assert server._pooled_usable("public", offline_only=True) is False


def test_explicit_pooled_request_rejected_when_policy_is_private(monkeypatch):
    monkeypatch.setattr(server.pooled_client, "available", lambda: True)
    monkeypatch.setattr(server.app.state, "enabled_routes", {"small", "medium", "pooled"})
    req = QueryRequest(text="x", model="pooled", policy="private")
    with pytest.raises(Exception):
        server.resolve_route(req, extract_features("x"), _state())


def test_enabled_routes_exclude_pooled_without_a_room(monkeypatch):
    monkeypatch.setattr(server.pooled_client, "available", lambda: False)
    monkeypatch.setattr(server, "_available_models", lambda: {"small", "medium"})
    enabled = server.refresh_enabled()
    assert "pooled" not in enabled
    assert {"small", "medium"} <= enabled


def test_enabled_routes_include_pooled_with_a_room(monkeypatch):
    monkeypatch.setattr(server.pooled_client, "available", lambda: True)
    monkeypatch.setattr(server, "_available_models", lambda: {"small", "medium"})
    enabled = server.refresh_enabled()
    assert "pooled" in enabled
