"""Cost function tests (PLAN 16.1): quality-constrained selection."""

from app.router.cost import RouteCost, select_min_cost, weighted_cost

W = {
    "latency": 1e-4, "energy": 1e-3, "memory": 1e-4,
    "cloud": 5.0, "privacy": 5.0, "quality_penalty": 10.0,
}


def make(name, latency=1000, energy=50, memory=500, cloud=0.0,
         privacy=0.0, quality=0.5):
    return RouteCost(name, name, latency, energy, memory, cloud, privacy, quality)


def test_quality_penalty_applied_below_target():
    low = make("tiny", quality=0.3)
    high = make("big", latency=100, quality=0.9)
    assert weighted_cost(low, W, quality_target=0.6) > weighted_cost(low, W, 0.0)
    assert weighted_cost(high, W, 0.5) == weighted_cost(high, W, 0.5)


def test_cloud_is_expensive():
    local = make("small", latency=200, energy=40, memory=500, quality=0.4)
    remote = make("cloud", latency=100, energy=1, memory=0,
                  cloud=0.15, privacy=1.0, quality=0.95)
    # with default privacy/cloud weights, remote pays ~7.75 extra
    assert weighted_cost(remote, W, 0.5) > weighted_cost(local, W, 0.5)


def test_select_respects_quality_target(monkeypatch):
    monkeypatch.setattr("app.router.cost.ENABLED", {"small", "medium", "cloud"})
    best = select_min_cost(quality_target=0.9)
    assert best.quality >= 0.9  # only cloud meets 0.9


def test_select_falls_back_to_best_quality(monkeypatch):
    # if nothing meets a high target, return the best-quality route
    monkeypatch.setattr("app.router.cost.ENABLED", {"small", "medium", "cloud"})
    best = select_min_cost(quality_target=0.99)
    assert best.quality >= 0.9


def test_select_prefers_feasible_over_cheaper_infeasible(monkeypatch):
    monkeypatch.setattr(
        "app.router.cost.ROUTE_COSTS",
        [
            make("a", latency=10, quality=0.2),
            make("b", latency=50, quality=0.7),
        ],
    )
    monkeypatch.setattr("app.router.cost.ENABLED", {"a", "b"})
    best = select_min_cost(quality_target=0.6)
    assert best.name == "b"  # a is cheaper but below target


def test_low_target_prefers_small(monkeypatch):
    monkeypatch.setattr("app.router.cost.ENABLED", {"small", "medium", "cloud"})
    best = select_min_cost(quality_target=0.3)
    assert best.name in {"small", "medium"}