"""Rule router tests (PLAN 15.15)."""

from app.router.features import extract_features
from app.router.rule_router import select_route
from app.schemas import QueryRequest, SystemState


def state(ram=4000, temp=55, network=True):
    return SystemState(
        free_ram_mb=ram, cpu_percent=15,
        cpu_temperature_c=temp, network_available=network,
    )


def test_simple_prompt_uses_small_model():
    q = QueryRequest(text="Hello")
    assert select_route(extract_features(q.text), state(), q).route == "small"


def test_private_complex_prompt_never_uses_cloud():
    text = "Derive and compare three algorithms with code and mathematical analysis."
    q = QueryRequest(text=text, private=True)
    assert select_route(extract_features(text), state(), q).route in {"small", "medium"}


def test_document_request_uses_rag():
    q = QueryRequest(text="Summarize the uploaded paper", use_documents=True)
    assert select_route(extract_features(q.text, True), state(), q).route == "local_rag"


def test_resource_pressure_can_offload():
    q = QueryRequest(text="Analyse this difficult multi-step problem and provide code.")
    assert select_route(extract_features(q.text), state(ram=700, temp=80), q).route in {"small", "cloud"}


def test_offline_only_stays_local_even_when_cloud_faster():
    q = QueryRequest(text="Explain why the sky is blue in detail.", offline_only=True)
    route = select_route(extract_features(q.text), state(), q).route
    assert route in {"small", "medium"}


def test_moderate_complexity_uses_medium():
    text = (
        "Compare the benefits of solar and wind energy step by step: "
        "analyse costs and efficiency, then derive the break-even "
        "formula A = B + C."
    )
    q = QueryRequest(text=text)
    decision = select_route(extract_features(text), state(), q)
    assert decision.route == "medium"
    assert "complexity" in decision.reason