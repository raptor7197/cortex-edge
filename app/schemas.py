"""Pydantic request/result/state models (PLAN.md section 15.3)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["small", "medium", "moe", "large", "local_rag", "cloud", "cache"]
Router = Literal["rule", "cost", "blur", "learned"]
Policy = Literal["public", "private", "restricted", "ephemeral"]


class QueryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None
    router: Router = "rule"
    policy: Policy = "public"
    offline_only: bool = False
    private: bool = False
    use_documents: bool = False
    use_search: bool = False
    response_mode: Literal["text", "voice"] = "text"
    max_tokens: int = Field(1024, ge=1, le=4096)
    quality_priority: float = Field(0.5, ge=0.0, le=1.0)
    latency_priority: float = Field(0.5, ge=0.0, le=1.0)
    energy_priority: float = Field(0.5, ge=0.0, le=1.0)
    latency_budget_s: float = Field(2.0, ge=0.5, le=30.0)
    model: Literal["auto", "small", "medium", "moe", "large"] = "auto"


class InferenceResult(BaseModel):
    text: str
    route: Route
    latency_ms: float
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    metadata: dict[str, Any] = {}


class SystemState(BaseModel):
    free_ram_mb: float
    cpu_percent: float
    cpu_temperature_c: float | None
    network_available: bool
    loaded_models: list[str] = []
