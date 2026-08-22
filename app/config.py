"""Central configuration for CortexEdge.

All values can be overridden via environment variables or `.env`
(see .env.example). Settings follow PLAN.md section 15.2.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # --- inference backend -------------------------------------------------
    # "ollama"  -> Ollama native API (http://ollama_api_url/api/chat).
    #              Reliable token counts (prompt_eval_count/eval_count).
    # "openai"  -> OpenAI-compatible endpoints (llama.cpp /v1/chat/completions,
    #              one server per tier: ports 8101/8102 as per PLAN).
    backend: str = "ollama"
    ollama_api_url: str = "http://localhost:11434"
    small_model_url: str = "http://127.0.0.1:8101/v1/chat/completions"
    medium_model_url: str = "http://127.0.0.1:8102/v1/chat/completions"

    # Ollama model names per route (only meaningful with backend="ollama")
    model_small: str = "qwen2.5:0.5b"
    model_medium: str = "gemma2:2b"
    model_moe: str = "qwen2.5:1.5b"  # placeholder until a true MoE is available
    model_large: str = "qwen3:4b"

    # --- cloud fallback ----------------------------------------------------
    cloud_model_url: str | None = None
    cloud_api_key: str | None = None
    cloud_model: str = "openai/gpt-4o-mini"
    cloud_max_failures: int = 3      # circuit breaker: consecutive failures
    cloud_cooldown_s: int = 60       # circuit breaker: cooldown after opening

    # --- RAG ----------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: Path = ROOT / "datasets" / "faiss.index"
    chunk_store_path: Path = ROOT / "datasets" / "chunks.json"
    rag_generation_route: str = "medium"
    rag_chunk_size: int = 450
    rag_chunk_overlap: int = 70

    # --- cache --------------------------------------------------------------
    cache_path: Path = ROOT / ".cache"
    semantic_cache_threshold: float = 0.95

    # --- monitoring / limits ------------------------------------------------
    database_path: Path = ROOT / "experiments" / "cortexedge.db"
    max_cpu_temperature_c: float = 75.0
    minimum_free_ram_mb: int = 1500
    request_timeout_s: int = 90

    # --- routing ------------------------------------------------------------
    # default router: "rule" (deterministic baseline) | "cost" | "blur"
    default_router: str = "rule"
    # default privacy policy: public | private | restricted | ephemeral
    default_policy: str = "public"
    quality_target: float = 0.6  # default quality floor for cost router

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")


settings = Settings()
