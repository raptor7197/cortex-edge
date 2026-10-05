"""Central configuration for CortexEdge.

All values can be overridden via environment variables or `.env`
(see .env.example). Settings follow PLAN.md section 15.2.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


def keep_alive_value() -> int | str:
    """Ollama accepts `keep_alive` as a number of seconds or a duration
    string ("5m", "1h"). The env var is a string, so normalise numeric
    values to int — Ollama's embed endpoint rejects `"-1"` with a 400."""
    v = str(settings.ollama_keep_alive).strip()
    try:
        return int(v)
    except ValueError:
        return v


class Settings(BaseSettings):
    # --- inference backend -------------------------------------------------
    # "ollama"  -> Ollama native API (http://ollama_api_url/api/chat).
    #              Reliable token counts (prompt_eval_count/eval_count).
    # "openai"  -> OpenAI-compatible endpoints (llama.cpp /v1/chat/completions,
    #              one server per tier: ports 8101/8102 as per PLAN).
    backend: str = "ollama"
    # "auto" probes localhost, then the container's default gateway (the dev
    # host that actually runs Ollama) — see app/inference/llm_client.py.
    ollama_api_url: str = "auto"
    small_model_url: str = "http://127.0.0.1:8101/v1/chat/completions"
    medium_model_url: str = "http://127.0.0.1:8102/v1/chat/completions"

    # Ollama model names per route (only meaningful with backend="ollama").
    # Defaults name models that exist on this machine — all four are imported
    # from the GGUF files already on disk (scripts/import_local_models.sh);
    # nothing is downloaded.
    model_small: str = "qwen2.5:1.5b"
    model_medium: str = "gemma2:2b"
    model_moe: str = "qwen2.5:1.5b"  # placeholder until a true MoE is available
    model_large: str = "llama3.1:8b"

    # --- local inference tuning (docs/LOCAL_LATENCY.md) ---------------------
    ollama_keep_alive: str = "-1"    # -1 = never unload; kills the reload cost
    ollama_num_ctx: int = 4096       # smaller KV cache = less memory traffic
    ollama_num_batch: int = 512      # prefill batch size
    ollama_num_thread: int = 0       # 0 => let llama.cpp pick (host cores)
    ollama_warmup: bool = True       # preload tier models at startup
    # which tiers to preload — keep this to the tiers the router actually
    # uses: preloading an 8B tier on a 16 GB box pushes the machine into
    # swap and makes *every* request slower (docs/LOCAL_LATENCY.md).
    ollama_warmup_routes: str = "small,medium"
    ollama_embed_model: str = "nomic-embed-text"
    # the embedding model must NOT hold a resident slot forever: with
    # OLLAMA_MAX_LOADED_MODELS=2 it evicts the chat tiers and every request
    # pays a reload (measured: 4.5 s instead of 0.5 s).
    ollama_embed_keep_alive: str = "30m"
    # slots in the Ollama scheduler: small + medium + the embedder
    ollama_max_loaded_models: int = 3

    # --- pooled bridge: peer-to-peer browser inference (Nehanth/pooled) -----
    # `npx @pooled/cli serve "<room link>"` exposes the room as an
    # OpenAI-compatible endpoint; CortexEdge uses it as the `pooled` route.
    pooled_enabled: bool = True
    pooled_base_url: str = "http://127.0.0.1:8080/v1"
    pooled_api_key: str | None = None      # set when the bridge uses POOLED_TOKEN
    pooled_model: str = "pooled"
    pooled_room_link: str | None = None    # https://pooled.run/r/<code>#k=...
    pooled_timeout_s: int = 180
    pooled_probe_timeout_s: float = 2.0    # /v1/models health probe

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
