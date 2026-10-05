# Changelog

## 0.3.0 — pooled route + measured local latency

### Added
- **`pooled` route** (peer-to-peer browser inference, upstream
  <https://github.com/Nehanth/pooled>): `app/inference/pooled_client.py`
  speaks the room's OpenAI-compatible bridge with one reused HTTP session,
  SSE streaming, usage reporting and a circuit breaker.
- `POST /pooled/code` — one agent step through a room with tool calling
  (pooled grammar-constrains calls to the declared schema).
- `GET /pooled/status`, `GET /routes`, `?refresh=true` on `/health`.
- Routing: `pooled` is priced in the cost table (quality 0.93, $0 per 1k,
  privacy risk 0.8) and preferred over paid cloud; the rule router offloads
  complex queries to a room when one exists; blur has a pooled profile.
  Policy gate: `pooled` never serves private/restricted/ephemeral or
  offline-only requests.
- Latency tuning: `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_CTX`, `OLLAMA_NUM_BATCH`,
  `OLLAMA_NUM_THREAD`, `OLLAMA_WARMUP_ROUTES`, plus startup warm-up of the
  tiers the router actually uses.
- Ollama-based embeddings (`nomic-embed-text`) as the preferred embedder
  for RAG and the semantic cache; `OLLAMA_API_URL=auto` discovers a local or
  gateway Ollama.
- Scripts: `start_ollama_local.sh`, `import_local_models.sh` (GGUF -> Ollama,
  no downloads), `smoke_test.sh`, `restart_api.sh`, `pooled_serve.sh`,
  `pooled_mock_bridge.py`, `verify_pooled_route.sh`, `bench_latency.py`,
  `bench_end_to_end.py`.
- Docs: `docs/LOCAL_LATENCY.md` (measured numbers + playbook),
  `docs/POOLED_INTEGRATION.md` (design, capabilities, limits).
- Tests: `tests/test_pooled_client.py` (real HTTP stand-in bridge),
  `tests/test_router_pooled.py` (routing + privacy gate). 50 tests total.

### Changed
- Default model names now point at models imported from GGUF files already
  on the machine (`qwen2.5:1.5b`, `gemma2:2b`, `llama3.1:8b`); nothing is
  downloaded from a registry.
- Cost/blur route estimates replaced with measured values from
  `scripts/bench_latency.py`.
- All Ollama calls share one keep-alive HTTP session and carry explicit
  `keep_alive` — a cold model load (2.3-3.8 s) no longer sits in the
  critical path of every first request.

### Fixed
- `keep_alive` sent as a string (`"-1"`) made `/api/embed` answer 400, which
  broke the semantic cache; numeric values are now normalised before sending.
- Startup warm-up preloaded every tier including the 4.9 GB 8B model, which
  pushed a 16 GB machine into swap and slowed every subsequent request.
- A failing pooled bridge now degrades to cloud/local tiers instead of
  raising a 500 (`/pooled/code` returns 503/502 with the upstream message).
- Streaming responses from bridges that omit `usage` report an estimated
  token count flagged `usage_estimated` instead of a misleading 0.
