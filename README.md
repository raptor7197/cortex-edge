# CortexEdge

**Quality-Constrained and Resource-Aware Adaptive Orchestration of Large
Language Models Across Mobile, Edge and Cloud Platforms**

Adaptive hybrid edge-AI runtime for LLMs on Raspberry Pi 5 / Android. The core
research contribution is a **quality-constrained, resource-aware routing
system** that selects the least expensive execution route (small local model /
medium local model / local RAG / cloud fallback) capable of meeting a required
response-quality target.

The full plan lives in [PLAN.md](PLAN.md); the routed prototype this project
grew from is in `mvp/` (kept for reference).

---

## 1. Quickstart (dev machine)

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env          # edit if needed (defaults are local-only)

make run                      # uvicorn app.api.server:app --port 8000
# or: uvicorn app.api.server:app --host 127.0.0.1 --port 8000

make ui                       # streamlit UI in another terminal
make test                     # pytest tests/
```

The backend auto-detects which model tiers are pulled (Ollama). Default models:

| tier | model | notes |
|---|---|---|
| small | `qwen2.5:0.5b` | fast, low quality (~0.7 s) |
| medium | `gemma2:2b` | 2B quality tier (~3 s) |
| moe | `qwen2.5:1.5b` | placeholder until a true MoE model is available |
| large | `qwen3:4b` | slow on CPU; enabled only if pulled |

## 2. API

```
GET  /health          system state, enabled routes, RAG/DB status
POST /query           non-streaming (benchmarks)
POST /query/stream    NDJSON streaming (TTFT, token counts, route reason)
GET  /db/stats        experiment logger summary
```

```bash
curl -X POST 127.0.0.1:8000/query -H 'Content-Type: application/json' -d '{
  "text": "Explain what a GPU does in two sentences.",
  "router": "rule", "policy": "public"}' | python3 -m json.tool
```

`QueryRequest` fields: `text`, `session_id`, `router` (`rule|cost|blur`),
`policy` (`public|private|restricted|ephemeral`), `offline_only`, `private`,
`use_documents` (RAG), `use_search` (DDG), `quality_priority`,
`latency_budget_s`, `max_tokens`, `model` (`auto|small|medium|moe|large`).

### Routers

- **rule** — transparent baseline (PLAN §7): policy -> resource pressure ->
  complexity thresholds. Every decision logs a reason.
- **cost** — the research contribution (PLAN 16.1): min weighted cost
  (latency, energy, memory, cloud $, privacy risk) among routes meeting the
  quality target; falls back to best quality if none feasible.
- **blur** — budget-latency-aware utility routing (ported from the MVP):
  maximize quality within a latency budget. Good for interactive UX.

### Privacy policies (PLAN §10)

| policy | cloud | cache | logging |
|---|---|---|---|
| public | permitted | exact + semantic | full prompt/response stored |
| private | never | none | hashes only |
| restricted | never | none | aggregate metrics only |
| ephemeral | never | none | nothing on disk |

## 3. RAG (local document answering)

```bash
mkdir -p datasets/documents        # drop PDFs/txt/md/docx in here
make ingest                        # chunk (450/70 words), embed, FAISS index
curl -X POST 127.0.0.1:8000/query/stream -H 'Content-Type: application/json' \
  -d '{"text":"What does the doc say about X?","use_documents":true}'
```

Embeddings: sentence-transformers `all-MiniLM-L6-v2` when installed; a
zero-dependency hashing embedder keeps the pipeline working otherwise. FAISS
`IndexFlatIP` with a numpy fallback.

> Note: on GPU-less machines the PyPI `torch` wheel ships CUDA binaries that
> fail at import (`libcublas.so not found`). The whole RAG/cache stack then
> degrades gracefully to the hash embedder. To enable transformer embeddings,
> install a torch build matched to your hardware
> (`pip install torch --index-url https://download.pytorch.org/whl/cpu` on
> x86-64 Linux), then rerun `make ingest`.

## 4. Dataset & learned router (Phases 5–6)

```bash
make dataset    # prompts x routes -> routing_training.csv, cost-labelled best_route
make train      # RandomForest classifier -> experiments/router.joblib
```

The server loads `experiments/router.joblib` at startup if present
(`/health` reports `learned_router: true`). Training prompts are in
`datasets/prompts.txt` (format: `task class | prompt`).

## 5. Benchmarks, energy, speech

```bash
make benchmark          # experiments/results/route_benchmark.csv
python3 scripts/build_dataset.py --skip-cloud   # without a cloud key
scripts/log_power.py    # INA219 on a Raspberry Pi only
```

- STT/TTS: `app/speech/wrappers.py` (whisper.cpp + Piper) — binaries and
  models are Pi-deployment concerns; the wrappers give clear errors locally.
- Energy: net energy = query energy − (idle power × query duration);
  ≥5 repeats after warm-up (PLAN §9).

## 6. Raspberry Pi deployment notes

- `BACKEND=openai` in `.env`, run two `llama-server` instances on ports
  8101/8102 (PLAN 15.1), or use Ollama ARM64 — the code supports both.
- Install `adafruit-circuitpython-ina219` + wire the INA219 on I2C for
  power logging.
- Speech models go in `models/stt/` (whisper ggml) and `models/tts/` (piper).

## 7. Repository layout

```
app/
  api/server.py          FastAPI orchestrator
  router/                features, rule/cost/blur/learned routers
  inference/             llm_client (ollama + OpenAI backends, circuit
                         breaker), model_manager
  rag/                   ingest, retriever, embeddings
  cache/                 exact + semantic cache (FAISS-backed)
  monitoring/            system state, privacy-gated SQLite logger
  memory/                session store with rolling summary compaction
  tools/                 web search (DDG + Exa)
  speech/                whisper.cpp / Piper wrappers
  ui.py                  Streamlit UI
scripts/                 benchmark, dataset build, router training,
                         ingest, power logging
datasets/                prompts.txt, documents/, faiss.index, chunks.json
experiments/             cortexedge.db, results/, router.joblib
tests/                   pytest suite (32 tests)
mvp/                     the routed prototype this project grew from
```

## 8. Reproducibility discipline

- Identical prompts & generation settings across routes (`temperature 0.2`,
  `stream_chat` for TTFT).
- Latency reported as median over repeats; P50/P95 reported in Phase 12.
- Quality regret = quality lost vs. best-available route per query (label
  generation in `scripts/build_dataset.py`).