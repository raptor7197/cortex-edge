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

# local models: nothing is downloaded — existing GGUF files are imported
# into Ollama (see scripts/import_local_models.sh and .env for paths)
bash scripts/start_ollama_local.sh      # on the machine that runs Ollama
bash scripts/import_local_models.sh     # GGUF -> ollama, no network fetch

make run                      # uvicorn app.api.server:app --port 8000
# or: bash scripts/restart_api.sh       # restart + preload the tier models
#     bash scripts/smoke_test.sh        # start, exercise every route, print stats

make ui                       # streamlit UI in another terminal
make test                     # pytest tests/
```

The backend auto-detects which model tiers Ollama can serve. Defaults name
models that are imported from GGUF files already on disk — nothing is pulled
from a registry:

| tier | model | notes |
|---|---|---|
| small | `qwen2.5:1.5b` | fastest tier; ~22 tok/s tuned, TTFT ~0.3 s warm |
| medium | `gemma2:2b` | quality tier; ~11 tok/s on CPU |
| moe | `qwen2.5:1.5b` | placeholder until a true MoE GGUF is available |
| large | `llama3.1:8b` | 4.9 GB; slow on CPU and pushes a 16 GB box into swap |
| pooled | *a browser room* | peer-to-peer inference, no local model — §6 |

`OLLAMA_API_URL=auto` probes `localhost:11434` and then this host's default
gateway, so the API can run in a container while Ollama runs on the laptop.
Latency tuning (thread count, context size, keep-alive, preloading) is in
[docs/LOCAL_LATENCY.md](docs/LOCAL_LATENCY.md).

## 2. API

```
GET  /health          system state, enabled routes, pooled/cloud/RAG status
GET  /routes          route table with the cost/latency/quality estimates
POST /query           non-streaming (benchmarks)
POST /query/stream    NDJSON streaming (TTFT, token counts, route reason)
POST /pooled/code     one agent step on a pooled room (tool calling)
GET  /pooled/status   pooled bridge reachability
GET  /db/stats        experiment logger summary
```

```bash
curl -X POST 127.0.0.1:8000/query -H 'Content-Type: application/json' -d '{
  "text": "Explain what a GPU does in two sentences.",
  "router": "rule", "policy": "public"}' | python3 -m json.tool
```

`QueryRequest` fields: `text`, `session_id`, `router` (`rule|cost|blur|learned`),
`policy` (`public|private|restricted|ephemeral`), `offline_only`, `private`,
`use_documents` (RAG), `use_search` (DDG), `quality_priority`,
`latency_budget_s`, `max_tokens`,
`model` (`auto|small|medium|moe|large|pooled`), `pooled_tools`.

### Routers

- **rule** — transparent baseline (PLAN §7): policy -> resource pressure ->
  complexity thresholds. Complex work offloads to `pooled` (free, own
  devices) before paid `cloud`. Every decision logs a reason.
- **cost** — the research contribution (PLAN 16.1): min weighted cost
  (latency, energy, memory, cloud $, privacy risk) among routes meeting the
  quality target; falls back to best quality if none feasible.
- **blur** — budget-latency-aware utility routing (ported from the MVP):
  maximize quality within a latency budget. Good for interactive UX.
- **learned** — RandomForest over query features (`experiments/router.joblib`).

### Privacy policies (PLAN §10)

`cloud` and `pooled` both leave the device, so both obey the same gate.

| policy | cloud | pooled | cache | logging |
|---|---|---|---|---|
| public | permitted | permitted | exact + semantic | full prompt/response stored |
| private | never | never | none | hashes only |
| restricted | never | never | none | aggregate metrics only |
| ephemeral | never | never | none | nothing on disk |

## 3. RAG (local document answering)

```bash
mkdir -p datasets/documents        # drop PDFs/txt/md/docx in here
make ingest                        # chunk (450/70 words), embed, FAISS index
curl -X POST 127.0.0.1:8000/query/stream -H 'Content-Type: application/json' \
  -d '{"text":"What does the doc say about X?","use_documents":true}'
```

Embeddings: Ollama `nomic-embed-text` (768-dim, already on the box) is used
when reachable; sentence-transformers `all-MiniLM-L6-v2` is the fallback, and a
zero-dependency hashing embedder (512-dim) keeps the pipeline working with no
model at all. FAISS `IndexFlatIP` with a numpy fallback. The semantic cache
detects an embedder change and skips records of the old dimension instead of
mixing vector spaces.

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

## 6. Pooled route — peer-to-peer browser inference

[`pooled`](https://github.com/Nehanth/pooled) runs one open model across the
GPUs of every device in a *room* (browser tabs, WebGPU + WebRTC) and can serve
that room over an OpenAI-compatible bridge. CortexEdge treats it as a route:
cloud-grade quality, **no per-token cost**, no local RAM — but it leaves the
device, so it is gated exactly like cloud.

```bash
bash scripts/pooled_serve.sh "https://pooled.run/r/<code>#k=..."   # join a room
curl -s localhost:8000/pooled/status
# without a room, use the stand-in bridge to exercise the route end to end:
python scripts/pooled_mock_bridge.py --port 8080 --upstream-model qwen2.5:1.5b
bash scripts/verify_pooled_route.sh
```

Design, borrowed capabilities (tool calling / Code mode) and limits:
[docs/POOLED_INTEGRATION.md](docs/POOLED_INTEGRATION.md).

## 7. Local latency

Measured, reproducible numbers and the tuning playbook:
[docs/LOCAL_LATENCY.md](docs/LOCAL_LATENCY.md). Short version — cold to tuned
is 53% less wall time on the `small` tier and TTFT drops from 3.0 s to 0.28 s:

```bash
python scripts/bench_latency.py        # option sweep (threads/ctx/batch)
python scripts/bench_end_to_end.py     # cold vs warm vs tuned
```

## 8. Raspberry Pi deployment notes

- `BACKEND=openai` in `.env`, run two `llama-server` instances on ports
  8101/8102 (PLAN 15.1), or use Ollama ARM64 — the code supports both.
- Install `adafruit-circuitpython-ina219` + wire the INA219 on I2C for
  power logging.
- Speech models go in `models/stt/` (whisper ggml) and `models/tts/` (piper).

## 9. Repository layout

```
app/
  api/server.py          FastAPI orchestrator
  router/                features, rule/cost/blur/learned routers
  inference/             llm_client (ollama + OpenAI backends, circuit
                         breaker, keep-alive session), pooled_client
                         (P2P browser rooms), model_manager
  rag/                   ingest, retriever, embeddings (ollama > ST > hash)
  cache/                 exact + semantic cache (FAISS-backed)
  monitoring/            system state, privacy-gated SQLite logger
  memory/                session store with rolling summary compaction
  tools/                 web search (DDG + Exa)
  speech/                whisper.cpp / Piper wrappers
  ui.py                  Streamlit UI
scripts/                 smoke_test, restart_api, start_ollama_local,
                         import_local_models, pooled_serve, verify_pooled_route,
                         pooled_mock_bridge, bench_latency, bench_end_to_end,
                         benchmark_routes, dataset build, router training,
                         ingest, power logging
docs/                    LOCAL_LATENCY.md, POOLED_INTEGRATION.md
datasets/                prompts.txt, documents/, faiss.index, chunks.json
experiments/             cortexedge.db, results/, router.joblib
tests/                   pytest suite (50 tests)
mvp/                     the routed prototype this project grew from
```

## 10. Reproducibility discipline

- Identical prompts & generation settings across routes (`temperature 0.2`,
  `stream_chat` for TTFT).
- Latency reported as median over repeats; P50/P95 reported in Phase 12.
- Quality regret = quality lost vs. best-available route per query (label
  generation in `scripts/build_dataset.py`).