# CortexEdge MVP — Local LLM Showcase

Minimum MVP for local inference on Ollama, with speed comparable to online
services. Includes streaming, BLUR routing, memory (compaction), a response
DB, web search, and a cloud comparison harness.

## Models (routes)
| Route | Ollama model | Notes |
|---|---|---|
| `small` | `qwen2.5:0.5b` | Fastest, ~0.2–0.7 s, ~40–55 tok/s |
| `medium` | `gemma2:2b` | 2B quality tier (~1.6 GB) |
| `moe` | `qwen2.5:1.5b` (placeholder) | True MoE (OLMoE-1B-7B / Qwen1.5-MoE-A2.7B) is **not** on the Ollama registry — requires llama.cpp + more RAM. See IMPROVEMENT_PLAN.md §P1. |
| `large` | `qwen3:4b` | Pulled; slow on this laptop (swap risk) |
| `auto` | — | BLUR routing picks a route from your latency budget + quality priority |

Routes are auto-enabled at server startup based on models already pulled.

## Quick start
```bash
# 1. Ollama running + models pulled
ollama serve
ollama pull qwen2.5:0.5b gemma2:2b qwen2.5:1.5b

# 2. API server
uvicorn mvp.server:app --host 127.0.0.1 --port 8000

# 3. Streamlit UI
streamlit run mvp/ui.py
```

### API examples
```bash
curl http://127.0.0.1:8000/health

# streaming (NDJSON), explicit route
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain GPUs in two sentences.","model":"small","max_tokens":512}'

# BLUR auto-routing with latency budget + quality priority
curl -N -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain GPUs.","model":"auto","latency_budget_s":3.0,"quality_priority":0.8}'

# session memory (compaction keeps prior context)
curl -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"My name is Alice.","model":"small","session_id":"ses-demo"}'
curl -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"What is my name?","model":"small","session_id":"ses-demo"}'

# web search (DuckDuckGo, no key needed)
curl -X POST http://127.0.0.1:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Latest Raspberry Pi model news","model":"small","use_search":true}'

# non-streaming (for benchmarks)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the capital of France?","model":"medium"}'
```

## Benchmark
```bash
python3 mvp/benchmark.py          # latency + tok/s per route (native API, real token counts)
python3 mvp/tests/ -m pytest -q   # unit tests: BLUR, store, compaction
```

## Cloud quality comparison
```bash
cp mvp/.env.example mvp/.env      # add CLOUD_API_KEY (+ CLOUD_MODELS list)
python3 mvp/compare_cloud.py --prompts 10       # local vs cloud, LLM-as-judge
python3 mvp/compare_cloud.py --prompts 10 --judge local   # offline judge (biased)
```
Output: `mvp/compare_results.csv` + console table.

## AirLLM evaluation (bigger models)
```bash
pip install airllm                # pulls torch (~2 GB)
python3 mvp/airllm_eval.py --model Qwen/Qwen2.5-1.5B-Instruct
```
Layer-wise inference runs models that don't fit in RAM — but it's slow and
disk-heavy; expected verdict on this laptop: viable offline, not interactive.

## Storage
- `mvp/cortexedge.db` (SQLite): `responses` (every query server-side),
  `sessions` + `messages` (conversation memory). The responses table seeds
  the Phase-5 routing dataset.
- `mvp/.env` — optional keys: `CLOUD_API_KEY`, `CLOUD_BASE_URL`,
  `CLOUD_MODELS`, `EXA_API_KEY`.

## UI features
- Model selector incl. **auto (BLUR)** with latency-budget + quality sliders.
- Streaming tokens (see text appear as generated), TTFT + total latency
  (adaptive s/ms), real tok/s and token counts.
- Live graphs per model; session control ("New session").
- Web-search toggle (DuckDuckGo).