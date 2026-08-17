# CortexEdge MVP — Local LLM Showcase (laptop)

Minimum MVP proving local inference on Ollama, with speed comparable to online services.

## Models (routes)
| Route | Ollama model | Observed speed (CPU, 16 cores) |
|---|---|---|
| `small` | `qwen2.5:0.5b` | ~200–700 ms, ~40–55 tok/s |
| `medium` | `llama3.2:1b` | ~400 ms–4.7 s, ~17–21 tok/s |
| `large` | `qwen3:4b` | pulled; slow on this laptop (skip) |

## Quick start
```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Run API server
uvicorn mvp.server:app --host 127.0.0.1 --port 8000

# 3. Query it
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain GPUs in two sentences.","model":"small"}'
```

## Benchmark
```bash
python3 mvp/benchmark.py
```

## Demo flow
1. `curl /health` → shows both models available.
2. `POST /query` (small) → response in ~0.2–1.3 s with tokens/sec in the reply.
3. `POST /query` (medium) → better quality but 2–4× slower.