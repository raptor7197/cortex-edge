# Local latency: measured numbers and the playbook

Everything here was measured on the dev box (16-core laptop, **no dGPU**,
16 GB RAM, ~5 GB of swap in use, browser + node running) with the GGUF
models already on disk, served by Ollama on the CPU. Numbers are medians
over repeated requests; re-run `scripts/bench_latency.py` and
`scripts/bench_end_to_end.py` to reproduce them on your machine.

Models (all imported from GGUF files already on disk, nothing downloaded —
see `scripts/import_local_models.sh`):

| tier   | model          | size on disk |
|--------|----------------|--------------|
| small  | qwen2.5:1.5b   | 986 MB       |
| medium | gemma2:2b      | 1.7 GB       |
| moe    | qwen2.5:1.5b   | placeholder  |
| large  | llama3.1:8b    | 4.9 GB       |

## 1. Where the time actually goes

A CPU-only generation request splits into three parts:

```
load (mmap + graph init)  ->  prefill (prompt tokens)  ->  decode (output tokens)
   2.3 s - 3.8 s                 50-110 ms                  100 ms/token
```

Measured, 64 output tokens, streaming (`scripts/bench_end_to_end.py`):

| phase | tier   | TTFT     | total    | load    | decode     |
|-------|--------|----------|----------|---------|------------|
| cold  | small  | 2957 ms  | 7085 ms  | 2333 ms | 16.1 tok/s |
| warm  | small  | 277 ms   | 4453 ms  | 194 ms  | 16.0 tok/s |
| tuned | small  | 255 ms   | 3243 ms  | 182 ms  | 22.9 tok/s |
| cold  | medium | 5018 ms  | 11749 ms | 3832 ms | 9.9 tok/s  |
| warm  | medium | 379 ms   | 6947 ms  | 241 ms  | 9.9 tok/s  |
| tuned | medium | 368 ms   | 5693 ms  | 239 ms  | 11.2 tok/s |

- **cold -> tuned: 54% less wall time on `small`, 52% on `medium`.**
- **TTFT drops from 3.0 s to 0.26 s** — that is the number a user feels.
- The reload cost (`load`) is 2.3-3.8 s: it dwarfs everything else, which
  is why `keep_alive` is the single biggest lever.

Through the running API (`/query`, 16-20-token answers, `small` tier), with
one extra bug fixed that this measurement exposed — the *embedder* was
holding one of Ollama's two resident slots forever, so nearly every request
paid a reload:

| before (embedder pinned, 2 slots) | after (`OLLAMA_EMBED_KEEP_ALIVE=30m`, 3 slots) |
|---|---|
| 3987 / 1119 / 1836 / 1416 ms, 4.8-13.4 tok/s | 1392 / 994 / 1239 / 1395 ms, **11.5-16.1 tok/s** |

All three models (`qwen2.5:1.5b`, `gemma2:2b`, `nomic-embed-text`) stay
resident, each loaded with the same `num_ctx` the requests use.

## 2. What was changed (and why)

| lever | setting | measured effect |
|---|---|---|
| never unload the model | `OLLAMA_KEEP_ALIVE=-1` | removes the 2.3-3.8 s reload; TTFT 2957 ms -> 277 ms |
| preload only the tiers the router uses | `OLLAMA_WARMUP_ROUTES=small,medium` | preloading the 8B tier pushed 5 GB into swap and slowed every request; the 8B tier now loads on demand |
| warm up with the *same* options as real requests | warm-up sends `ollama_options()` | a warm-up with a different `num_ctx` leaves the model loaded in the wrong shape, so the first real request reloads it |
| don't let the embedder hold a resident slot | `OLLAMA_EMBED_KEEP_ALIVE=30m` | the embedder was pinned forever in 1 of 2 slots, so chat models were evicted constantly: 3.5-4.1 s/request -> 1.0-1.4 s |
| more scheduler slots | `OLLAMA_MAX_LOADED_MODELS=3` (server env) | small + medium + embedder stay resident together |
| explicit CPU thread count | `OLLAMA_NUM_THREAD=12` | 22.9 tok/s vs 16.0 default on `small` (+43%); 11.2 vs 9.9 on `medium` (+13%) |
| trim the KV cache | `OLLAMA_NUM_CTX=2048` | neutral on short prompts, ~half the KV memory -> less swap on long RAG prompts |
| prefill batch | `OLLAMA_NUM_BATCH=512` | 1024 measured neutral here; keep 512 |
| reuse one HTTP connection | `requests.Session` in `llm_client`/`pooled_client` | removes a TCP handshake per request (small but free) |
| stream tokens | `/query/stream` (NDJSON) | first token at 0.26 s instead of waiting 3-7 s for the full answer |
| cache repeats | exact + semantic cache | identical question: **0.0 ms** (measured `/query` cache hit) |
| route to the cheapest feasible tier | `cost` router | `small` is ~2x faster than `medium`; the router only pays for quality the query needs |

`num_thread=16` (all cores) is **worse** than 12 — 10.1 tok/s on `small`,
5.7 on `medium` — because the browser, node and sshd are competing for the
same cores. The measured sweep:

| threads | small tok/s | medium tok/s |
|---|---|---|
| 4  | 17.0 | 10.1 |
| 6  | 20.7 | 11.4 |
| 8  | 20.9 | 11.5 |
| 10 | 18.6 | 10.8 |
| 12 | **23.4** | **11.6** |
| 16 | 10.1 | 5.7 |

## 3. Not available on this stack (checked, don't re-invent)

- **Speculative decoding** (what pooled uses to raise tok/s): Ollama does
  not expose draft models; llama.cpp's `--draft-model` would need a second
  model file, and the constraint here is "no new downloads". Skipped.
- **GPU offload** (`num_gpu`): no dGPU on this laptop. On the Raspberry Pi
  5 path the same knobs apply, minus threads.
- **flash attention / q8_0 KV cache**: enabled on the server
  (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`); on CPU the
  win is small (kernel is memory-bound) but it halves KV memory.

## 4. Playbook for a slow answer

1. `bash scripts/restart_api.sh` — startup warm-up preloads `small`+`medium`.
2. Check `/health`: `ollama.reachable`, `enabled_routes`, and
   `system.loaded_models`. Two resident models is the practical maximum on
   16 GB; more means swap and a slower *every* request.
3. If `large` (llama3.1:8b) was used, unload it before interactive work —
   it is 4.9 GB and will not stay out of swap:
   `curl -s host:11434/api/chat -d '{"model":"llama3.1:8b","messages":[],"keep_alive":0}'`.
4. Long prompts (RAG) pay prefill: keep `OLLAMA_NUM_CTX` just above the
   real context size, and let `local_rag` route to `medium`.
5. Measure, don't guess: `python scripts/bench_latency.py` (option sweep)
   and `python scripts/bench_end_to_end.py` (cold vs warm vs tuned).
