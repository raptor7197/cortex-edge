# MVP Improvement Plan

> Source: `notes.md` + current `mvp/` state. Goal: make local inference **fast, higher-quality, and comparable to cloud**, while building the pieces (routing, memory, DB, web access) that the CortexEdge research plan needs.

**Hardware reality check (laptop):** 16-core CPU, Intel Iris Xe (no discrete GPU), 15 GB RAM but only ~5 GB free. All local inference is CPU-bound; any model must fit in ≤4 GB RAM to avoid swap.

---

## P0 — Quick wins (1–2 days)

| # | Item | Why | Action |
|---|---|---|---|
| 1 | **Token bug** | `ui.py` counts tokens as `len(text.split())` — wrong. The streaming endpoint reads `usage` but Ollama's OpenAI-style stream often omits it in the final chunk. | Use Ollama **native API** `POST /api/chat` (returns `prompt_eval_count` + `eval_count` reliably) for the stream; fall back to `tiktoken`/word-count only for display. Verify end-to-end in UI. |
| 2 | **Latency units** | UI shows ms; notes want seconds. | Keep ms internally, display adaptively (`1.2 s` when >1 s). Show both in the stats caption. |
| 3 | **Streaming verification** | `/query/stream` was added but never fully smoke-tested (interrupted). | Restart server, curl the stream, confirm tokens + final stats chunk arrive; fix if the final `"done"` line never comes. |
| 4 | **Max output tokens** | Default 256 caps responses — user wants more output tokens per response. | Raise default to 1024–2048 in `Query` schema + UI slider. |

## P1 — Speed & quality tiers (Week 1–2)

### 1. MoE models (notes: "activate only one expert at a time")
MoE = Mixture of Experts: only a fraction of parameters compute per token → fewer FLOPs → faster CPU inference at same quality budget.

**Candidates (fit ≤4 GB at Q4):**
| Model | Total / Active params | Q4 size | Notes |
|---|---|---|---|
| **OLMoE-1B-7B** | 7B / 1B | ~4.1 GB | Best fit for this laptop; on Ollama as `olmo`/GGUF |
| **Qwen1.5-MoE-A2.7B** | 14.3B / 2.7B | ~8.4 GB | Too big at Q4 for 5 GB free RAM — needs Q3/Q2 or llama.cpp directly; **check llama.cpp/Ollama support first** (historically HF/vLLM only) |
| ~~Qwen3-30B-A3B~~ | 30B / 3B | ~18 GB | Out of budget |
| ~~Mixtral-8x7B~~ | 47B / 13B | ~26 GB | Out of budget |

**Plan:**
1. Pull `olmo` (OLMoE-1B-7B) via Ollama; benchmark against current small/medium (same `benchmark.py`).
2. If A2.7B fits at Q3 via llama.cpp, try it as a third tier.
3. Decision rule: keep MoE tier **only if** latency/energy beat `llama3.2:1b` at similar quality — MoE gains are real but must be measured on CPU.
4. Extend `MODELS` map + UI selector with new tier names.

### 2. 2B model for quality (notes)
Add a ~2B dense tier as the "quality" local option:
- **`gemma2:2b`** (2.6B, Q4 ~1.6 GB) — strong quality-per-byte, good CPU speed.
- or `qwen2.5:3b` (Q4 ~1.9 GB).
Benchmark quality + speed vs. `llama3.2:1b` and vs. cloud. This becomes the `medium` tier; 1b drops to `small-fast`.

### 3. Cloud comparison harness (notes: "compare with cloud models for quality")
Build `mvp/compare_cloud.py`:
1. Fixed prompt set (~30–50 prompts: factual, coding, math, summarization, reasoning).
2. Run every local model **and** cloud models (OpenAI / Anthropic / free-tier OpenRouter — key via `.env`).
3. Score with an **LLM-as-judge** (one model scores both, 1–5 on correctness/completeness) + cheap auto-metrics (ROUGE/BERTScore where applicable).
4. Output table: local vs cloud quality %, latency, tok/s. Target: local ≥85–90% of cloud quality on the prompt set.

## P2 — System pieces (Week 2–3)

### 4. BLUR routing policy (notes: "BLUR ( budget latency …)")
Proposed interpretation — **B**udget-**L**atency-aware **U**tility **R**outing (confirm acronym with author):
- Inputs: query complexity, latency budget (from user/UI), quality floor, cost budget, device state, network state.
- Output: route = `small-fast | medium-quality | moe | cloud` that minimizes cost while satisfying latency budget + quality floor.
- Implementation: extend the current model selector with an **"auto"** mode; first rule-based (per PLAN §7), later learned.
- This is the core **patentable claim** (see §6) — keep the decision trace logged.

### 5. Centralized response DB (notes)
- Add SQLite `mvp/store.py`: table `responses(id, ts, model, route, prompt_hash, prompt, response, latency_ms, ttft_ms, tokens_per_second, completion_tokens, policy, metadata)`.
- Log every query from the API (server-side, not UI-side — UI can crash/close).
- This DB doubles as the Phase-5 routing dataset for the learned router.

### 6. Compaction layer — memory (notes)
The model "remembers" previous context without blowing up the context window:
1. **Rolling summary:** every N turns, ask the medium model to compress the conversation into a summary; prepend as system message; drop old raw turns.
2. **Optional vector memory:** embed past Q&A (sentence-transformers + FAISS), retrieve similar past answers on new queries (mini-RAG on chat history).
3. Implement as a wrapper around the chat history in the API (`/query` accepts `session_id`), so memory survives UI restarts via the DB.
- Patent angle: compacted memory layer in an edge routing system.

### 7. Web access — EXA + Google + DDG (notes)
Tool-use / search integration:
1. **Exa AI API** (`exa.ai`) — neural search, primary (API key in `.env`).
2. **DuckDuckGo** (`ddg`/`duckduckgo_search`) — free fallback, no key.
3. **Google Custom Search JSON API** — optional third backend.
4. Pattern: router detects knowledge-freshness need (or user toggles "search"), model receives search results as context with citations; results cached in the DB to avoid repeat queries.

### 8. AirLLM evaluation (notes: "look into air llm")
- AirLLM = layer-wise inference: loads one transformer layer at a time → runs models that don't fit in RAM, **at the cost of speed and disk I/O** (first run decomposes the model; heavy disk usage).
- On this CPU-only laptop it will be slow — suited for **offline quality experiments with bigger models (7B+)**, not interactive chat.
- Plan: install, try a 7B at Q4, measure tok/s + RAM. **Adopt only if** a larger model's quality gain justifies the latency; otherwise document the tradeoff and skip.

## P3 — Patent & documentation (rolling)

### 9. Design patent outline (notes: "design patent look into the doc")
Patentable elements from the CortexEdge doc + this MVP (draft plain-language claims for a patent attorney):
1. **Quality-constrained multi-tier routing decision engine** — selecting among small/medium/MoE/RAG/cloud routes via weighted cost + quality floor (doc §1.3, §16).
2. **BLUR budget-latency-aware routing policy** (new).
3. **Compaction memory layer** for edge LLM sessions (new).
4. **Resource-aware model switching** with fallback chain and circuit breaker (doc §21.1).
Next: run a novelty search (Google Patents / USPTO), draft provisional application claims, then file (provisional = cheap, 12-month priority).

### 10. Docs & reproducibility
- Update `mvp/README.md` with new tiers, DB schema, search keys, benchmark results table.
- Pin deps in `requirements.txt`; add `.env.example`.

---

## Priority & risk summary

| Item | Priority | Effort | Risk |
|---|---|---|---|
| Token bug + units + streaming verify | P0 | 0.5 d | low |
| Output token limit | P0 | 0.1 d | none |
| 2B quality tier + MoE benchmark | P1 | 1–2 d | medium (MoE support on Ollama unverified) |
| Cloud comparison harness | P1 | 1–2 d | low (needs API key) |
| BLUR routing | P2 | 2–3 d | medium (acronym needs confirmation) |
| Response DB | P2 | 0.5 d | low |
| Compaction memory | P2 | 2 d | medium |
| EXA/DDG/Google search | P2 | 1–2 d | low (keys) |
| AirLLM eval | P2 | 1 d | medium (slow) |
| Patent outline | P3 | 1–2 d | legal review needed |

**Open questions to confirm:** (1) BLUR expansion? (2) which cloud provider for comparison (OpenAI/Anthropic/OpenRouter)? (3) prioritize MoE speed or 2B quality first?
