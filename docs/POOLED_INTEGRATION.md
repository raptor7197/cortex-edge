# Integrating `pooled` into CortexEdge

Upstream: <https://github.com/Nehanth/pooled> — *peer-to-peer LLM inference
in the browser: pool your devices to run big open models.* Each device in a
room holds a slice of the model's layers and runs them on its own GPU with
WGSL kernels; the hidden state (4-10 KB) hops between tabs over WebRTC, and
the host samples the next token. A room serves its model to any tool that
speaks OpenAI Chat Completions / Responses or Anthropic Messages.

## What was integrated, concretely

| pooled capability | CortexEdge integration |
|---|---|
| a room runs a 27B/35B MoE open model on your own devices | new **`pooled` route**, priced in `app/router/cost.py` with cloud-grade quality (0.93) and **zero** $ per 1k queries |
| the room's `@pooled/cli serve` bridge speaks OpenAI Chat Completions | `app/inference/pooled_client.py` — one reused HTTP session, SSE streaming, token usage, circuit breaker |
| serving to agents: tool calling, JSON mode, reasoning | `run_pooled_tools()` + `POST /pooled/code` (one agent step, default file/list/read/write tool set) — pooled grammar-constrains calls to the declared schema |
| the room is remote hardware: prompts reach the host and the room | policy gate: `pooled` is treated like cloud for privacy — refused for `private`, `restricted`, `ephemeral` and for `offline_only` |
| fallbacks in the plan (PLAN §8) | `pooled -> cloud -> large -> moe -> medium -> small`; a room that disappears mid-stream finishes on a local tier |
| speculative decoding, batched prefill (pooled's speed tricks) | not portable to Ollama; the local equivalent (`keep_alive`, thread/context tuning, streaming, cache) is in [LOCAL_LATENCY.md](LOCAL_LATENCY.md) |
| Code mode (browser agent that edits files and previews them) | exposed through `/pooled/code`; the room's own host still approves on-disk edits |

Routing consequences (all covered by `tests/test_router_pooled.py`):

- the **rule** router offloads high-complexity queries to `pooled` when a
  room is available, and only falls back to paid `cloud` without one;
- the **cost** router prices `pooled` below `cloud` (same quality class,
  no per-token cost) so a room wins whenever it is feasible;
- **blur** has a `pooled` profile (2.5 s estimated, quality 0.93);
- `pooled` only appears in `enabled_routes` when the bridge answers
  `/v1/models`, so a machine with no room behaves exactly as before.

## Running it

```bash
# 1. open a room and copy its invite link: https://pooled.run/room
# 2. join the room as an API client (Node 22+, no GPU needed here)
bash scripts/pooled_serve.sh "https://pooled.run/r/<code>#k=..."
#    -> pooled room is serving at http://127.0.0.1:8080/v1

# 3. point CortexEdge at it (defaults already match) and restart
bash scripts/restart_api.sh
curl -s localhost:8000/pooled/status
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"text":"Explain CRDTs in two sentences.","model":"pooled","max_tokens":64}'
```

`POOLED_BASE_URL` accepts any OpenAI-compatible endpoint, so the same route
also works against llama.cpp, vLLM or LiteLLM.

### Testing without a room

A room needs other people's browser tabs, so the repo ships a stand-in that
speaks the same protocol and forwards to a local Ollama model:

```bash
python scripts/pooled_mock_bridge.py --port 8080 --upstream-model gemma2:2b
bash scripts/verify_pooled_route.sh          # end-to-end assertions
```

Verified output from that harness (real run, not a mock of the client):

```
enabled_routes: ['large', 'medium', 'moe', 'pooled', 'small']
explicit pooled route -> route: pooled, 37 tokens, 3.3 tok/s
cost router, quality_target=0.95 -> pooled (cloud cost $0.0)
private policy + pooled       -> 400 "Route 'pooled' not available"
streaming: done route=pooled ttft_ms=1035.4 latency_ms=1835.0
```

## Limits worth knowing

- Rooms are for people you would share a document link with: peers compute on
  hidden states and everyone in the room sees questions and answers
  (upstream SECURITY.md). Hence the cloud-like privacy gate.
- Requests run one at a time in the room's queue, and every device adds one
  network round-trip per token; on a slow link the room loses to a local
  1.5B model on latency even though it wins on quality.
- A room that only has one device is just a slow local model in a browser
  tab — pool at least two.
