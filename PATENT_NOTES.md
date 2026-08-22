# Patent Notes — CortexEdge (preliminary, not legal advice)

Draft claims for a provisional patent application. Review with a patent
attorney; run a novelty search (Google Patents / USPTO / EPO) first.

## Candidate claims

### 1. Quality-constrained multi-tier routing decision engine
A system for selecting an execution route for a language-model query among
a plurality of local and cloud routes, comprising:
- a feature extractor computing a complexity score from query text;
- a device-state monitor (memory, temperature, CPU, battery);
- a network-state monitor;
- a policy engine (privacy level, offline mode, latency/quality priorities);
- a route selector that computes, for each candidate route, a weighted cost
  over latency, energy, memory, cloud cost and privacy risk, and selects the
  minimum-cost route whose predicted quality meets a configurable quality
  floor;
- a decision-trace logger recording route, reason and system state per query.

### 2. BLUR — budget-latency-aware utility routing
A routing policy that selects among model routes by maximizing
`quality − w · (est_latency / budget)` over routes whose estimated latency
is within a user-supplied latency budget, with fallback to the fastest
route when none is feasible; parameters (budget, quality priority) exposed
per-request.

### 3. Compaction memory layer for edge LLM sessions
A method of maintaining conversation state on a resource-constrained
device, comprising: storing session turns in a local database; after a
threshold number of turns, invoking a compacting model to produce a rolling
summary of prior context; injecting the summary as a system message while
retaining only the most recent raw turns; and persisting summary + turns
across client sessions (session_id).

### 4. Resource-aware model switching with fallback chain and circuit breaker
An inference service that: runs multiple local model servers; monitors free
memory and temperature; unloads the larger model under pressure and routes
to a smaller model or cloud fallback; applies a circuit breaker that
temporarily disables a route after repeated failures; and respects privacy
policies by never sending private/restricted prompts to cloud routes.

### 5. Local-first streaming proxy with per-request analytics
A proxy server that streams tokens from a local inference engine to a UI,
computing TTFT, total latency, tokens/sec and token counts in real time,
and persisting all requests/responses to a central response database.

## What is *not* novel (avoid claiming)
- Combining LLM + RAG + STT + TTS (commonplace).
- Running a model locally; using Ollama/llama.cpp.
- Semantic/exact caching generally.

## Next steps
1. Novelty search on claims 1–5.
2. Decide jurisdiction (India provisional vs US provisional).
3. Provisional filing captures filing date cheaply; 12 months to file PCT/full.
4. Document the demo video + code as evidence of reduction to practice.
