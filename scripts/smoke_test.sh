#!/bin/bash
# End-to-end smoke test for CortexEdge: starts the API, exercises every
# route that is available on this machine, and prints one line per check.
#
#   bash scripts/smoke_test.sh [--port 8000]
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
PORT=8000
LOG=experiments/server.log
mkdir -p experiments

api() { curl -s -m 300 "$@"; }
pretty() { "$PY" -c 'import json,sys;print(json.dumps(json.load(sys.stdin),indent=2)[:2500])'; }

if ! curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "starting uvicorn on :${PORT} (log: ${LOG})"
  setsid nohup "$PY" -m uvicorn app.api.server:app --host 127.0.0.1 --port "${PORT}" \
    >>"${LOG}" 2>&1 < /dev/null &
  for _ in $(seq 1 40); do
    sleep 1
    curl -sf -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  done
fi

if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "FAIL: API did not come up; tail of ${LOG}:" >&2
  tail -30 "${LOG}" >&2
  exit 1
fi

echo "=== /health"
api "http://127.0.0.1:${PORT}/health" | pretty

echo "=== /routes (cost table)"
api "http://127.0.0.1:${PORT}/routes" | pretty

q() { # q <label> <json>
  echo "--- $1"
  api -X POST "http://127.0.0.1:${PORT}/query" -H 'Content-Type: application/json' -d "$2" \
    | "$PY" -c 'import json,sys
d=json.load(sys.stdin)
print("route=%s model=%s latency_ms=%s ttft=%s tps=%s tokens=%s" % (
    d.get("route"), d.get("model"), d.get("latency_ms"), d.get("ttft_ms"),
    d.get("tokens_per_second"), d.get("completion_tokens")))
print("reason:", (d.get("reason") or "")[:160])
print("response:", (d.get("response") or d.get("error") or "")[:200].replace("\n", " "))'
}

echo "=== /query"
q "explicit small (qwen2.5:1.5b)" \
  '{"text":"Reply with exactly one sentence: what is a CPU?","model":"small","max_tokens":64}'
q "explicit medium (gemma2:2b)" \
  '{"text":"In one sentence, what does a GPU do?","model":"medium","max_tokens":64}'
q "cost router, quality target 0.5" \
  '{"text":"Name three sorting algorithms.","router":"cost","quality_priority":0.5,"max_tokens":64}'
q "cost router, quality target 0.9" \
  '{"text":"Explain why merge sort is O(n log n).","router":"cost","quality_priority":0.9,"max_tokens":64}'
q "private policy must stay local" \
  '{"text":"My secret plan is to learn Rust. Acknowledge in five words.","policy":"private","max_tokens":32}'
q "blur router, 3s budget" \
  '{"text":"Summarise what an LRU cache does.","router":"blur","latency_budget_s":3.0,"max_tokens":64}'
q "cache hit (same text as the first small query)" \
  '{"text":"Reply with exactly one sentence: what is a CPU?","model":"small","max_tokens":64}'

echo "--- /query/stream (NDJSON, first+last line)"
api -N -X POST "http://127.0.0.1:${PORT}/query/stream" -H 'Content-Type: application/json' \
  -d '{"text":"Count from 1 to 5.","model":"small","max_tokens":48}' | head -3
echo "..."
api -N -X POST "http://127.0.0.1:${PORT}/query/stream" -H 'Content-Type: application/json' \
  -d '{"text":"Say the word ready.","model":"small","max_tokens":16}' | tail -1

echo "=== /pooled/status"
api "http://127.0.0.1:${PORT}/pooled/status" | pretty

echo "=== /db/stats"
api "http://127.0.0.1:${PORT}/db/stats" | pretty

echo "smoke test done"
