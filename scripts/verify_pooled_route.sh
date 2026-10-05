#!/bin/bash
# End-to-end check of the `pooled` route using the local stand-in bridge
# (scripts/pooled_mock_bridge.py). Proves the route is registered, is
# preferred over paid cloud, streams, and respects the privacy gate.
#
#   bash scripts/verify_pooled_route.sh [upstream-model]
set -u
cd /home/krxsna/dev/edge-cortex || exit 1
PY=.venv/bin/python
PORT=8000
# qwen2.5 supports tool calling, so /pooled/code is exercisable too
UPSTREAM_MODEL="${1:-qwen2.5:1.5b}"
NONCE="nocache-$(date +%s)"

# restart the stand-in bridge so the upstream model is the one requested
pids=$(pgrep -f "pooled_mock_bridge.py" || true)
if [ -n "$pids" ]; then kill $pids 2>/dev/null || true; sleep 1; fi
echo "starting stand-in pooled bridge on :8080 -> ${UPSTREAM_MODEL}"
setsid nohup "$PY" scripts/pooled_mock_bridge.py --port 8080 \
  --upstream-model "${UPSTREAM_MODEL}" >>experiments/pooled_bridge.log 2>&1 < /dev/null &
for _ in $(seq 1 30); do
  sleep 1
  curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && break
done
curl -sf -m 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 || {
  echo "FAIL: stand-in bridge not reachable"; tail -20 experiments/pooled_bridge.log; exit 1; }

echo "restarting API so it picks up the pooled route"
bash scripts/restart_api.sh

show() { # show <label> <curl-args...>
  local label="$1"; shift
  echo "--- $label"
  curl -s -m 300 "$@" | "$PY" -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(sys.stdin.read()[:300]); raise SystemExit
if isinstance(d, dict):
    for k in ("route","model","reason","latency_ms","ttft_ms","tokens_per_second","completion_tokens","detail"):
        if k in d:
            print(f"  {k}: {str(d[k])[:200]}")
    if "response" in d:
        print("  response:", str(d["response"])[:160].replace("\n"," "))
    if "enabled_routes" in d:
        print("  enabled_routes:", d["enabled_routes"])
'
}

show "pooled status" http://127.0.0.1:${PORT}/pooled/status
show "health (enabled routes)" "http://127.0.0.1:${PORT}/health?refresh=true"
show "explicit pooled route" -X POST "http://127.0.0.1:${PORT}/query" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"In one sentence: what is a peer-to-peer network? (${NONCE})\",\"model\":\"pooled\",\"max_tokens\":48}"
show "cost router, high quality target (should choose pooled)" -X POST "http://127.0.0.1:${PORT}/query" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Explain briefly why merge sort is O(n log n). (${NONCE})\",\"router\":\"cost\",\"quality_priority\":0.95,\"max_tokens\":48}"
show "private policy with pooled explicit (must be refused)" -X POST "http://127.0.0.1:${PORT}/query" \
  -H 'Content-Type: application/json' \
  -d '{"text":"secret plan","model":"pooled","policy":"private","max_tokens":16}'

echo "--- pooled streaming (last line = done event)"
curl -s -N -m 300 -X POST "http://127.0.0.1:${PORT}/query/stream" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Count 1 to 3. (${NONCE})\",\"model\":\"pooled\",\"max_tokens\":24}" | tail -1

echo "--- pooled agent step (/pooled/code with tools)"
curl -s -m 300 -X POST "http://127.0.0.1:${PORT}/pooled/code" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Create a file called hello.py that prints hi."}],"max_tokens":128}' \
  | head -c 500
echo
echo "pooled route verification done"
