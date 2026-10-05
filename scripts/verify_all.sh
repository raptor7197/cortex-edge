#!/bin/bash
# Bring the whole dev stack up and verify each surface answers.
#   bash scripts/verify_all.sh
set -u
cd /home/krxsna/dev/edge-cortex || exit 1
PY=.venv/bin/python
rc=0

echo "=== 1. test suite"
$PY -m pytest tests/ -q 2>&1 | tail -3 || rc=1

echo "=== 2. API (restart + warm)"
bash scripts/restart_api.sh || rc=1

echo "=== 3. health"
curl -s -m 10 "http://127.0.0.1:8000/health?refresh=true" | $PY -c 'import json,sys
d=json.load(sys.stdin)
print("  enabled_routes:", d["enabled_routes"])
print("  ollama:", d["ollama"]["base_url"], "reachable:", d["ollama"]["reachable"])
print("  pooled reachable:", d["pooled"]["reachable"])
print("  rag:", d["rag"], "learned_router:", d["learned_router"])' || rc=1

echo "=== 4. one query per available local route"
NONCE="nocache-$(date +%s)"
for r in small medium; do
  printf "  %-6s " "$r"
  curl -s -m 300 -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
    -d "{\"text\":\"Reply with one short sentence about ${r} models. (${NONCE})\",\"model\":\"${r}\",\"max_tokens\":32}" \
    | $PY -c 'import json,sys
d=json.load(sys.stdin)
print("route=%s latency=%sms tps=%s :: %s" % (d.get("route"), d.get("latency_ms"), d.get("tokens_per_second"), (d.get("response") or d.get("detail") or "")[:70].replace("\n"," ")))'
done

echo "=== 5. RAG (local document answering)"
curl -s -m 300 -X POST http://127.0.0.1:8000/query -H 'Content-Type: application/json' \
  -d "{\"text\":\"What does the manual say about CortexEdge? (${NONCE})\",\"use_documents\":true,\"max_tokens\":64}" \
  | $PY -c 'import json,sys
d=json.load(sys.stdin)
print("  route=%s sources=%s :: %s" % (d.get("route"), len(d.get("metadata",{}).get("sources",[])), (d.get("response") or d.get("detail") or "")[:80].replace("\n"," ")))'

echo "=== 6. Streamlit UI"
if curl -sf -m 3 http://127.0.0.1:8501 >/dev/null 2>&1; then
  echo "  UI already running on :8501"
else
  setsid nohup $PY -m streamlit run app/ui.py --server.headless true \
    --server.port 8501 --browser.gatherUsageStats false \
    >>experiments/ui.log 2>&1 < /dev/null &
  for _ in $(seq 1 40); do
    sleep 1
    curl -sf -m 3 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
  done
fi
if curl -sf -m 5 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  echo "  UI health: $(curl -s -m 5 http://127.0.0.1:8501/_stcore/health)"
else
  echo "  UI FAILED to start; tail of experiments/ui.log:"; tail -20 experiments/ui.log; rc=1
fi

echo "=== 7. pooled bridge (stand-in) status"
curl -s -m 5 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 \
  && echo "  stand-in bridge up" || echo "  no bridge running (start scripts/pooled_serve.sh or the mock)"

echo "verify_all exit=$rc"
exit $rc
