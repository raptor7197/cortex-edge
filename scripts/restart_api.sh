#!/bin/bash
# Restart the CortexEdge dev API on :8000 with the current code.
# Only ever terminates the uvicorn process for THIS repo's app.
set -u
cd /home/krxsna/dev/edge-cortex || exit 1

PATTERN="uvicorn app.api.server:app"
pids=$(pgrep -f "$PATTERN" || true)
if [ -n "$pids" ]; then
  echo "stopping old server pids: $pids"
  for pid in $pids; do kill "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 10); do
    pgrep -f "$PATTERN" >/dev/null 2>&1 || break
    sleep 1
  done
fi

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
mkdir -p experiments
setsid nohup "$PY" -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 \
  >>experiments/server.log 2>&1 < /dev/null &

for _ in $(seq 1 40); do
  sleep 1
  if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "server up on :8000"
    exit 0
  fi
done
echo "server failed to start; tail of experiments/server.log:" >&2
tail -30 experiments/server.log >&2
exit 1
