#!/bin/bash
# Join a pooled room as an API client and expose it to CortexEdge.
#
#   bash scripts/pooled_serve.sh "https://pooled.run/r/4TKG9P#k=..."
#   bash scripts/pooled_serve.sh --stop
#
# The bridge listens on 127.0.0.1:8080 and speaks OpenAI Chat Completions,
# OpenAI Responses and Anthropic Messages. CortexEdge picks it up as the
# `pooled` route (POOLED_BASE_URL). Needs Node 22+; no GPU on this host.
set -u
ROOM_LINK="${1:-}"
PORT="${POOLED_PORT:-8080}"
LOG="${HOME}/.pooled-bridge.log"

NODE=$(ls -d /home/krxsna/.hermes/tools/node-*/bin 2>/dev/null | head -1)
[ -n "${NODE:-}" ] && export PATH="${NODE}:${PATH}"

if [ "$ROOM_LINK" = "--stop" ]; then
  pids=$(pgrep -f "@pooled/cli serve" || true)
  [ -n "$pids" ] && kill $pids && echo "stopped pooled bridge: $pids"
  exit 0
fi

if [ -z "$ROOM_LINK" ]; then
  echo "usage: $0 \"https://pooled.run/r/<code>#k=...\"   (or --stop)" >&2
  echo "open a room at https://pooled.run/room on the devices you want to pool," >&2
  echo "then use the invite link from the room's Serve API page." >&2
  exit 2
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx not found; install Node 22+ or use scripts/pooled_mock_bridge.py for local tests" >&2
  exit 1
fi

if curl -sf -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "a pooled bridge is already serving on :${PORT}"
  exit 0
fi

echo "starting pooled bridge on :${PORT} (log: ${LOG})"
POOLED_PORT="${PORT}" setsid nohup npx -y @pooled/cli serve "${ROOM_LINK}" >>"${LOG}" 2>&1 < /dev/null &

for _ in $(seq 1 60); do
  sleep 2
  if curl -sf -m 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "pooled room is serving at http://127.0.0.1:${PORT}/v1"
    curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models"
    echo
    exit 0
  fi
done

echo "bridge did not come up; tail of ${LOG}:" >&2
tail -20 "${LOG}" >&2
exit 1
