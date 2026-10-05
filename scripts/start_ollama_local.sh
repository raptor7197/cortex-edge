#!/bin/bash
# Start Ollama on the dev host with latency-tuned settings so the
# CortexEdge container/dev machine can reach it.
#
#   bash scripts/start_ollama_local.sh          # start if not running
#
# Latency knobs (see docs/LOCAL_LATENCY.md):
#   OLLAMA_KEEP_ALIVE=-1        never unload the model between requests
#   OLLAMA_FLASH_ATTENTION=1    fused attention kernels
#   OLLAMA_KV_CACHE_TYPE=q8_0   half-size KV cache -> less memory traffic
#   OLLAMA_NUM_PARALLEL=1       one request at a time, less scheduler churn
#   OLLAMA_MAX_LOADED_MODELS=2  keep small+medium resident, evict the rest
set -u

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
# slots in the scheduler: small + medium + the embedding model
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-3}"

LOG="${HOME}/.ollama-serve.log"

if [ "${1:-}" = "--restart" ]; then
  pids=$(pgrep -f "ollama serve" || true)
  if [ -n "$pids" ]; then
    echo "restarting ollama (pids: $pids)"
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 15); do
      pgrep -f "ollama serve" >/dev/null 2>&1 || break
      sleep 1
    done
  fi
fi

if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "ollama already running on 127.0.0.1:11434 (use --restart to apply new env)"
else
  echo "starting ollama (host=${OLLAMA_HOST}, kv=${OLLAMA_KV_CACHE_TYPE}, keep_alive=${OLLAMA_KEEP_ALIVE}, max_loaded=${OLLAMA_MAX_LOADED_MODELS})"
  setsid nohup ollama serve >>"${LOG}" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    sleep 1
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      break
    fi
  done
fi

if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "ollama up. listening addresses:"
  ss -ltn 2>/dev/null | grep 11434 || true
  echo "installed models:"
  ollama list
else
  echo "FAILED to start ollama; tail of ${LOG}:" >&2
  tail -20 "${LOG}" >&2
  exit 1
fi
