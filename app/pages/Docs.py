"""CortexEdge user documentation page (Streamlit multipage app)."""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CortexEdge Docs", page_icon="?", layout="wide")
st.title("CortexEdge — How it works")
st.caption(
    "Quality-constrained, resource-aware routing of LLM queries across edge models. "
    "This page explains the routes, the routing decisions, and what is implemented."
)

health = None
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
except Exception:
    pass

if not health:
    st.error("API server offline — run: `uvicorn app.api.server:app --port 8000`")

st.header("1. Routes (model tiers)")
st.markdown(
    "A **route** is the execution path an answer takes. Local routes run Ollama "
    "models on this machine; `cloud` sends to a hosted API (only when policy and "
    "network allow). The model actually serving each route:"
)

if health:
    enabled = set(health.get("enabled_routes", []))
    rows = [
        {
            "route": route,
            "model": model,
            "status": "enabled" if route in enabled else "not pulled",
        }
        for route, model in health.get("routes", {}).items()
    ]
    rows.append({"route": "local_rag", "model": "retriever + docs", "status": "enabled" if health.get("rag") else "run `make ingest`"})
    rows.append({"route": "cloud", "model": "hosted API (`.env`)", "status": "enabled if network + policy"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.code("API offline — start the server to see live routes")

st.markdown(
    "| route | role |\n"
    "|---|---|\n"
    "| `small` | fastest, lowest quality — greetings, trivial facts |\n"
    "| `medium` | balanced quality/speed for everyday questions |\n"
    "| `moe` | mid tier (placeholder until a true MoE model is pulled) |\n"
    "| `large` | highest local quality — slow on CPU |\n"
    "| `local_rag` | answers grounded in your documents (`datasets/documents/`) |\n"
    "| `cloud` | best quality, costs money, sends data off-device |\n"
    "| `cache` | identical/repeated question answered from memory in ~0 ms |"
)

st.header("2. What the router decides")
st.markdown(
    "The **router** picks which route answers each query, aiming for the cheapest "
    "route that still meets your quality/response goals. Every response reports "
    "`route:` (what ran) and `reason:` (why, in plain text). Example reasons:"
)
st.code(
    "route: medium | reason: Moderate predicted complexity\n"
    "route: large  | reason: cost: quality 0.90 -> large (min cost meeting target)\n"
    "route: small  | reason: Low predicted complexity\n"
    "route: cache  | reason: exact/semantic cache hit",
    language="text",
)

st.markdown("Four routers are available (`router` field in the query or sidebar):")

st.markdown(
    "| router | logic | best for |\n"
    "|---|---|---|\n"
    "| `rule` | fixed rules: policy → CPU/RAM pressure → complexity thresholds | transparent baseline; every decision explainable |\n"
    "| `cost` | weighted cost (latency, energy, RAM, cloud $, privacy) minimised among routes meeting the quality target | research mode; cost-vs-quality tradeoffs |\n"
    "| `blur` | maximises quality within your latency budget | interactive chat UX |\n"
    "| `learned` | ML model (`experiments/router.joblib`) trained on benchmark data; falls back to `rule` when unavailable | after `make dataset && make train` |"
)

st.header("3. Privacy policies")
st.markdown(
    "| policy | cloud allowed | caching | what is stored |\n"
    "|---|---|---|---|\n"
    "| `public` | yes | exact + semantic | full prompt & response |\n"
    "| `private` | never | none | hashes only |\n"
    "| `restricted` | never | none | aggregate metrics only |\n"
    "| `ephemeral` | never | none | nothing on disk |"
)

st.header("4. RAG and memory")
st.markdown(
    "- **RAG**: toggle “Answer from local documents” — the query is answered "
    "only from indexed document chunks, with cited sources.\n"
    "- **Memory**: each conversation has a `session_id`; recent turns are "
    "remembered and older ones compacted into a summary."
)

st.header("5. Speech (STT / TTS) status")
st.markdown(
    "The wrappers exist (`app/speech/wrappers.py`) and the API has a "
    "`response_mode: \"voice\"` field, but they are **not wired into the "
    "running server or this UI yet** — that is a Raspberry Pi deployment "
    "step. Status of each half:\n"
    "- **STT** (`transcribe`): implemented wrapper for whisper.cpp — needs the "
    "`whisper-cli` binary and a ggml model in `models/stt/`.\n"
    "- **TTS** (`synthesize`): implemented wrapper for Piper — needs the "
    "`piper` binary and a voice pack in `models/tts/`.\n"
    "Subtitles/voice output will appear in the UI after the Pi deployment "
    "phase (PLAN Phase 9)."
)

st.header("6. Getting started")
st.code(
    "make run          # API on :8000\n"
    "make ui           # UI on :8501\n"
    "make ingest       # index documents/ for RAG\n"
    "make dataset && make train   # rebuild the learned router",
    language="bash",
)