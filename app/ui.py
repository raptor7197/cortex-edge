"""CortexEdge Streamlit UI (ported from the MVP, extended with policy,
RAG and system-state controls). Run: `streamlit run app/ui.py`."""

import json
import time
import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

ROUTERS = {
    "rule (baseline)": "rule",
    "cost (quality-constrained)": "cost",
    "blur (latency budget)": "blur",
}
POLICIES = {
    "public": "public",
    "private": "private",
    "restricted": "restricted",
    "ephemeral": "ephemeral",
}
MODELS = {
    "auto": "auto",
    "small": "small",
    "medium": "medium",
    "moe": "moe",
    "large": "large",
}

st.set_page_config(page_title="CortexEdge", page_icon="⚡", layout="wide")
st.title("⚡ CortexEdge — Quality-Constrained Edge LLM Routing")
st.caption("Local inference via Ollama/llama.cpp · cloud only when the router allows")


def fmt_time(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


if "chat" not in st.session_state:
    st.session_state.chat = []
if "perf" not in st.session_state:
    st.session_state.perf = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = f"ses-{uuid.uuid4().hex[:12]}"

health = None
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
except Exception:
    pass

with st.sidebar:
    st.header("Configuration")
    router = ROUTERS[st.selectbox("Router", list(ROUTERS.keys()), index=0)]
    model_choice = st.selectbox("Model tier", list(MODELS.keys()), index=0)
    policy = POLICIES[st.selectbox("Privacy policy", list(POLICIES.keys()), index=0)]
    use_documents = st.toggle("Answer from local documents (RAG)", value=False)
    use_search = st.toggle("Web search (DuckDuckGo)", value=False)
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=128)

    if model_choice == "auto" and router == "blur":
        latency_budget = st.slider("Latency budget (s)", 0.5, 15.0, 2.0, 0.5)
        quality_priority = st.slider("Quality priority", 0.0, 1.0, 0.5, 0.1)
    else:
        latency_budget, quality_priority = 2.0, 0.5

    new_session = st.button("New session", type="secondary")
    if new_session:
        st.session_state.session_id = f"ses-{uuid.uuid4().hex[:12]}"
        st.rerun()
    st.caption(f"session: `{st.session_state.session_id[:16]}…`")

    st.divider()
    st.header("System state")
    if health:
        sys_state = health["system"]
        st.metric("Free RAM", f"{sys_state['free_ram_mb'] / 1024:.1f} GB")
        st.metric("CPU load", f"{sys_state['cpu_percent']:.0f} %")
        st.metric("Network", "available" if sys_state["network_available"] else "offline")
        if sys_state.get("cpu_temperature_c"):
            st.metric("Temp", f"{sys_state['cpu_temperature_c']:.1f} °C")
        if sys_state.get("loaded_models"):
            st.caption("loaded: " + ", ".join(sys_state["loaded_models"]))
        st.caption("Routes (tier → model):")
        enabled = set(health.get("enabled_routes", []))
        for route, model in health.get("routes", {}).items():
            suffix = "" if route in enabled else "  _(disabled)_"
            st.markdown(f"**{route}** → `{model}`{suffix}")
        st.divider()
        with st.expander("What does the router do?"):
            st.markdown(
                "**Route** = which model tier answers your query — `small`, "
                "`medium`, `moe`, `large`, `local_rag` (documents) or `cloud`. "
                "The **router** picks the cheapest route that satisfies your "
                "settings; every answer shows `route:` and a short `reason` "
                "explaining the decision (e.g. `cost: quality 0.5 -> large`). "
                "Full details in the **Docs** page."
            )
        if health.get("rag"):
            st.success("RAG index ready")
        else:
            st.warning("RAG index missing — run `make ingest`")
    else:
        st.error("API server offline — run: `uvicorn app.api.server:app --port 8000`")

st.divider()

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "stats" in msg:
            st.caption(msg["stats"])

prompt = st.chat_input("Ask anything…")


def run_query(prompt: str):
    state = {"text": "", "final": None}
    body = {
        "text": prompt,
        "model": model_choice,
        "router": router,
        "policy": policy,
        "use_documents": use_documents,
        "use_search": use_search,
        "max_tokens": max_tokens,
        "session_id": st.session_state.session_id,
        "latency_budget_s": latency_budget,
        "quality_priority": quality_priority,
    }
    r = requests.post(f"{API_URL}/query/stream", json=body, stream=True, timeout=240)
    r.raise_for_status()

    def token_stream():
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in msg:
                yield f"\n[error: {msg['error']}]"
                break
            if msg.get("done"):
                state["final"] = msg
                break
            delta = msg.get("delta", "")
            if delta:
                state["text"] += delta
                yield delta

    return r, state, token_stream()


if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_running locally…_")
        try:
            r, state, token_stream = run_query(prompt)
            placeholder.write_stream(token_stream)
            stats_text = ""
            if state["final"]:
                f = state["final"]
                stats_text = (
                    f"route: `{f.get('route')}` · model: `{f.get('model', '?')}` · "
                    f"TTFT: **{fmt_time(f['ttft_ms'])}** · "
                    f"total: **{fmt_time(f['latency_ms'])}** · "
                    f"speed: **{f['tokens_per_second'] or 0:.1f} tok/s** · "
                    f"tokens: **{f['completion_tokens']}**\n\n"
                    f"_{f.get('reason', '')}_"
                )
                st.caption(stats_text)
            st.session_state.chat.append(
                {"role": "assistant", "content": state["text"], "stats": stats_text}
            )
        except Exception as exc:
            placeholder.markdown(f"**Error:** {exc}")
            st.session_state.chat.append(
                {"role": "assistant", "content": f"**Error:** {exc}"}
            )

st.divider()
st.subheader("Performance per model tier")

col_metrics, col_clear = st.columns([4, 1])
with col_metrics:
    if st.session_state.chat:
        last_assistant = [m for m in st.session_state.chat if m["role"] == "assistant"]
        if last_assistant and last_assistant[-1].get("stats"):
            st.caption(last_assistant[-1]["stats"])
with col_clear:
    if st.button("Clear history", type="secondary"):
        st.session_state.chat = []
        st.rerun()

if health:
    stats = requests.get(f"{API_URL}/db/stats", timeout=2).json()
    st.caption(
        f"Logged requests: {stats.get('total', 0)} · "
        f"avg latency: {fmt_time(stats['avg_latency_ms'])} · "
        f"avg speed: {stats['avg_tokens_per_second']} tok/s · "
        f"by route: {stats.get('by_route', {})}"
    )
