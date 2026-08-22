import json
import time
import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

MODELS = {
    "auto (BLUR routing)": "auto",
    "small (qwen2.5:0.5b)": "small",
    "medium (gemma2:2b)": "medium",
    "moe (qwen2.5:1.5b)": "moe",
    "large (qwen3:4b)": "large",
}

st.set_page_config(page_title="CortexEdge MVP", page_icon="⚡", layout="wide")

st.title("⚡ CortexEdge — Local LLM Showcase")
st.caption("All inference runs locally via Ollama. No data leaves this machine.")


def fmt_time(ms: float) -> str:
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms"


if "chat" not in st.session_state:
    st.session_state.chat = []
if "perf" not in st.session_state:
    st.session_state.perf = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = f"ses-{uuid.uuid4().hex[:12]}"

with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Model", list(MODELS.keys()), index=0)
    route = MODELS[model_choice]

    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=128)

    if route == "auto":
        latency_budget = st.slider("Latency budget (s)", 0.5, 15.0, 2.0, 0.5)
        quality_priority = st.slider("Quality priority", 0.0, 1.0, 0.5, 0.1)
    else:
        latency_budget, quality_priority = 2.0, 0.5

    use_search = st.toggle("Web search (DuckDuckGo)", value=False)
    new_session = st.button("New session", type="secondary")
    if new_session:
        st.session_state.session_id = f"ses-{uuid.uuid4().hex[:12]}"
        st.rerun()
    st.caption(f"session: `{st.session_state.session_id[:16]}…`")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"API online · routes: {', '.join(health['enabled_routes'])}")
        for r, info in health["models"].items():
            st.caption(f"{r} → {info['ollama_name']}")
    except Exception:
        st.error("API server offline\nRun: `uvicorn mvp.server:app --port 8000`")

st.divider()

if st.session_state.chat:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "stats" in msg:
                st.caption(msg["stats"])

prompt = st.chat_input("Ask anything…")


def run_query(prompt: str, route: str):
    state = {"text": "", "ttft_ms": None, "tokens": 0, "final": None}
    start = time.perf_counter()
    body = {
        "text": prompt,
        "model": route,
        "max_tokens": max_tokens,
        "session_id": st.session_state.session_id,
        "use_search": use_search,
    }
    if route == "auto":
        body["latency_budget_s"] = latency_budget
        body["quality_priority"] = quality_priority
    r = requests.post(
        f"{API_URL}/query/stream",
        json=body,
        stream=True,
        timeout=240,
    )
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
            if not delta:
                continue
            if state["ttft_ms"] is None:
                state["ttft_ms"] = (time.perf_counter() - start) * 1000
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
            r, state, token_stream = run_query(prompt, route)
            placeholder.write_stream(token_stream)
            stats_text = ""
            if state["final"]:
                f = state["final"]
                stats = {
                    "model": f["model"],
                    "route": f.get("route", "?"),
                    "latency_ms": f["latency_ms"],
                    "ttft_ms": f["ttft_ms"],
                    "tokens_per_second": f["tokens_per_second"] or 0,
                    "completion_tokens": f["completion_tokens"],
                }
                stats_text = (
                    f"model: **{stats['model']}** · route: `{stats['route']}` · "
                    f"TTFT: **{fmt_time(stats['ttft_ms'])}** · "
                    f"total: **{fmt_time(stats['latency_ms'])}** · "
                    f"speed: **{stats['tokens_per_second']:.1f} tok/s** · "
                    f"tokens: **{stats['completion_tokens']}**"
                )
                st.caption(stats_text)
                key = stats["model"]
                st.session_state.perf.setdefault(key, []).append(stats)
            else:
                stats_text = f"tokens: **{state['tokens']}**"
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
st.subheader("Real-time performance")

col_metrics, col_clear = st.columns([4, 1])
with col_metrics:
    all_perf = [p for lst in st.session_state.perf.values() for p in lst]
    if all_perf:
        last = all_perf[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Last total latency", fmt_time(last["latency_ms"]))
        m2.metric("Speed", f"{last['tokens_per_second']:.1f} tok/s")
        m3.metric("Completion tokens", f"{last['completion_tokens']}")
with col_clear:
    if st.button("Clear history", type="secondary"):
        st.session_state.chat = []
        st.session_state.perf = {}
        st.rerun()

if st.session_state.perf:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Token speed over queries (tok/s)**")
        st.line_chart(
            {
                name: [p["tokens_per_second"] for p in lst]
                for name, lst in st.session_state.perf.items()
                if lst
            }
        )
    with c2:
        st.markdown("**Latency over queries (ms)**")
        st.line_chart(
            {
                name: [p["latency_ms"] for p in lst]
                for name, lst in st.session_state.perf.items()
                if lst
            }
        )
else:
    st.info("Send a message above to see live performance graphs.")
