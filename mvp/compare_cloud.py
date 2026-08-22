"""Cloud comparison harness: local models vs cloud models on a fixed
prompt set, scored by LLM-as-judge.

Usage:
    cp .env.example .env   # add OPENROUTER_API_KEY / OPENAI_API_KEY
    python3 mvp/compare_cloud.py [--prompts N] [--judge openrouter]

If no cloud key is set, only local models are benchmarked and the judge
falls back to the best local model (biased — see README).
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests

OLLAMA_API = "http://localhost:11434/api/chat"

LOCAL_MODELS = {
    "small": "qwen2.5:0.5b",
    "medium": "gemma2:2b",
    "moe": "qwen2.5:1.5b",
    "large": "qwen3:4b",
}

CLOUD_MODELS = {}  # filled from env


def _load_env():
    env = Path(__file__).resolve().parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def get_prompt_set(n: int) -> list[str]:
    prompts = [
        "Explain what a GPU does in two sentences.",
        "What is the capital of France?",
        "Write a Python function that returns the nth Fibonacci number.",
        "Summarize the Internet in two sentences.",
        "Solve: if x + 2 = 5, what is x?",
        "List three benefits of renewable energy.",
        "What is the difference between TCP and UDP?",
        "Write a one-line apology email.",
        "Explain the water cycle briefly.",
        "What does 'open source' mean?",
        "Give two examples of a metaphor.",
        "What causes a rainbow?",
        "Translate 'good morning' to three languages.",
        "Name three planets in order of distance from the sun.",
        "What is the main idea of this sentence: Cats are independent "
        "animals that often sleep 12-16 hours per day.",
        "Write a haiku about a computer.",
        "What is 15% of 200?",
        "Suggest two study techniques for exams.",
        "What is a blockchain in one sentence?",
        "Describe the taste of lemon in one sentence.",
    ]
    return prompts[:n]


def run_local(model: str, prompt: str) -> dict:
    start = time.perf_counter()
    r = requests.post(
        OLLAMA_API,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0.2},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["message"]["content"],
        "latency_ms": (time.perf_counter() - start) * 1000,
        "tokens": data.get("eval_count") or 0,
    }


def run_cloud(model: str, prompt: str) -> dict:
    base = os.environ.get("CLOUD_BASE_URL", "https://openrouter.ai/api/v1")
    key = os.environ.get("CLOUD_API_KEY", "")
    if not key:
        raise RuntimeError("no cloud key")
    start = time.perf_counter()
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 200, "temperature": 0.2},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["choices"][0]["message"]["content"],
        "latency_ms": (time.perf_counter() - start) * 1000,
        "tokens": (data.get("usage") or {}).get("completion_tokens") or 0,
    }


def judge(prompt: str, answer: str, judge_model: str) -> dict:
    """LLM-as-judge: score 1-5 for correctness/completeness."""
    if judge_model in LOCAL_MODELS.values():
        return run_local(judge_model, _judge_prompt(prompt, answer))
    return run_cloud(judge_model, _judge_prompt(prompt, answer))


def _judge_prompt(prompt: str, answer: str) -> str:
    return (
        "You are an impartial judge. Score the ANSWER for the QUESTION "
        "from 1 to 5 on correctness and completeness (1=wrong, 5=excellent). "
        "Reply with ONLY a JSON object: {\"score\": N, \"reason\": \"...\"}\n\n"
        f"QUESTION: {prompt}\nANSWER: {answer}"
    )


def parse_score(text: str) -> float:
    import json
    import re

    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return float(json.loads(m.group())["score"])
    except Exception:
        pass
    for i in range(1, 6):
        if str(i) in text[:20]:
            return float(i)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=10)
    ap.add_argument("--judge", default=None,
                    help="judge model id (cloud) or 'local'")
    args = ap.parse_args()

    _load_env()
    cloud_models = [m for m in os.environ.get("CLOUD_MODELS", "").split(",") if m]

    judge_model = args.judge or (cloud_models[0] if cloud_models else "gemma2:2b")
    if args.judge == "local":
        judge_model = "gemma2:2b"

    prompts = get_prompt_set(args.prompts)
    targets = {f"local:{k}": v for k, v in LOCAL_MODELS.items()}
    for m in cloud_models:
        targets[f"cloud:{m}"] = m

    out = Path(__file__).resolve().parent / "compare_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "avg_score", "avg_latency_ms", "avg_tokens"])
        rows = []
        for name, model in targets.items():
            scores, latencies, tokens = [], [], []
            for prompt in prompts:
                try:
                    if name.startswith("cloud:"):
                        res = run_cloud(model, prompt)
                    else:
                        res = run_local(model, prompt)
                except Exception as e:
                    print(f"{name}: skip ({e})", flush=True)
                    continue
                score = 0.0
                try:
                    j = judge(prompt, res["text"], judge_model)
                    score = parse_score(j["text"])
                except Exception:
                    pass
                scores.append(score)
                latencies.append(res["latency_ms"])
                tokens.append(res["tokens"])
                print(
                    f"{name:<18} q{len(scores):<3} score={score:.1f} "
                    f"lat={res['latency_ms']:.0f}ms",
                    flush=True,
                )
            if scores:
                avg_s = sum(scores) / len(scores)
                avg_l = sum(latencies) / len(latencies)
                avg_t = sum(tokens) / len(tokens)
                rows.append((name, avg_s, avg_l, avg_t))
                w.writerow([name, round(avg_s, 2), round(avg_l, 0), round(avg_t, 1)])
                f.flush()
                print(f">>> {name}: avg score {avg_s:.2f} (judge={judge_model})", flush=True)

    print(f"\nSaved to {out}")

    if len(rows) > 1:
        best = max(rows, key=lambda r: r[1])
        print(f"Best quality: {best[0]} ({best[1]:.2f})")
        for name, s, l, t in sorted(rows, key=lambda r: -r[1]):
            print(f"  {name:<18} score={s:.2f} latency={l:.0f}ms tok/s={t / (l / 1000) if l > 0 else 0.0:.1f}")


if __name__ == "__main__":
    main()
