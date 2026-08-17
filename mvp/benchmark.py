"""Benchmark Ollama local models: latency + tokens/sec per route."""

import json
import statistics
import time

import requests

OLLAMA_BASE = "http://localhost:11434/v1/chat/completions"

MODELS = {
    "small": "qwen2.5:0.5b",
    "medium": "llama3.2:1b",
    "large": "qwen3:4b",
}

PROMPTS = {
    "greeting": "Say hello in one line.",
    "factual": "What is the capital of France? Answer in one line.",
    "explain": "Explain what a GPU does in two sentences.",
    "summary": "Summarize the Internet in two sentences.",
}

REPEATS = 3


SKIP_MODELS = {"large"}


def run_once(model: str, prompt: str):
    start = time.perf_counter()
    r = requests.post(
        OLLAMA_BASE,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.2,
            "stream": False,
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    elapsed_ms = (time.perf_counter() - start) * 1000
    completion = (data.get("usage") or {}).get("completion_tokens") or 0
    return elapsed_ms, completion, data["choices"][0]["message"]["content"]


def main():
    print(f"{'route':<8} {'model':<14} {'prompt':<10} {'latency_ms':>12} {'tok/s':>8}", flush=True)
    print("-" * 60, flush=True)
    for route, model in MODELS.items():
        if route in SKIP_MODELS:
            print(f"{route:<8} {model:<14} skipped (low RAM)", flush=True)
            continue
        for prompt_name, prompt in PROMPTS.items():
            samples = [run_once(model, prompt) for _ in range(REPEATS)]
            latencies = [s[0] for s in samples]
            tps = [
                s[1] / (s[0] / 1000) if s[0] else 0
                for s in samples
            ]
            med_latency = statistics.median(latencies)
            med_tps = statistics.median(tps)
            print(
                f"{route:<8} {model:<14} {prompt_name:<10} "
                f"{med_latency:>9.0f}ms {med_tps:>7.1f}",
                flush=True,
            )
            print(f"  -> {samples[int(len(samples)/2)][2][:80]}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()