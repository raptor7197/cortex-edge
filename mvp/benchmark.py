"""Benchmark local Ollama models: latency + tokens/sec per route.

Uses the native /api/chat endpoint for reliable token counts
(prompt_eval_count / eval_count). Prints a comparison table.
"""

import statistics
import time

import requests

OLLAMA_API = "http://localhost:11434/api/chat"

MODELS = {
    "small": "qwen2.5:0.5b",
    "medium": "gemma2:2b",
    "moe": "qwen2.5:1.5b",
    "large": "qwen3:4b",
}

SKIP = {"large"}  # too slow / low RAM

PROMPTS = {
    "greeting": "Say hello in one line.",
    "factual": "What is the capital of France? Answer in one line.",
    "explain": "Explain what a GPU does in two sentences.",
    "summary": "Summarize the Internet in two sentences.",
}

REPEATS = 3


def run_once(model: str, prompt: str):
    start = time.perf_counter()
    r = requests.post(
        OLLAMA_API,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": 256, "temperature": 0.2},
        },
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    elapsed_ms = (time.perf_counter() - start) * 1000
    completion = data.get("eval_count") or 0
    return elapsed_ms, completion, data["message"]["content"]


def main():
    print(
        f"{'route':<8} {'model':<14} {'prompt':<10} {'latency_ms':>12} {'tok/s':>8} {'tokens':>7}",
        flush=True,
    )
    print("-" * 66, flush=True)
    for route, model in MODELS.items():
        if route in SKIP:
            print(f"{route:<8} {model:<14} skipped (low RAM)", flush=True)
            continue
        for prompt_name, prompt in PROMPTS.items():
            samples = [run_once(model, prompt) for _ in range(REPEATS)]
            latencies = [s[0] for s in samples]
            tps = [s[1] / (s[0] / 1000) if s[0] else 0 for s in samples]
            tokens = [s[1] for s in samples]
            med_latency = statistics.median(latencies)
            med_tps = statistics.median(tps)
            med_tokens = statistics.median(tokens)
            print(
                f"{route:<8} {model:<14} {prompt_name:<10} "
                f"{med_latency:>9.0f}ms {med_tps:>7.1f} {med_tokens:>6.0f}",
                flush=True,
            )
            time.sleep(1)
        print(flush=True)


if __name__ == "__main__":
    main()
