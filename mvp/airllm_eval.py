"""AirLLM evaluation: layer-wise inference for models too big for RAM.

AirLLM loads one transformer layer at a time from disk — tiny memory
footprint, but slow (disk-bound) and heavy on first run (it decomposes
the model into per-layer shards).

Run:  pip install airllm   (pulls torch; needs ~2 GB)
      python3 mvp/airllm_eval.py --model Qwen/Qwen2.5-1.5B-Instruct

Decision rule: adopt only if a bigger model's quality gain justifies the
latency on this CPU-only laptop. Expected: too slow for interactive chat;
fine for offline quality experiments.
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="Explain what a GPU does in two sentences.")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        from airllm import AirLLMLlama2
    except ImportError as exc:
        print(f"[skip] airllm/torch not installed: {exc}")
        print("Install with: pip install airllm  (expect a large download)")
        return

    print(f"Loading {args.model} via AirLLM (first run decomposes layers — slow)…")
    start = time.perf_counter()
    model = AirLLMLlama2.from_pretrained(args.model)
    load_s = time.perf_counter() - start
    print(f"Loaded in {load_s:.1f}s")

    start = time.perf_counter()
    out = model.generate(
        [args.prompt],
        max_new_tokens=args.max_tokens,
        do_sample=False,
    )
    latency_s = time.perf_counter() - start
    text = out[0][0] if isinstance(out[0], (list, tuple)) else str(out[0])
    print(f"Response ({latency_s:.1f}s, {args.max_tokens} tokens -> "
          f"{args.max_tokens / latency_s:.1f} tok/s):\n{text[:500]}")

    print("\nVerdict thresholds:")
    print("  tok/s >= 8  -> viable for interactive chat")
    print("  tok/s 1-8   -> viable for offline batch/quality experiments")
    print("  tok/s < 1   -> not worth it on this laptop")


if __name__ == "__main__":
    main()
