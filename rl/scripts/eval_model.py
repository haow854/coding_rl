"""Evaluate a model on a normalised problem JSONL (our sandbox judge) — use this
for the in-house competitive held-out set and for base-vs-RL comparison.

    # baseline (base model), pass@1 and pass@5 estimated from n samples
    python scripts/eval_model.py --model Qwen/Qwen3-4B \
        --data data/holdout.jsonl --out outputs/eval/base.json

    # after SFT/GRPO (LoRA adapter) — keep sampling identical to the baseline
    python scripts/eval_model.py --model Qwen/Qwen3-4B --lora outputs/qwen3_4b_grpo \
        --data data/holdout.jsonl --out outputs/eval/rl.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_clean_jsonl   # noqa: E402
from rlcoder.eval.run_eval import evaluate         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--data", default="data/holdout.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8,
                    help="Samples/problem. >1 gives a low-variance pass@1 estimate; "
                         "single greedy pass@1 is too noisy for small deltas.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1,
                    help="Set to 20 for Qwen3 report-like thinking eval.")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Matches GRPO --max-completion; raise for very "
                         "long-reasoning problems (slower, less batching).")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM context length. Use 32768+ for report-like long thinking.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Render prompts in non-thinking mode; use the same "
                         "setting as training for base-vs-RL comparisons.")
    ap.add_argument("--ks", default="1,5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problems = load_clean_jsonl(args.data, limit=args.limit)
    ks = [int(x) for x in args.ks.split(",")]
    res, _ = evaluate(args.model, problems, n=args.n, temperature=args.temperature,
                      top_p=args.top_p, top_k=args.top_k,
                      max_tokens=args.max_tokens,
                      max_model_len=args.max_model_len,
                      gpu_mem_util=args.gpu_mem_util,
                      lora_path=args.lora, ks=ks,
                      enable_thinking=not args.no_thinking)
    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "lora": args.lora, "data": args.data,
                       "no_thinking": args.no_thinking, **res}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
