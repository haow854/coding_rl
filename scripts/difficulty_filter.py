"""Difficulty filtering = DAPO-style dynamic sampling, done offline once.

Probe the current policy with k samples per problem and KEEP only those it
solves sometimes-but-not-always (full-pass count in [keep_lo, keep_hi]).
All-fail and all-pass problems give weak GRPO signal, so drop them for the
first clean run. Output is the curated training pool. Runs on the GPU box
(vLLM).

    python scripts/difficulty_filter.py --model Qwen/Qwen3.5-2B \
        --in data/clean_problems.jsonl --out data/train_problems.jsonl \
        --k 4 --keep-lo 1 --keep-hi 3 --max-tokens 1536
"""
import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_clean_jsonl       # noqa: E402
from rlcoder.prompting import build_messages, load_processing_class, render_chat_prompt  # noqa: E402
from rlcoder.rollout.single_turn import score_batch    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--lora", default=None,
                    help="Optional LoRA adapter used while probing difficulty.")
    ap.add_argument("--in", dest="inp", default="data/clean_problems.jsonl")
    ap.add_argument("--out", default="data/train_problems.jsonl")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep-lo", type=int, default=1)
    ap.add_argument("--keep-hi", type=int, default=3)
    ap.add_argument("--max-problems", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--reward-timeout", type=float, default=10.0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    problems = load_clean_jsonl(args.inp, limit=args.max_problems)
    print(f"probing {len(problems)} problems with k={args.k} samples each...")

    proc = load_processing_class(args.model)
    prompts = [
        render_chat_prompt(proc, build_messages(p))
        for p in problems
    ]

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              enable_lora=args.lora is not None,
              max_lora_rank=args.max_lora_rank)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens)
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("adapter", 1, args.lora)
    outputs = llm.generate(prompts, sp, lora_request=lora_req)

    # flatten all completions, score in one concurrent batch, regroup by problem
    comps, tests, owner = [], [], []
    for i, (p, o) in enumerate(zip(problems, outputs)):
        for c in o.outputs:
            comps.append(c.text)
            tests.append(p.tests)
            owner.append(i)
    bds = score_batch(comps, tests, ["stdin_stdout"] * len(comps),
                      timeout=args.reward_timeout, concurrency=128)

    counts = [0] * len(problems)
    for idx, b in zip(owner, bds):
        if b.all_passed:
            counts[idx] += 1

    kept = []
    for p, c in zip(problems, counts):
        p.base_pass_count = c
        if args.keep_lo <= c <= args.keep_hi:
            kept.append(p)

    dist = Counter(counts)
    print(f"pass-count distribution (out of {args.k}): {dict(sorted(dist.items()))}")
    print(f"kept {len(kept)}/{len(problems)} "
          f"(dropped {dist[0]} all-fail + {dist[args.k]} all-pass)")

    with open(args.out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"wrote curated training pool -> {args.out}")


if __name__ == "__main__":
    main()
