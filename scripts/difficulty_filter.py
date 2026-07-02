"""Difficulty filtering = DAPO-style dynamic sampling, done offline once.

Probe the current policy with k samples per problem and KEEP only those it
solves sometimes-but-not-always (full-pass count in [keep_lo, keep_hi]).
All-fail and all-pass problems give weak GRPO signal, so drop them for the
first clean run. Output is the curated training pool. Runs on the GPU box
(vLLM).

    python scripts/difficulty_filter.py --model Qwen/Qwen3.5-2B \
        --in data/rl_pool.jsonl --out data/grpo_train.jsonl \
        --save-rollouts outputs/filter_rollouts.jsonl \
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


def _maybe_mkdir(path: str | None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _write_rollouts(path: str, problems, outputs) -> None:
    _maybe_mkdir(path)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for problem_index, (p, o) in enumerate(zip(problems, outputs)):
            for sample_index, c in enumerate(o.outputs):
                f.write(json.dumps(
                    {
                        "problem_index": problem_index,
                        "sample_index": sample_index,
                        "source": p.source,
                        "problem_id": p.problem_id,
                        "completion": c.text,
                    },
                    ensure_ascii=False,
                ) + "\n")
                n += 1
    print(f"saved {n} raw rollouts -> {path}", flush=True)


def _load_rollouts(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} raw rollouts <- {path}", flush=True)
    return rows


def _score_with_progress(comps, tests, modes, args):
    total = len(comps)
    bds = []
    batch_size = max(1, args.score_batch_size)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bds.extend(score_batch(
            comps[start:end],
            tests[start:end],
            modes[start:end],
            timeout=args.reward_timeout,
            concurrency=args.score_concurrency,
        ))
        print(f"scored {end}/{total} completions", flush=True)
    return bds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--lora", default=None,
                    help="Optional LoRA adapter used while probing difficulty.")
    ap.add_argument("--in", dest="inp", default="data/rl_pool.jsonl")
    ap.add_argument("--out", default="data/grpo_train.jsonl")
    ap.add_argument("--save-rollouts", default=None,
                    help="Optional JSONL path to save raw generated completions "
                         "before sandbox scoring.")
    ap.add_argument("--load-rollouts", default=None,
                    help="Read previously saved rollouts and only redo sandbox "
                         "scoring/filtering. Use with the same --in ordering.")
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
    ap.add_argument("--score-concurrency", type=int, default=64)
    ap.add_argument("--score-batch-size", type=int, default=512)
    args = ap.parse_args()

    problems = load_clean_jsonl(args.inp, limit=args.max_problems)

    # flatten all completions, score in one concurrent batch, regroup by problem
    comps, tests, owner = [], [], []
    if args.load_rollouts:
        rollout_rows = _load_rollouts(args.load_rollouts)
        for row in rollout_rows:
            i = int(row["problem_index"])
            if i >= len(problems):
                raise ValueError(
                    f"rollout problem_index={i} exceeds loaded problems={len(problems)}; "
                    "use the same --in and --max-problems as generation"
                )
            comps.append(row["completion"])
            tests.append(problems[i].tests)
            owner.append(i)
    else:
        from vllm import LLM, SamplingParams

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
        if args.save_rollouts:
            _write_rollouts(args.save_rollouts, problems, outputs)

        for i, (p, o) in enumerate(zip(problems, outputs)):
            for c in o.outputs:
                comps.append(c.text)
                tests.append(p.tests)
                owner.append(i)
    bds = _score_with_progress(
        comps,
        tests,
        ["stdin_stdout"] * len(comps),
        args,
    )

    counts = [0] * len(problems)
    for idx, b in zip(owner, bds):
        if b.all_passed:
            counts[idx] += 1

    covered = sorted(set(owner))
    kept = []
    for i in covered:
        p = problems[i]
        c = counts[i]
        p.base_pass_count = c
        if args.keep_lo <= c <= args.keep_hi:
            kept.append(p)

    dist = Counter(counts[i] for i in covered)
    dropped = len(covered) - len(kept)
    print(f"pass-count distribution (out of {args.k}): {dict(sorted(dist.items()))}")
    print(f"kept {len(kept)}/{len(covered)} "
          f"(dropped {dropped} outside keep range [{args.keep_lo}, {args.keep_hi}]; "
          f"all-fail={dist[0]}, all-pass={dist[args.k]})")

    _maybe_mkdir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"wrote curated training pool -> {args.out}")


if __name__ == "__main__":
    main()
