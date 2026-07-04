"""Build the Stage-1 distillation SFT pool from nvidia/OpenCodeReasoning.

OCR has ~735k R1 traces over ~28k unique competitive-programming problems. Raw,
it is too big and its long traces meander ("wait, let me reconsider..."), which
a small student imitates badly. This script curates a compact, coverage-first
subset:

    split_0 (self-contained)                       ~585k rows
      -> keep the SHORTEST k trace(s) per problem  ~28k problems  (coverage > count)
      -> drop the longest --drop-longest-frac      (trim the meandering tail)
      -> difficulty-stratified subsample           -> --target-size (default 30k)

Output is serialised Problem JSONL (gold_solution = <think>trace</think> + code),
directly consumable by rlcoder/train/sft_trl.py via load_clean_jsonl.

    # on the GPU box (needs `datasets`)
    python scripts/build_sft_data.py --out data/sft_ocr.jsonl --target-size 30000

    # smoke test on a small slice
    python scripts/build_sft_data.py --limit 20000 --target-size 2000 \
        --out data/sft_ocr_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.parse_ocr import ocr_row_to_problem, problem_group_key  # noqa: E402


def _load_stream(dataset: str, config: str | None, split: str):
    from datasets import load_dataset  # heavy dep; GPU box only

    try:
        if config:
            return load_dataset(dataset, config, split=split, streaming=True)
        return load_dataset(dataset, split=split, streaming=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not load {dataset} (config={config}, split={split}): {e}\n"
            "Check the dataset card for the exact config/split names, then pass "
            "--config / --split."
        )


def _percentiles(values, ps=(50, 75, 90, 95, 99)):
    if not values:
        return {}
    s = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
        out[f"p{p}"] = s[idx]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nvidia/OpenCodeReasoning")
    ap.add_argument("--config", default=None,
                    help="Dataset config name, if the card requires one.")
    ap.add_argument("--split", default="split_0",
                    help="split_0 is self-contained (carries the problem text).")
    ap.add_argument("--out", default="data/sft_ocr.jsonl")
    ap.add_argument("--solutions-per-problem", type=int, default=1,
                    help="Keep the k shortest traces per unique problem.")
    ap.add_argument("--drop-longest-frac", type=float, default=0.15,
                    help="Drop this fraction of the longest surviving traces.")
    ap.add_argument("--max-trace-chars", type=int, default=48000,
                    help="Hard cap (~15k tokens) applied while streaming.")
    ap.add_argument("--target-size", type=int, default=30000,
                    help="Final sample count after difficulty-stratified subsampling.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after scanning this many raw rows (debug/smoke).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = _load_stream(args.dataset, args.config, args.split)

    # Stream rows, keeping only the shortest-k trace(s) per unique problem.
    best: dict[str, list[tuple[int, object]]] = defaultdict(list)
    scanned = mapped = had_think = too_long = 0
    k = max(1, args.solutions_per_problem)
    for row in ds:
        scanned += 1
        if "<think>" in (row.get("output") or ""):
            had_think += 1
        p = ocr_row_to_problem(row)
        if p is None:
            continue
        n = len(p.gold_solution or "")
        if n > args.max_trace_chars:
            too_long += 1
            continue
        mapped += 1
        key = problem_group_key(p)
        bucket = best[key]
        if len(bucket) < k:
            bucket.append((n, p))
            bucket.sort(key=lambda t: t[0])
        elif n < bucket[-1][0]:
            bucket[-1] = (n, p)
            bucket.sort(key=lambda t: t[0])
        if args.limit is not None and scanned >= args.limit:
            break

    candidates = [(n, p) for bucket in best.values() for (n, p) in bucket]
    print(f"scanned {scanned} rows; mapped {mapped}; "
          f"{had_think}/{scanned} raw outputs had <think> tags; "
          f"dropped {too_long} over {args.max_trace_chars} chars")
    print(f"unique problems: {len(best)}; per-problem candidates: {len(candidates)}")
    if not candidates:
        raise SystemExit("no candidates survived; check dataset/config/split names")

    # Trim the longest tail (the meandering traces).
    candidates.sort(key=lambda t: t[0])
    keep_n = int(round(len(candidates) * (1.0 - args.drop_longest_frac)))
    trimmed = candidates[:keep_n]
    print(f"after dropping longest {args.drop_longest_frac:.0%}: {len(trimmed)} "
          f"(trace-char {_percentiles([n for n, _ in trimmed])})")

    problems = [p for _, p in trimmed]

    # Difficulty-stratified subsample down to the target size.
    if len(problems) > args.target_size:
        buckets: dict[str, list] = defaultdict(list)
        for p in problems:
            buckets[p.difficulty or "unknown"].append(p)
        for b in buckets.values():
            rng.shuffle(b)
        total = len(problems)
        selected: list = []
        for label, b in buckets.items():
            quota = int(round(args.target_size * len(b) / total))
            selected.extend(b[:quota])
        rng.shuffle(selected)
        # Fix rounding drift against the target.
        if len(selected) > args.target_size:
            selected = selected[: args.target_size]
        elif len(selected) < args.target_size:
            chosen = {id(p) for p in selected}
            leftover = [p for p in problems if id(p) not in chosen]
            rng.shuffle(leftover)
            selected.extend(leftover[: args.target_size - len(selected)])
        problems = selected

    rng.shuffle(problems)
    print(f"final: {len(problems)} samples")
    print("difficulty:", dict(sorted(Counter(p.difficulty or "unknown"
                                              for p in problems).items())))
    print("source:", dict(sorted(Counter(p.source for p in problems).items())))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"wrote {len(problems)} SFT examples -> {args.out}")


if __name__ == "__main__":
    main()
