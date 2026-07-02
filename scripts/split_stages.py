"""Stratified split for the stage-1 SFT -> GRPO workflow.

The raw HF viewer is often grouped by source/problem id, so avoid head/tail
splits. This script shuffles within source/difficulty buckets and writes three
disjoint files:

    python scripts/split_stages.py --in data/clean_problems.jsonl \
        --sft-out data/sft_train.jsonl --rl-out data/rl_pool.jsonl \
        --dev-out data/dev_internal.jsonl --dev 1000 --sft 10000
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


Row = Tuple[str, dict]


def _read_rows(path: str) -> List[Row]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append((line, json.loads(line)))
    return rows


def _bucket_key(obj: dict) -> tuple:
    return (
        obj.get("source") or "unknown",
        obj.get("difficulty") or "unknown",
    )


def _alloc(bucket_sizes: Dict[tuple, int], target: int) -> Dict[tuple, int]:
    total = sum(bucket_sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"target {target} is outside 0..{total}")

    exact = {k: bucket_sizes[k] * target / total for k in bucket_sizes}
    out = {k: int(v) for k, v in exact.items()}
    remaining = target - sum(out.values())
    order = sorted(bucket_sizes, key=lambda k: (exact[k] - out[k], bucket_sizes[k]), reverse=True)
    for k in order[:remaining]:
        out[k] += 1
    return out


def _write(path: str, rows: List[Row]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line, _ in rows:
            f.write(line + "\n")


def _counts(rows: List[Row]) -> dict:
    return {
        "n": len(rows),
        "source": dict(Counter(obj.get("source", "unknown") for _, obj in rows)),
        "difficulty": dict(Counter(obj.get("difficulty", "unknown") for _, obj in rows)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean_problems.jsonl")
    ap.add_argument("--sft-out", default="data/sft_train.jsonl")
    ap.add_argument("--rl-out", default="data/rl_pool.jsonl")
    ap.add_argument("--dev-out", default="data/dev_internal.jsonl")
    ap.add_argument("--dev", type=int, default=1000)
    ap.add_argument("--sft", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    buckets: Dict[tuple, List[Row]] = defaultdict(list)
    for row in _read_rows(args.inp):
        buckets[_bucket_key(row[1])].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    bucket_sizes = {k: len(v) for k, v in buckets.items()}
    total = sum(bucket_sizes.values())
    if args.dev + args.sft > total:
        raise ValueError(f"dev+sft={args.dev + args.sft} exceeds total rows={total}")

    dev_alloc = _alloc(bucket_sizes, args.dev)
    remaining_sizes = {k: bucket_sizes[k] - dev_alloc[k] for k in bucket_sizes}
    sft_alloc = _alloc(remaining_sizes, args.sft)

    dev_rows: List[Row] = []
    sft_rows: List[Row] = []
    rl_rows: List[Row] = []
    for k, rows in buckets.items():
        d = dev_alloc[k]
        s = sft_alloc[k]
        dev_rows.extend(rows[:d])
        sft_rows.extend(rows[d:d + s])
        rl_rows.extend(rows[d + s:])

    rng.shuffle(dev_rows)
    rng.shuffle(sft_rows)
    rng.shuffle(rl_rows)

    _write(args.dev_out, dev_rows)
    _write(args.sft_out, sft_rows)
    _write(args.rl_out, rl_rows)

    print("wrote splits:")
    for name, path, rows in [
        ("dev", args.dev_out, dev_rows),
        ("sft", args.sft_out, sft_rows),
        ("rl", args.rl_out, rl_rows),
    ]:
        print(f"  {name:3s} {len(rows):5d} -> {path}")
        print(f"      source={_counts(rows)['source']}")


if __name__ == "__main__":
    main()
