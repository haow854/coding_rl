"""Split a normalised problem JSONL into train / held-out eval (disjoint).

    python scripts/split_data.py --in data/clean_problems.jsonl \
        --train-out data/train_problems.jsonl --holdout-out data/holdout.jsonl --holdout 200
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean_problems.jsonl")
    ap.add_argument("--train-out", default="data/train_problems.jsonl")
    ap.add_argument("--holdout-out", default="data/holdout.jsonl")
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lines = [ln for ln in open(args.inp, encoding="utf-8") if ln.strip()]
    random.Random(args.seed).shuffle(lines)
    holdout, train = lines[: args.holdout], lines[args.holdout:]

    for path, rows in [(args.train_out, train), (args.holdout_out, holdout)]:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(rows)
    print(f"train={len(train)} -> {args.train_out}  |  holdout={len(holdout)} -> {args.holdout_out}")


if __name__ == "__main__":
    main()
