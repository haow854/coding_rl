"""Build a clean RLVR training pool: keep only problems whose gold solution
passes its own tests under our judge, and save the normalised Problems to JSONL.

    # local, on the fetched sample
    python scripts/build_dataset.py --source jsonl --in data/sample_problems.jsonl

    # on AutoDL, on the decontaminated HF dataset (streams; needs `datasets`)
    python scripts/build_dataset.py --source hf --limit 8000 --out data/clean_problems.jsonl

    # fast RL bootstrap: trust the upstream tested/decontaminated flag
    python scripts/build_dataset.py --source hf --limit 6000 --skip-verify \
        --out data/clean_problems.jsonl

    # optional CoT-trace SFT ablation data (reasoning + code, distilled from R1)
    python scripts/build_dataset.py --source cots --limit 3000 \
        --out data/cots_problems.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_cots, load_hf, load_jsonl   # noqa: E402
from rlcoder.data.verify import filter_verifiable                # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["jsonl", "hf", "cots"], default="jsonl")
    ap.add_argument("--hf-name",
                    default="open-r1/verifiable-coding-problems-python_decontaminated-tested")
    ap.add_argument("--cots-config", default="solutions_w_editorials_py_decontaminated")
    ap.add_argument("--max-completion-tokens", type=int, default=5500,
                    help="cots source only: drop rows whose reasoning+code trace "
                         "exceeds this many tokens (R1 CoT length is heavy-tailed).")
    ap.add_argument("--in", dest="inp", default="data/sample_problems.jsonl")
    ap.add_argument("--out", default="data/clean_problems.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tests", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--skip-verify", action="store_true",
                    help="Trust upstream-tested rows and only normalise to local JSONL.")
    a = ap.parse_args()

    if a.source == "jsonl":
        problems = load_jsonl(a.inp, limit=a.limit, max_tests=a.max_tests)
    elif a.source == "cots":
        problems = load_cots(config=a.cots_config, limit=a.limit, max_tests=a.max_tests,
                             max_completion_tokens=a.max_completion_tokens)
    else:
        problems = load_hf(name=a.hf_name, limit=a.limit, max_tests=a.max_tests)
    if a.skip_verify:
        kept = problems
        print(f"loaded {len(problems)} problems; skipping local gold verification")
    else:
        print(f"loaded {len(problems)} problems; verifying gold solutions "
              f"(concurrency={a.concurrency})...")
        kept, _ = await filter_verifiable(problems, concurrency=a.concurrency, timeout=a.timeout)

    total_by_src = Counter(p.source for p in problems)
    kept_by_src = Counter(p.source for p in kept)
    rate = len(kept) / len(problems) if problems else 0.0
    print(f"kept {len(kept)}/{len(problems)} ({rate:.0%})")
    print("by source:", {s: f"{kept_by_src[s]}/{total_by_src[s]}" for s in total_by_src})

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} verified problems -> {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
