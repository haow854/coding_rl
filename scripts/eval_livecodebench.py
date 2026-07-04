"""LiveCodeBench (code generation) eval through our own sandbox judge.

Loads livecodebench/code_generation_lite, keeps the STDIN/STDOUT problems
(AtCoder/Codeforces — the same distribution we train on) and scores them with
the SAME sandbox judge used for training/eval everywhere else, so base/SFT/GRPO
numbers are directly comparable. Functional (LeetCode) problems need a call-based
judge and are counted + skipped; for a leaderboard-comparable number over ALL
problems, run the official LiveCodeBench harness on the generated outputs.

Use the date window to control contamination (keep problems released after the
base model's training cutoff).

    python scripts/eval_livecodebench.py --model Qwen/Qwen3-4B \
        --version-tag release_v5 --start-date 2025-01-01 \
        --out outputs/eval/lcb_base.json

    python scripts/eval_livecodebench.py --model Qwen/Qwen3-4B \
        --lora outputs/qwen3_4b_sft --version-tag release_v5 --start-date 2025-01-01 \
        --out outputs/eval/lcb_sft.json
"""
import argparse
import base64
import json
import os
import pickle
import sys
import zlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.schema import Problem              # noqa: E402
from rlcoder.eval.metrics import aggregate_pass_at_k  # noqa: E402
from rlcoder.eval.run_eval import evaluate            # noqa: E402


def _load_lcb(version_tag: str, split: str):
    """LCB's layout has shifted across releases; try the known call shapes."""
    from datasets import load_dataset

    errors = []
    attempts = (
        lambda: load_dataset("livecodebench/code_generation_lite",
                             version_tag=version_tag, split=split, trust_remote_code=True),
        lambda: load_dataset("livecodebench/code_generation_lite",
                             version_tag, split=split, trust_remote_code=True),
        lambda: load_dataset("livecodebench/code_generation_lite",
                             version_tag, split=split),
    )
    for attempt in attempts:
        try:
            return attempt()
        except Exception as e:  # noqa: BLE001
            errors.append(str(e).splitlines()[-1] if str(e) else repr(e))
    raise SystemExit("could not load LiveCodeBench code_generation_lite "
                     f"(version_tag={version_tag}, split={split}):\n  " +
                     "\n  ".join(errors) +
                     "\nTip: keep datasets<4.0, or `pip install livecodebench`.")


def _decode_tests(raw):
    """LCB test cases: public = plain JSON string; private = base64+zlib+pickle."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
    except Exception:  # noqa: BLE001
        return []


def _lcb_to_problem(row: dict, max_tests: int):
    """Build a stdin/stdout Problem, or None for functional (call-based) rows."""
    tests_raw = list(_decode_tests(row.get("public_test_cases"))) + \
        list(_decode_tests(row.get("private_test_cases")))
    if not tests_raw:
        return None
    # Only stdin problems: our sandbox judges by stdin->stdout, not by calling a fn.
    if any((t.get("testtype") or "stdin") != "stdin" for t in tests_raw):
        return None
    tests = [{"input": t.get("input", "") or "", "output": t.get("output", "") or ""}
             for t in tests_raw]
    if max_tests:
        tests = tests[:max_tests]
    diff = row.get("difficulty")
    return Problem(
        problem_id=str(row.get("question_id") or ""),
        source=str(row.get("platform") or "livecodebench"),
        statement=row.get("question_content") or "",
        tests=tests,
        mode="stdin_stdout",
        difficulty=str(diff) if diff is not None else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--version-tag", default="release_v5",
                    help="release_v1..v6; pick a window newer than the base's cutoff.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--start-date", default=None,
                    help="Keep contest_date >= YYYY-MM-DD (contamination control).")
    ap.add_argument("--end-date", default=None, help="Keep contest_date <= YYYY-MM-DD.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--ks", default="1,5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = _load_lcb(args.version_tag, args.split)

    problems, skipped_functional = [], 0
    for row in ds:
        cd = str(row.get("contest_date") or "")[:10]
        if args.start_date and cd and cd < args.start_date:
            continue
        if args.end_date and cd and cd > args.end_date:
            continue
        p = _lcb_to_problem(row, args.max_tests)
        if p is None:
            skipped_functional += 1
            continue
        problems.append(p)
        if args.limit and len(problems) >= args.limit:
            break

    print(f"LCB {args.version_tag}: {len(problems)} stdin problems judged "
          f"(skipped {skipped_functional} functional/other; those need the "
          f"official harness)")
    if not problems:
        raise SystemExit("no stdin problems after filtering; widen the date window "
                         "or --version-tag")

    ks = [int(x) for x in args.ks.split(",")]
    res, per = evaluate(args.model, problems, n=args.n, temperature=args.temperature,
                        max_tokens=args.max_tokens, lora_path=args.lora, ks=ks)

    # pass@1 broken down by LCB difficulty label.
    diff_total, diff_correct = defaultdict(list), defaultdict(list)
    for prob, tot, cor in per:
        d = prob.difficulty or "unknown"
        diff_total[d].append(tot)
        diff_correct[d].append(cor)
    res["pass@1_by_difficulty"] = {
        d: round(aggregate_pass_at_k(diff_total[d], diff_correct[d], [1])["pass@1"], 4)
        for d in sorted(diff_total)
    }
    res["skipped_functional"] = skipped_functional
    res["version_tag"] = args.version_tag
    res["judged"] = "stdin_subset"

    print(json.dumps(res, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "lora": args.lora, **res}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
