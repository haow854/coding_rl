"""Validate the data + sandbox pipeline locally (CPU, no model needed):
run each gold solution against its own tests and report the pass rate.

A high full-pass rate means our stdin/stdout judging is correct. The remainder
are expected: special-judge problems (multiple valid outputs), float precision,
or imperfect gold solutions in the dataset.

    python scripts/validate_gold.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_jsonl       # noqa: E402
from rlcoder.rewards import extract_code        # noqa: E402
from rlcoder.sandbox import execute_async       # noqa: E402

SAMPLE = os.path.join("data", "sample_problems.jsonl")


async def main() -> None:
    problems = load_jsonl(SAMPLE, max_tests=8)
    print(f"loaded {len(problems)} problems from {SAMPLE}")

    sem = asyncio.Semaphore(8)

    async def check(p):
        code = extract_code(p.gold_solution or "")
        async with sem:
            res = await execute_async(code, p.tests, mode="stdin_stdout", timeout=10.0)
        return p, res

    results = await asyncio.gather(*[check(p) for p in problems])

    n = len(results)
    ran = sum(1 for _, r in results if r.ran and not r.timed_out)
    full = sum(1 for _, r in results if r.all_passed)
    timeouts = sum(1 for _, r in results if r.timed_out)
    avg = sum(r.pass_rate for _, r in results) / n if n else 0.0
    print(f"gold solutions: ran={ran}/{n}  full-pass={full}/{n} ({full/n:.0%})  "
          f"timeouts={timeouts}  avg pass_rate={avg:.1%}")

    by_src = {}
    for p, r in results:
        s = by_src.setdefault(p.source, [0, 0])
        s[0] += 1
        s[1] += int(r.all_passed)
    print("by source (full-pass):", {k: f"{v[1]}/{v[0]}" for k, v in by_src.items()})

    shown = 0
    for p, r in results:
        if not r.all_passed and shown < 4:
            print(f"\n[{p.source}/{p.problem_id}] ran={int(r.ran)} "
                  f"pass={r.passed}/{r.total} timeout={int(r.timed_out)}")
            print("   " + r.feedback(220).replace("\n", "\n   "))
            shown += 1


if __name__ == "__main__":
    asyncio.run(main())
