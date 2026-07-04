"""Local CPU smoke test for the sandbox + reward (no GPU / no model needed).

    python scripts/sanity_check.py     # run from the repo root
"""
import asyncio
import os
import sys

# make `rlcoder` importable regardless of where we launch from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.rewards import compute_reward, extract_code  # noqa: E402
from rlcoder.sandbox import execute_async  # noqa: E402

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
CRASH = "```python\ndef add(a, b):\n    return undefined_name\n```"
LOOP = "```python\ndef add(a, b):\n    while True:\n        pass\n```"

TESTS = ["assert add(1, 2) == 3", "assert add(5, 5) == 10"]


async def main() -> None:
    cases = {"good": GOOD, "wrong": BAD, "crash": CRASH, "timeout": LOOP}
    for name, completion in cases.items():
        code = extract_code(completion)
        res = await execute_async(code, TESTS, timeout=5.0)
        rb = compute_reward(completion, res)
        print(f"[{name:8s}] ran={int(res.ran)} pass={res.passed}/{res.total} "
              f"timeout={int(res.timed_out)} reward={rb.reward:+.2f}  "
              f"feedback={res.feedback(120)!r}")


if __name__ == "__main__":
    asyncio.run(main())
