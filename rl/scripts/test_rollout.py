"""Local CPU test of the TRL-facing reward function (no GPU / no model).

Feeds a problem's gold solution as a fake "completion" (should score high) and a
broken variant (should score low), exercising the exact reward_fn that the GRPO
trainer will call.

    python scripts/test_rollout.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_clean_jsonl   # noqa: E402
from rlcoder.rollout import make_reward_fn        # noqa: E402

problems = load_clean_jsonl("data/clean_problems.jsonl", limit=3)
reward_fn = make_reward_fn(timeout=10.0)

completions, tests, labels = [], [], []
for p in problems:
    gold = p.gold_solution or ""
    broken = gold.replace("```python", "```python\nimport sys; sys.exit(1)\n", 1)  # runs nothing useful
    completions += [gold, broken]
    tests += [p.tests, p.tests]
    labels += [f"{p.source}/{p.problem_id}:gold", f"{p.source}/{p.problem_id}:broken"]

rewards = reward_fn(completions=completions, tests=tests)
for lab, r in zip(labels, rewards):
    print(f"{r:+.2f}  {lab}")
