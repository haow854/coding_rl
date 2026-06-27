"""Prompt construction for the single policy.

The policy is asked to reason inside <think>...</think> and then emit ONE Python 3
program (in a ```python block) that reads stdin and writes stdout — matching how
the dataset is judged (competitive stdin/stdout). Kept dependency-free; the chat
template is applied later by the trainer/eval (which own the tokenizer).
"""
from __future__ import annotations

from typing import Dict, List

from rlcoder.data.schema import Problem

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. First reason about the problem "
    "inside <think> </think>: identify the algorithm, edge cases, and complexity. "
    "Then output your final solution as a single Python 3 program inside one "
    "```python code block. The program must read input from standard input (stdin) "
    "and write the answer to standard output (stdout), exactly matching the "
    "output format described in the problem."
)


def build_messages(problem: Problem) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.statement},
    ]
