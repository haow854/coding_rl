"""Map a raw open-r1 verifiable coding row to a normalised Problem.

Known compatible datasets:
    open-r1/verifiable-coding-problems-python
    open-r1/verifiable-coding-problems-python_decontaminated-tested

The raw row schema is:
    problem_statement/problem str (prefixed with a boilerplate instruction)
    gold_standard_solution str   (```python ... ``` fenced)
    verification_info      dict  {"language": "python",
                                  "test_cases": [{"type":"stdin_stdout",
                                                  "input":..,"output":..,"fn_name":None}]}
    metadata               dict  {"difficulty": ..., ...}
    source, problem_id, in_source_id, task_type
"""
from __future__ import annotations

import re
from typing import Optional

from rlcoder.data.schema import Problem

_PREFIX = re.compile(
    r"^\s*Solve the following coding problem using the programming language python:\s*",
    re.IGNORECASE,
)


def clean_statement(s: str) -> str:
    return _PREFIX.sub("", s or "").strip()


def row_to_problem(row: dict, max_tests: Optional[int] = 15) -> Optional[Problem]:
    """Returns None if the row has no usable python stdin/stdout tests."""
    vi = row.get("verification_info") or {}
    if (vi.get("language") or "python").lower() != "python":
        return None

    tests = [
        {"input": tc.get("input", "") or "", "output": tc.get("output", "") or ""}
        for tc in (vi.get("test_cases") or [])
        if (tc.get("type") or "stdin_stdout") == "stdin_stdout"
    ]
    if not tests:
        return None
    if max_tests is not None:
        tests = tests[:max_tests]

    md = row.get("metadata") or {}
    diff = md.get("difficulty")
    statement = row.get("problem_statement") or row.get("problem") or ""
    return Problem(
        problem_id=str(row.get("problem_id") or row.get("in_source_id") or ""),
        source=row.get("source", "unknown"),
        statement=clean_statement(statement),
        tests=tests,
        mode="stdin_stdout",
        difficulty=str(diff) if diff is not None else None,
        gold_solution=row.get("gold_standard_solution"),
    )
