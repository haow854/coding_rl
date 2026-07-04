"""Map an open-r1/codeforces-cots row to a normalised Problem.

Known compatible dataset/config:
    open-r1/codeforces-cots, config="solutions_w_editorials_py_decontaminated"
    (R1 is prompted with the human-written editorial before it reasons, which
    empirically gives much cleaner traces than the plain "solutions_py_*"
    config -- that one lets R1 search blind, and on hard problems a real
    fraction of its traces devolve into unproductive "wait, let me
    reconsider..." loops that end in a guess rather than a derivation. Also
    compatible with any other config sharing the same row schema, e.g.
    "solutions_py_decontaminated".)

Unlike parse.py's row_to_problem (PrimeIntellect schema, gold_solution = bare
code), gold_solution here is the FULL assistant turn: a <think>...</think>
reasoning trace followed by a ```python fenced solution. SFT imitates the
reasoning, not just the answer; extract_code() already pulls only the fenced
block back out for judging, so verify.py and the sandbox need no changes.
"""
from __future__ import annotations

from typing import Optional

from rlcoder.data.schema import Problem


def _assemble_statement(row: dict) -> str:
    parts = [row.get("description") or ""]
    if row.get("input_format"):
        parts.append("Input Format:\n" + row["input_format"])
    if row.get("output_format"):
        parts.append("Output Format:\n" + row["output_format"])
    if row.get("note"):
        parts.append("Note:\n" + row["note"])
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _assistant_content(row: dict) -> Optional[str]:
    for m in row.get("messages") or []:
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return None


def _completion_tokens(row: dict, gold: str) -> float:
    """Prefer the token count recorded at generation time; fall back to a
    chars-per-token estimate if api_metadata is missing."""
    n = (row.get("api_metadata") or {}).get("completion_tokens")
    return float(n) if n is not None else len(gold) / 3.2


def cots_row_to_problem(
    row: dict,
    max_tests: Optional[int] = 15,
    max_completion_tokens: Optional[int] = 5500,
) -> Optional[Problem]:
    """Returns None if the row has no public tests, no assistant CoT turn, or
    (when max_completion_tokens is set) the reasoning trace is too long for the
    training sequence budget. R1-distilled CoT length is heavily right-skewed —
    most problems land under ~1500 tokens but hard ones run past 15k — so this
    is a real filter, not a formality; without it, long rows get silently
    truncated mid-reasoning (before the code block) or dropped by the trainer,
    depending on TRL version."""
    pt = row.get("public_tests") or {}
    inputs = pt.get("input") or []
    outputs = pt.get("output") or []
    tests = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
    if not tests:
        return None
    if max_tests is not None:
        tests = tests[:max_tests]

    gold = _assistant_content(row)
    if not gold:
        return None

    if max_completion_tokens is not None and _completion_tokens(row, gold) > max_completion_tokens:
        return None

    return Problem(
        problem_id=str(row.get("id") or ""),
        source="codeforces_cots",
        statement=_assemble_statement(row),
        tests=tests,
        mode="stdin_stdout",
        difficulty=None,
        gold_solution=gold,
    )
