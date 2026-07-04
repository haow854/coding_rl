"""Map a nvidia/OpenCodeReasoning row to a normalised Problem.

OpenCodeReasoning (OCR) is R1-distilled competitive-code reasoning, ~735k
solutions over ~28k unique problems. We use it for Stage-1 distillation SFT, so
gold_solution carries the FULL assistant turn: a <think>...</think> reasoning
trace followed by a ```python fenced program (same contract as parse_cots.py;
extract_code() pulls the final block back out for judging).

OCR split_0 fields we rely on:
    id        unique row id
    input     problem statement (split_0 only; split_1 references external sets)
    output    R1's full response (reasoning + code)
    solution  code-only portion of R1's response
    source    platform (codeforces / codechef / ...)
    difficulty
    dataset   origin corpus (apps / taco / code_contests)

OCR ships NO executable tests, so tests=[] here — that is fine for SFT (the
trainer only needs gold_solution). Do not feed OCR Problems to the GRPO reward
path expecting a pass/fail signal; use the verifiable-coding pool for that.
"""
from __future__ import annotations

import re
from typing import Optional

from rlcoder.data.schema import Problem

_CODE_FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def _reasoning_from_output(output: str) -> str:
    """Best-effort extraction of the reasoning text (no code, no think tags)."""
    if "</think>" in output:
        head = output.split("</think>", 1)[0]
        return head.replace("<think>", "").strip()
    # No explicit tags: treat everything before the first code fence as reasoning.
    m = _CODE_FENCE.search(output)
    return (output[: m.start()] if m else output).strip()


def _final_code(output: str, solution: Optional[str]) -> Optional[str]:
    """Prefer the clean code-only `solution` field; fall back to the last fence."""
    if solution and solution.strip():
        return solution.strip()
    blocks = _CODE_FENCE.findall(output or "")
    return blocks[-1].strip() if blocks else None


def ocr_row_to_problem(row: dict) -> Optional[Problem]:
    """Returns None if the row lacks a problem statement or any usable code.

    Rebuilds a canonical `<think>reasoning</think>\\n\\n```python ...``` trace so
    the target always matches our inference format regardless of how the raw
    `output` was punctuated.
    """
    statement = (row.get("input") or "").strip()
    if not statement:
        return None

    output = row.get("output") or ""
    code = _final_code(output, row.get("solution"))
    if not code:
        return None

    reasoning = _reasoning_from_output(output)
    if reasoning:
        gold = f"<think>\n{reasoning}\n</think>\n\n```python\n{code}\n```"
    else:
        gold = f"```python\n{code}\n```"

    difficulty = row.get("difficulty")
    return Problem(
        problem_id=str(row.get("id") or ""),
        source=str(row.get("source") or row.get("dataset") or "open_code_reasoning"),
        statement=statement,
        tests=[],  # OCR ships no executable tests; SFT does not need them
        mode="stdin_stdout",
        difficulty=str(difficulty) if difficulty is not None else None,
        gold_solution=gold,
    )


def problem_group_key(problem: Problem) -> str:
    """Stable per-problem key for de-duplication (OCR ids are per-solution)."""
    return re.sub(r"\s+", " ", problem.statement).strip().lower()[:512]
