"""Gold-solution verification.

A problem is usable for RLVR only if its reference ("gold") solution actually
passes its own tests under our judge. Problems where it fails are noise — the
reward signal would be broken — so we drop them. Typical failures: Python-2 gold
code, special-judge / multiple-valid-answer problems, float edge cases.

This same machinery is reused later (P1) to probe *model* difficulty.
"""
from __future__ import annotations

import asyncio
from typing import List, Tuple

from rlcoder.data.schema import Problem
from rlcoder.rewards import extract_code
from rlcoder.sandbox import execute_async


async def gold_passes(p: Problem, timeout: float = 10.0) -> bool:
    if not p.gold_solution:
        return False
    code = extract_code(p.gold_solution)
    res = await execute_async(code, p.tests, mode=p.mode, timeout=timeout)
    return res.all_passed


async def filter_verifiable(
    problems: List[Problem],
    concurrency: int = 8,
    timeout: float = 10.0,
) -> Tuple[List[Problem], List[Tuple[Problem, bool]]]:
    """Returns (kept, all_pairs) where kept = problems whose gold solution passes."""
    from tqdm import tqdm

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(problems), desc="verify gold", unit="problem")

    async def one(p: Problem):
        async with sem:
            ok = await gold_passes(p, timeout=timeout)
        pbar.update(1)
        return p, ok

    try:
        pairs = await asyncio.gather(*[one(p) for p in problems])
    finally:
        pbar.close()

    kept = [p for p, ok in pairs if ok]
    return kept, pairs
