"""Single-turn rollout scoring: completion text -> extract code -> sandbox -> reward.

`make_reward_fn` returns a TRL-compatible reward function. TRL calls it with the
batch of `completions` plus the dataset columns as keyword args (here: `tests`,
and optionally `mode`); it must return a list of floats aligned with completions.

The sandbox is async, so we score a whole batch concurrently in one event loop.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from rlcoder.rewards import RewardBreakdown, RewardConfig, compute_reward, extract_code
from rlcoder.sandbox import execute_async


def _content(completion: Any) -> str:
    """Accept a plain string or a chat message / list-of-messages completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", "")
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return str(completion)


async def _score_async(texts, tests_list, modes, timeout, concurrency, cfg) -> List[RewardBreakdown]:
    sem = asyncio.Semaphore(concurrency)

    async def one(text, tests, mode):
        code = extract_code(text)
        async with sem:
            res = await execute_async(code, tests, mode=mode, timeout=timeout)
        return compute_reward(text, res, cfg)

    return await asyncio.gather(*[one(t, ts, m) for t, ts, m in zip(texts, tests_list, modes)])


def score_batch(
    completions: List[Any],
    tests_list: List[list],
    modes: Optional[List[str]] = None,
    timeout: float = 10.0,
    concurrency: int = 64,
    cfg: Optional[RewardConfig] = None,
) -> List[RewardBreakdown]:
    texts = [_content(c) for c in completions]
    if modes is None:
        modes = ["stdin_stdout"] * len(texts)
    return asyncio.run(_score_async(texts, tests_list, modes, timeout, concurrency, cfg))


def make_reward_fn(
    cfg: Optional[RewardConfig] = None,
    timeout: float = 10.0,
    concurrency: int = 64,
) -> Callable[..., List[float]]:
    """Build a TRL `reward_funcs` callable. Expects a `tests` dataset column
    (per-sample list of {"input","output"}); `mode` column is optional."""

    def reward_fn(completions=None, **kwargs) -> List[float]:
        if completions is None:                  # robust across TRL call conventions
            completions = kwargs["completions"]
        tests_list = kwargs["tests"]
        modes = kwargs.get("mode")
        if isinstance(modes, str):
            modes = [modes] * len(completions)
        bds = score_batch(completions, tests_list, modes, timeout=timeout,
                           concurrency=concurrency, cfg=cfg)
        return [b.reward for b in bds]

    reward_fn.__name__ = "code_rlvr_reward"
    return reward_fn
