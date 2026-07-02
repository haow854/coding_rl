"""RLVR reward for code: reward = f(execution result).

Verifiable and execution-based. Dense (partial credit by fraction of tests
passed) so even smaller models get a gradient signal, with a bonus for fully
solving and penalties for not running / timing out. A light format term
encourages a clean ```python block (and optionally `<think>` reasoning).

Hidden tests are the real defense against reward hacking; the cheat tripwires
below only zero-out the most blatant cases.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from rlcoder.sandbox.executor import ExecResult

_CODE_FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)

# Cheap, extensible reward-hacking tripwires (not exhaustive).
_CHEAT_PATTERNS = [
    re.compile(r"\bos\.\s*_exit\b"),
    re.compile(r"\bsys\.\s*exit\b"),
    re.compile(r"\bquit\s*\("),
]


def extract_code(text: str) -> str:
    """Pull the python code out of a model completion (```python ... ``` if present)."""
    m = _CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


def has_think(text: str) -> bool:
    return bool(_THINK.search(text))


@dataclass
class RewardConfig:
    run_bonus: float = 0.1
    pass_weight: float = 1.0
    full_bonus: float = 0.5
    not_run_penalty: float = -1.0
    timeout_penalty: float = -1.0
    format_bonus: float = 0.05   # clean ```python block present
    think_bonus: float = 0.05    # non-empty <think>...</think> reasoning present
    cheat_penalty: float = -1.0


@dataclass
class RewardBreakdown:
    reward: float
    pass_rate: float
    ran: bool
    timed_out: bool
    all_passed: bool
    had_code_block: bool
    flagged_cheat: bool

    def to_metrics(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def compute_reward(
    completion: str,
    result: ExecResult,
    cfg: Optional[RewardConfig] = None,
) -> RewardBreakdown:
    cfg = cfg or RewardConfig()
    had_block = bool(_CODE_FENCE.search(completion))
    flagged = any(p.search(completion) for p in _CHEAT_PATTERNS)

    if result.timed_out:
        r = cfg.timeout_penalty
    elif not result.ran:
        r = cfg.not_run_penalty
    else:
        r = cfg.run_bonus + cfg.pass_weight * result.pass_rate
        if result.all_passed:
            r += cfg.full_bonus

    # light shaping, only when the code actually executed (don't reward noise)
    if result.ran and not result.timed_out:
        if had_block:
            r += cfg.format_bonus
        if cfg.think_bonus and has_think(completion):
            r += cfg.think_bonus

    if flagged:
        r += cfg.cheat_penalty

    return RewardBreakdown(
        reward=round(float(r), 4),
        pass_rate=result.pass_rate,
        ran=result.ran,
        timed_out=result.timed_out,
        all_passed=result.all_passed,
        had_code_block=had_block,
        flagged_cheat=flagged,
    )
