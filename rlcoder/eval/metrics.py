"""pass@k — the unbiased Codex estimator (pure Python, no numpy)."""
from __future__ import annotations

from typing import Dict, List


def pass_at_k(n: int, c: int, k: int) -> float:
    """Prob. that at least one of k samples is correct, given c/n are correct."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(n - c + 1, n + 1):
        p *= 1.0 - k / i
    return 1.0 - p


def aggregate_pass_at_k(
    num_samples: List[int], num_correct: List[int], ks: List[int]
) -> Dict[str, float]:
    out = {}
    for k in ks:
        vals = [pass_at_k(n, c, k) for n, c in zip(num_samples, num_correct)]
        out[f"pass@{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out
