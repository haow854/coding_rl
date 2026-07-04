"""Evaluate a model (base or base+LoRA) on a normalised problem JSONL using our
own sandbox judge — the *same* judge used for the training reward, so base-vs-RL
numbers are directly comparable. Generates n samples/problem -> pass@1 / pass@k.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rlcoder.data.schema import Problem
from rlcoder.eval.generate import generate
from rlcoder.eval.metrics import aggregate_pass_at_k
from rlcoder.prompting import build_messages
from rlcoder.rollout.single_turn import score_batch


def evaluate(
    model: str,
    problems: List[Problem],
    n: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    lora_path: Optional[str] = None,
    reward_timeout: float = 10.0,
    ks: Sequence[int] = (1,),
) -> Tuple[dict, List[Tuple[Problem, int, int]]]:
    gens = generate(
        model,
        [build_messages(p) for p in problems],
        n=n, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, lora_path=lora_path,
    )

    comps, tests, owner = [], [], []
    for i, (p, samples) in enumerate(zip(problems, gens)):
        for text in samples:
            comps.append(text)
            tests.append(p.tests)
            owner.append(i)
    bds = score_batch(comps, tests, ["stdin_stdout"] * len(comps),
                      timeout=reward_timeout, concurrency=128)

    total = [0] * len(problems)
    correct = [0] * len(problems)
    for idx, b in zip(owner, bds):
        total[idx] += 1
        if b.all_passed:
            correct[idx] += 1

    res = aggregate_pass_at_k(total, correct, list(ks))
    res["n_problems"] = len(problems)
    res["n_samples"] = n
    return res, list(zip(problems, total, correct))
