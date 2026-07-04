"""Load problems from a local JSONL sample (CPU dev, stdlib only) or from the
HuggingFace dataset (needs `datasets`; use on the GPU box)."""
from __future__ import annotations

import json
from typing import List, Optional

from rlcoder.data.parse import row_to_problem
from rlcoder.data.schema import Problem


def load_jsonl(path: str, limit: Optional[int] = None, max_tests: Optional[int] = 15) -> List[Problem]:
    out: List[Problem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = row_to_problem(json.loads(line), max_tests=max_tests)
            if p is not None:
                out.append(p)
            if limit is not None and len(out) >= limit:
                break
    return out


def load_clean_jsonl(path: str, limit: Optional[int] = None) -> List[Problem]:
    """Load a JSONL of already-normalised + verified Problems (output of
    scripts/build_dataset.py), i.e. serialised Problem dicts (not raw rows)."""
    out: List[Problem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Problem(**json.loads(line)))
            if limit is not None and len(out) >= limit:
                break
    return out


def load_hf(
    name: str = "open-r1/verifiable-coding-problems-python_decontaminated-tested",
    split: str = "train",
    limit: Optional[int] = None,
    max_tests: Optional[int] = 15,
) -> List[Problem]:
    from datasets import load_dataset  # heavy dep; only available on the GPU box

    ds = load_dataset(name, split=split, streaming=limit is not None)
    out: List[Problem] = []
    for row in ds:
        p = row_to_problem(row, max_tests=max_tests)
        if p is not None:
            out.append(p)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_cots(
    config: str = "solutions_w_editorials_py_decontaminated",
    split: str = "train",
    limit: Optional[int] = None,
    max_tests: Optional[int] = 15,
    max_completion_tokens: Optional[int] = 5500,
) -> List[Problem]:
    """Load open-r1/codeforces-cots: gold_solution carries a full <think>+code
    reasoning trace (distilled from R1), not a bare reference solution.
    max_completion_tokens drops rows whose trace is too long to train on --
    R1 CoT length is heavily right-skewed (hard problems can run 15k+ tokens),
    see rlcoder/data/parse_cots.py."""
    from datasets import load_dataset  # heavy dep; only available on the GPU box

    from rlcoder.data.parse_cots import cots_row_to_problem

    ds = load_dataset("open-r1/codeforces-cots", config, split=split,
                      streaming=limit is not None)
    out: List[Problem] = []
    seen = too_long = 0
    for row in ds:
        seen += 1
        p = cots_row_to_problem(row, max_tests=max_tests,
                                max_completion_tokens=max_completion_tokens)
        if p is not None:
            out.append(p)
        elif max_completion_tokens is not None:
            # would this row have been kept if not for the length cap?
            if cots_row_to_problem(row, max_tests=max_tests,
                                   max_completion_tokens=None) is not None:
                too_long += 1
        if limit is not None and len(out) >= limit:
            break
    if too_long:
        print(f"[load_cots] kept {len(out)}/{seen}; dropped {too_long} for "
              f"exceeding max_completion_tokens={max_completion_tokens}")
    return out


def load_problems(source: str = "jsonl", **kwargs) -> List[Problem]:
    """source="jsonl" (kwargs: path, limit, max_tests), "hf" (kwargs: name, split, ...),
    or "cots" (kwargs: config, split, limit, max_tests)."""
    if source == "jsonl":
        return load_jsonl(**kwargs)
    if source == "hf":
        return load_hf(**kwargs)
    if source == "cots":
        return load_cots(**kwargs)
    raise ValueError(f"unknown source: {source}")
