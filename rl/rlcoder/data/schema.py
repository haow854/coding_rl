"""Normalised problem representation, source-agnostic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Problem:
    problem_id: str
    source: str                       # apps | code_contests | taco | ...
    statement: str                    # cleaned problem text shown to the policy
    tests: List[Dict[str, str]]       # stdin_stdout: [{"input":..., "output":...}]
    mode: str = "stdin_stdout"        # judging mode for the sandbox
    difficulty: Optional[str] = None  # raw label (heterogeneous; use empirically)
    gold_solution: Optional[str] = None
    # filled in later by the difficulty-probing step (P1)
    base_pass_count: Optional[int] = None

    @property
    def n_tests(self) -> int:
        return len(self.tests)
