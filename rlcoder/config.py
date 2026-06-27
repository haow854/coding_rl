"""Central project configuration.

Defaults are intentionally conservative; training / eval scripts may override
fields via CLI or yaml later. Items marked TODO must be confirmed against the
upstream model/dataset/benchmark releases before a real run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "Qwen/Qwen3-14B"
    # Qwen3 pretraining cutoff is ~2024; LiveCodeBench must be evaluated on a
    # window strictly AFTER this to control contamination (see EvalConfig).
    train_cutoff: str = "2024-12"  # TODO: confirm exact Qwen3 data cutoff
    max_seq_len: int = 16384


@dataclass(frozen=True)
class EvalConfig:
    # Use a LiveCodeBench release window strictly after ModelConfig.train_cutoff.
    lcb_release_version: str = "release_v6"   # TODO: confirm latest version
    lcb_start_date: str = "2025-01-01"        # contamination-free window start
    lcb_end_date: str = ""                    # empty = latest available
    pass_k: int = 1
    n_samples: int = 1                        # >1 to estimate pass@k
    temperature: float = 0.2
    top_p: float = 0.95


@dataclass(frozen=True)
class DataConfig:
    # Primary RL training pool: Python competitive problems with stdin/stdout
    # tests (sources: apps / code_contests / taco), ~35.7k rows.
    train_dataset: str = "open-r1/verifiable-coding-problems-python"
    judge_mode: str = "stdin_stdout"
    max_tests_per_problem: int = 15  # cap tests used for reward to bound rollout cost
    difficulty_probe_k: int = 8      # rollouts per problem when measuring difficulty
    difficulty_keep_lo: int = 1      # keep problems with 1..k-1 passes (learnable)
    max_train_problems: int = 4000


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data: Path = ROOT / "data"
    outputs: Path = ROOT / "outputs"
    checkpoints: Path = ROOT / "outputs" / "checkpoints"
    eval_results: Path = ROOT / "outputs" / "eval"


MODEL = ModelConfig()
EVAL = EvalConfig()
DATA = DataConfig()
PATHS = Paths()
