"""Central project configuration.

Defaults target the first-stage experiment: start from the post-trained
Qwen3.5-2B checkpoint and run RLVR/GRPO directly on verified stdin/stdout
coding problems. SFT remains available as an optional ablation, not the default
path. Training and eval scripts can override these values via CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = "Qwen/Qwen3.5-2B"
    max_seq_len: int = 4096


@dataclass(frozen=True)
class EvalConfig:
    pass_k: int = 1
    n_samples: int = 1
    temperature: float = 0.2
    top_p: float = 0.95


@dataclass(frozen=True)
class DataConfig:
    # Python competitive-programming problems with stdin/stdout tests.
    train_dataset: str = "open-r1/verifiable-coding-problems-python_decontaminated-tested"
    judge_mode: str = "stdin_stdout"
    max_tests_per_problem: int = 10
    difficulty_probe_k: int = 4
    difficulty_keep_lo: int = 1
    max_train_problems: int = 5000
    holdout_problems: int = 500


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
