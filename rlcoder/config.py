"""Central project configuration.

Defaults target the SFT-then-GRPO route on a text-native reasoning base
(Qwen3-4B, thinking mode). Stage 1 is distillation SFT on competitive-code
reasoning traces (NVIDIA OpenCodeReasoning); Stage 2 is optional GRPO/RLVR on
verified stdin/stdout problems. Training and eval scripts can override these
values via CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    # Text-native reasoning model. The old Qwen3.5-2B route was multimodal +
    # non-thinking, which capped competitive-coding ability (see README).
    base_model: str = "Qwen/Qwen3-4B"
    max_seq_len: int = 16384          # room for a <think> trace + code (was 4096)
    enable_thinking: bool = True


@dataclass(frozen=True)
class SFTDataConfig:
    """Stage-1 distillation SFT data — built by scripts/build_sft_data.py."""
    dataset: str = "nvidia/OpenCodeReasoning"
    dataset_config: str = "split_0"   # OCR exposes split_0/split_1 as configs, not splits
    dataset_split: str = "split_0"    # the split inside that config is also named split_0
    solutions_per_problem: int = 1    # coverage first: keep the shortest trace(s)/problem
    drop_longest_frac: float = 0.15   # trim the meandering long tail
    max_trace_chars: int = 48000      # hard cap (~15k tokens) applied before the frac trim
    target_size: int = 30000


@dataclass(frozen=True)
class EvalConfig:
    # Multi-sample by default: a single greedy pass@1 is too noisy to detect the
    # small deltas SFT/RL produce on a small model. Compare against baseline with
    # identical sampling + thinking settings.
    pass_k: int = 1
    n_samples: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 4096            # matches GRPO max-completion; consistent gen budget
    ks: Tuple[int, ...] = (1, 5)


@dataclass(frozen=True)
class DataConfig:
    # Python competitive-programming problems with stdin/stdout tests (GRPO pool).
    train_dataset: str = "open-r1/verifiable-coding-problems-python_decontaminated-tested"
    judge_mode: str = "stdin_stdout"
    max_tests_per_problem: int = 10
    difficulty_probe_k: int = 8       # samples/problem when probing difficulty
    difficulty_keep_lo: int = 1       # keep problems solved in [lo, hi] of k probes
    difficulty_keep_hi: int = 7
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
SFT_DATA = SFTDataConfig()
EVAL = EvalConfig()
DATA = DataConfig()
PATHS = Paths()
