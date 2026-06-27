# llm_rl_coding — RLVR for Code (single-policy GRPO/DAPO)

Train a single 14B policy to **reason → write code → run tests → self-debug**,
using **RLVR** (Reinforcement Learning with Verifiable Rewards): unit-test
execution is the reward signal. Evaluated on **contamination-controlled** modern
benchmarks.

## TL;DR design
- **Base model:** `Qwen/Qwen3-14B` (native thinking) — used **as-is, no
  architecture changes**; trained with **LoRA** (base weights frozen).
- **Algorithm:** GRPO, upgraded to **DAPO**-style (clip-higher, dynamic
  sampling, token-level loss, overlong shaping). Optional Dr. GRPO.
- **Single policy:** the old planner→coder→optimizer pipeline is collapsed into
  one model. "Planning" becomes the model's `<think>`; "optimizing" becomes
  multi-turn self-debug (Stage B).
- **Reward:** test pass-fraction (dense) + full-pass bonus + light format term,
  run inside a sandbox with anti-reward-hacking tripwires.
- **Data:** verified, LCB-deduped code-RL pool (DeepCoder / open-r1) + a
  difficulty filter that keeps only problems with a learnable signal.
- **Eval:** LiveCodeBench on a **post-cutoff window** (contamination control),
  BigCodeBench, and HumanEval+/MBPP+ (evalplus) as regression sanity.
- **Framework:** TRL `GRPOTrainer` to bootstrap → veRL for DAPO + multi-turn.

## Why single-policy (not multi-agent)
Multi-agent planner/coder/optimizer makes RL credit assignment hard (whose fault
when the final code fails?) and forces you to freeze some models. A single
policy emits one trajectory (`<think>` + code + debug), so GRPO assigns
advantage cleanly. This is also the post-R1 consensus for code/reasoning.

## Status / Roadmap
- **P1 data + sandbox** ✅ sandbox (stdin/stdout + assert judging, float-tolerant
  compare, resource limits), RLVR reward, dataset load + **gold-verification
  filter**. Data: `open-r1/verifiable-coding-problems-python` (~35.7k, sources
  apps/code_contests/taco, all stdin/stdout); ~66% pass gold-verification and
  form the clean pool — the rest (Python-2 gold, special-judge, float edge) are
  dropped as RLVR noise.
- **P1 training** ⏳ rollout/prompt + single-turn GRPO (TRL), proven on a small model.
- **P0 eval** ✅ code ready: in-house pass@k judge (`rlcoder/eval`, `scripts/eval_model.py`)
  on a held-out competitive set (same judge as training); `scripts/eval_evalplus.py`
  (HumanEval+/MBPP+ sanity); LiveCodeBench/BigCodeBench via their official harnesses on
  the merged model (`scripts/merge_lora.py`). Runs on AutoDL.
- **P2** 14B single-turn GRPO + DAPO on veRL.
- **P3** multi-turn agentic self-debug (loss-masked environment tokens).
- **P4** ablations + technical report + (optional) open weights.

## Layout
```
rlcoder/
  sandbox/    safe code execution (subprocess + RLIMITs; nsjail-ready)
  rewards/    RLVR reward function
  data/       (P1) dataset loading + difficulty filter + contamination dedup
  eval/       (P0) LiveCodeBench / BigCodeBench / evalplus runners
  rollout/    single-turn (P1) and multi-turn self-debug (P3)
  train/      GRPO via TRL (P1); veRL configs (P2+)
legacy/       previous throwaway prototype (kept for reference)
```

## Compute
- **Local (Windows / CPU):** develop and unit-test sandbox / reward / data.
- **AutoDL (1× A100-80G):** baseline eval + all training. Checkpoint to HF Hub.

## References
DeepCoder-14B (`agentica-org/rllm`) · veRL (`volcengine/verl`) · TRL · DAPO
(`BytedanceSeed`) · open-r1 · LiveCodeBench · BigCodeBench · evalplus.
