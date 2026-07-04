# RunPod quickstart

Local dev is CPU-only. Everything below runs on the RunPod GPU box.

The flow is two stages: **Stage 1 = SFT distillation** (the main lift), then
**Stage 2 = GRPO** continuing the SFT adapter. Keep thinking mode, completion
length, and eval length consistent across all steps.

## 0. Model and GPU

```text
Qwen/Qwen3-4B      # text-native, thinking mode on
```

SFT trains on long reasoning traces (`--max-length 16384`), and GRPO/eval
generate long thinking completions (`--max-completion 4096`). A 40G card is a
comfortable minimum; 80G is convenient for full-parameter runs or
`--num-generations 8`. LoRA + gradient checkpointing fits a 24-32G card if you
keep `--per-device-batch` small.

## 1. Environment

Use a RunPod PyTorch image with CUDA/PyTorch already installed, then install the project
deps. Do not reinstall torch unless the image is broken.

```bash
pip install -U -r requirements-gpu.txt
```

Install vLLM only if it matches the image's torch/CUDA stack:

```bash
pip install "vllm>=0.10"
```

Install the official LiveCodeBench harness when you want a leaderboard-style
external score:

```bash
cd /workspace
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
python -m pip install -e LiveCodeBench
cd /workspace/coding_rl
```

If vLLM tries to downgrade torch or compile forever in a broken image, fix the
environment before launching long jobs.

On RTX 5090 / Blackwell, vLLM 0.24 may mis-detect FlashInfer sampler support.
This repo disables the FlashInfer sampler in eval/filter scripts by default. If
you run vLLM manually, use:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## 2. Stage 1 — SFT distillation (main)

Curate a compact, coverage-first SFT pool from OpenCodeReasoning (dedup to the
shortest trace per problem, trim the meandering long tail, difficulty-stratify):

```bash
python scripts/build_sft_data.py \
  --split split_0 --target-size 30000 \
  --out data/sft_ocr.jsonl
```

Sanity-check the printed distribution first with a small slice
(`--limit 20000 --target-size 2000`). Then SFT (LoRA):

```bash
python rlcoder/train/sft_trl.py --model Qwen/Qwen3-4B \
  --data data/sft_ocr.jsonl \
  --per-device-batch 1 --grad-accum 16 --max-length 16384 \
  --lr 1e-4 --epochs 2 --lora-r 32 \
  --bf16 --gradient-checkpointing --packing \
  --output outputs/qwen3_4b_sft
```

Full-parameter instead (needs ~80G; lower the LR): add `--full-ft --lr 1e-5`.

## 3. Stage-1 eval (base vs SFT)

Multi-sample so small deltas are visible; keep sampling identical for both:

```bash
python scripts/eval_model.py --model Qwen/Qwen3-4B \
  --data data/dev_internal.jsonl --limit 500 \
  --out outputs/eval/qwen3_4b_base.json

python scripts/eval_model.py --model Qwen/Qwen3-4B --lora outputs/qwen3_4b_sft \
  --data data/dev_internal.jsonl --limit 500 \
  --out outputs/eval/qwen3_4b_sft.json
```

This is the number that should move most. `eval_model.py` defaults to
`--n 8 --temperature 0.8 --max-tokens 8192 --ks 1,5`.

## 4. Stage 2 — GRPO pool + difficulty filter (optional)

Build and split the verifiable stdin/stdout pool (this one *does* have tests):

```bash
python scripts/build_dataset.py --source hf --limit 15000 --skip-verify \
  --max-tests 10 --concurrency 64 --out data/clean_problems.jsonl

python scripts/split_stages.py --in data/clean_problems.jsonl \
  --rl-out data/rl_pool.jsonl --dev-out data/dev_internal.jsonl \
  --sft-out data/unused_sft.jsonl --dev 1000 --sft 0 --seed 0
```

Probe with the **SFT adapter** (the policy GRPO will start from) and the **same**
`--max-tokens` as GRPO's `--max-completion`, so the kept difficulty band matches
what the trainer actually sees:

```bash
python scripts/difficulty_filter.py --model Qwen/Qwen3-4B \
  --lora outputs/qwen3_4b_sft \
  --in data/rl_pool.jsonl --out data/grpo_train.jsonl \
  --save-rollouts outputs/filter_rollouts_6k.jsonl \
  --max-problems 6000 \
  --k 8 --keep-lo 1 --keep-hi 7 \
  --max-tokens 4096 --reward-timeout 3 \
  --score-concurrency 64 --score-batch-size 512
```

Scoring interrupted after rollouts were saved? Resume with
`--load-rollouts outputs/filter_rollouts_6k.jsonl` (skips regeneration).

## 5. GRPO smoke, then real run

Continue the SFT adapter with `--init-adapter`:

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-4B \
  --init-adapter outputs/qwen3_4b_sft \
  --data data/grpo_train.jsonl --limit 256 \
  --num-generations 4 --per-device-batch 4 --grad-accum 8 \
  --max-prompt 2048 --max-completion 4096 --lr 5e-6 \
  --max-steps 100 --bf16 --gradient-checkpointing \
  --output outputs/qwen3_4b_grpo_smoke
```

Success criterion: reward mean should not collapse, `frac_reward_zero_std`
should stay well below 1, and **`completions/clipped_ratio` should be small**
(the old 1024 cap truncated ~half the rollouts — that was the bug). Metrics go to
`<output>/metrics.jsonl` and a plot to `<output>/metrics.png`.

Real run (larger GPU: `--num-generations 8 --per-device-batch 8`):

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-4B \
  --init-adapter outputs/qwen3_4b_sft \
  --data data/grpo_train.jsonl \
  --num-generations 8 --per-device-batch 8 --grad-accum 8 \
  --max-prompt 2048 --max-completion 4096 --lr 5e-6 \
  --epochs 1 --bf16 --gradient-checkpointing \
  --output outputs/qwen3_4b_grpo
```

## 6. Final eval

```bash
python scripts/eval_model.py --model Qwen/Qwen3-4B \
  --lora outputs/qwen3_4b_grpo \
  --data data/dev_internal.jsonl \
  --out outputs/eval/qwen3_4b_grpo.json

# EvalPlus sanity (function-completion, different prompt — basics check only)
python scripts/eval_evalplus.py --model Qwen/Qwen3-4B \
  --lora outputs/qwen3_4b_grpo \
  --dataset humaneval --out outputs/eval/he_grpo.jsonl
evalplus.evaluate --dataset humaneval --samples outputs/eval/he_grpo.jsonl
```

For the external headline number, run LiveCodeBench separately against the
official harness. The wrapper below registers `Qwen/Qwen3-4B` with official LCB
at runtime and delegates judging to `lcb_runner`:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1

python scripts/eval_livecodebench.py --lcb-root /workspace/LiveCodeBench \
  --model Qwen/Qwen3-4B \
  --release-version release_v5 \
  --start-date 2024-10-01 --end-date 2025-02-28 \
  --model-style QwQ \
  --n 1 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --max-model-len 40960 --gpu-mem-util 0.95 \
  --timeout 10 --num-process-evaluate 8
```

For an SFT/GRPO LoRA, merge first because official LiveCodeBench does not load
PEFT adapters directly:

```bash
python scripts/merge_lora.py --base Qwen/Qwen3-4B \
  --adapter outputs/qwen3_4b_grpo --out outputs/qwen3_4b_grpo_merged

python scripts/eval_livecodebench.py --lcb-root /workspace/LiveCodeBench \
  --model qwen3_4b_grpo_merged \
  --local-model-path outputs/qwen3_4b_grpo_merged \
  --release-version release_v5 \
  --start-date 2024-10-01 --end-date 2025-02-28 \
  --model-style QwQ \
  --n 1 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-tokens 32768 --max-model-len 40960 --gpu-mem-util 0.95 \
  --timeout 10 --num-process-evaluate 8
```

The previous stdin/stdout-only local judge is still available as
`scripts/eval_livecodebench_subset.py` for quick internal comparisons.

## Notes

- If vLLM shows Triton/JIT compilation messages on first run, that is usually
  warm-up. Check GPU utilization before killing it.
- If SFT/GRPO OOMs: lower `--max-length`/`--max-completion`, `--per-device-batch`,
  or `--num-generations`; keep LoRA (drop `--full-ft`).
- Stop GPU billing between sessions. Upload important adapters to the Hub.
