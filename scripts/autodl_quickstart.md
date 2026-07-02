# AutoDL quickstart

Local dev is CPU-only. Everything below runs on the rented GPU box.

## 0. Model and GPU

Stage 1 targets:

```text
Qwen/Qwen3.5-2B-Base
```

Use a 24G card for the first smoke run if cost matters. A 40G or 80G card gives
more room for longer completions and larger GRPO batches.

## 1. Environment

Use an AutoDL image with PyTorch already installed, then install the project
deps. Do not reinstall torch unless the image is broken.

```bash
pip install -U -r requirements-gpu.txt
```

Install vLLM only if it matches the image's torch/CUDA stack:

```bash
pip install "vllm>=0.10"
```

If vLLM tries to downgrade torch, skip vLLM for the first smoke run.

## 2. Build verified data

Two separate pools now: an **RL pool** (bare gold answers, used for GRPO +
eval) and an **SFT pool** (reasoning traces, used only for the SFT warm-up).
They come from different upstream datasets and are not split from each other.

### 2a. RL pool

```text
open-r1/verifiable-coding-problems-python_decontaminated-tested
```

It is already decontaminated/tested upstream. For the first run, trust the
upstream-tested rows and skip local verification; rerun strict verification
later before GRPO if you want the cleanest reward environment.

```bash
python scripts/build_dataset.py --source hf --limit 6000 --skip-verify \
  --max-tests 10 --concurrency 64 \
  --out data/clean_problems.jsonl

python scripts/split_stages.py --in data/clean_problems.jsonl \
  --rl-out data/rl_pool.jsonl \
  --dev-out data/dev_internal.jsonl \
  --sft-out data/unused.jsonl \
  --dev 1000 --sft 0 --seed 0
```

Strict verification version, slower and CPU-bound:

```bash
python scripts/build_dataset.py --source hf --limit 6000 \
  --max-tests 10 --concurrency 16 --timeout 5 \
  --out data/clean_problems_verified.jsonl
```

### 2b. SFT pool (reasoning traces)

```text
open-r1/codeforces-cots  (config: solutions_py_decontaminated)
```

Each row already carries a `<think>...</think>` + fenced-code assistant turn
distilled from DeepSeek-R1 — this is what SFT imitates, not a bare answer.
Only ~84% of upstream solutions actually pass their own tests, so **do not**
skip verification here:

```bash
python scripts/build_dataset.py --source cots --limit 3000 \
  --max-tests 10 --concurrency 16 --timeout 5 \
  --out data/sft_train.jsonl
```

## 3. Base eval

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B-Base \
  --data data/dev_internal.jsonl --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/base.json
```

Also record the official Instruct checkpoint as a reference point for the
headline table (no LoRA, it already has its own chat template):

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B-Instruct \
  --data data/dev_internal.jsonl --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/instruct.json
```

## 4. SFT warm-up

Reasoning traces run longer than bare code; `--max-length 4096` is already
generous but watch for truncated/OOM cases and raise `--grad-accum` instead of
lowering this if you need to save memory.

```bash
python rlcoder/train/sft_trl.py --model Qwen/Qwen3.5-2B-Base \
  --data data/sft_train.jsonl --limit 3000 \
  --per-device-batch 2 --grad-accum 8 --max-length 4096 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_sft
```

Evaluate the SFT adapter:

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B-Base \
  --lora outputs/qwen3_5_2b_sft \
  --data data/dev_internal.jsonl --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/sft.json
```

## 5. Difficulty filter for GRPO

Probe the SFT policy and keep problems it solves sometimes-but-not-always. Those
problems have reward variance, which is what GRPO needs.

```bash
python scripts/difficulty_filter.py --model Qwen/Qwen3.5-2B-Base \
  --lora outputs/qwen3_5_2b_sft \
  --in data/rl_pool.jsonl --out data/grpo_train.jsonl \
  --k 8 --keep-lo 1 --keep-hi 7 --max-tokens 2048
```

If this keeps too few problems, relax to `--keep-lo 0 --keep-hi 7` for the first
smoke run, then tighten later.

## 6. GRPO smoke

Start without vLLM if adapter + colocated vLLM is unstable in your environment.

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3.5-2B-Base \
  --init-adapter outputs/qwen3_5_2b_sft \
  --data data/grpo_train.jsonl --limit 256 \
  --num-generations 8 --per-device-batch 8 --grad-accum 4 \
  --max-prompt 1536 --max-completion 1536 --max-steps 60 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_grpo_smoke
```

Success criterion: reward mean trends upward and `frac_reward_zero_std` is not
near 1.0 for the whole run.

## 7. GRPO first real run

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3.5-2B-Base \
  --init-adapter outputs/qwen3_5_2b_sft \
  --data data/grpo_train.jsonl \
  --num-generations 8 --per-device-batch 8 --grad-accum 4 \
  --max-prompt 1536 --max-completion 2048 --epochs 1 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_grpo
```

## 8. Final eval

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B-Base \
  --lora outputs/qwen3_5_2b_grpo \
  --data data/dev_internal.jsonl --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/grpo.json
```

EvalPlus sanity:

```bash
python scripts/eval_evalplus.py --model Qwen/Qwen3.5-2B-Base \
  --lora outputs/qwen3_5_2b_grpo \
  --dataset humaneval --out outputs/eval/he_grpo.jsonl
evalplus.evaluate --dataset humaneval --samples outputs/eval/he_grpo.jsonl
```

## Notes

- If SFT OOMs: reduce `--max-length`, then increase `--grad-accum` to preserve
  effective batch size.
- If GRPO OOMs: reduce `--max-completion`, `--per-device-batch`, or `--lora-r`.
- If adapter loading fails for Qwen3.5 in a specific TRL/PEFT version, upgrade
  `transformers`, `trl`, and `peft` first; Qwen3.5 support is recent.
- Stop GPU billing between sessions. Keep the disk and upload important
  adapters/checkpoints to the Hub.
