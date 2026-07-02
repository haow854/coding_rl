# AutoDL quickstart

Local dev is CPU-only. Everything below runs on the rented GPU box.

## 0. Model and GPU

Stage 1 starts from the post-trained checkpoint:

```text
Qwen/Qwen3.5-2B
```

Use a 24G card for smoke tests if cost matters. A 40G card is more comfortable
for vLLM probing and GRPO with longer completions. An 80G card is convenient,
but not required for the first 2B run.

Do not attach the old `Qwen/Qwen3.5-2B-Base` SFT adapter to this model.

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

If vLLM tries to downgrade torch or compile forever in a broken image, fix the
environment before launching long jobs.

## 2. Build verified data

Main dataset:

```text
open-r1/verifiable-coding-problems-python_decontaminated-tested
```

For the first run, trust the upstream decontaminated/tested rows and normalize
them locally:

```bash
python scripts/build_dataset.py --source hf --limit 15000 --skip-verify \
  --max-tests 10 --concurrency 64 \
  --out data/clean_problems.jsonl
```

Split into internal dev and RL pool. No SFT split is needed for the main route:

```bash
python scripts/split_stages.py --in data/clean_problems.jsonl \
  --rl-out data/rl_pool.jsonl \
  --dev-out data/dev_internal.jsonl \
  --sft-out data/unused_sft.jsonl \
  --dev 1000 --sft 0 --seed 0
```

If you want stricter data quality later, rebuild without `--skip-verify`; it is
slower and CPU-bound:

```bash
python scripts/build_dataset.py --source hf --limit 15000 \
  --max-tests 10 --concurrency 16 --timeout 5 \
  --out data/clean_problems_verified.jsonl
```

## 3. Post-trained baseline eval

Start with `n=1` for a cheap signal:

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B \
  --data data/dev_internal.jsonl --limit 200 \
  --n 1 --temperature 0.2 --ks 1 \
  --out outputs/eval/qwen3_5_2b_post_dev200.json
```

Then run a wider sample if the 200-problem result looks sane:

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B \
  --data data/dev_internal.jsonl \
  --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/qwen3_5_2b_post_dev1000_n5.json
```

## 4. Difficulty filter for GRPO

Probe the same post-trained policy and keep problems it solves
sometimes-but-not-always. These have useful within-prompt reward variance.

Cheap probe:

```bash
python scripts/difficulty_filter.py --model Qwen/Qwen3.5-2B \
  --in data/rl_pool.jsonl --out data/grpo_train_probe.jsonl \
  --max-problems 1000 \
  --k 4 --keep-lo 1 --keep-hi 3 --max-tokens 1536
```

If the kept count is reasonable, run a larger filter:

```bash
python scripts/difficulty_filter.py --model Qwen/Qwen3.5-2B \
  --in data/rl_pool.jsonl --out data/grpo_train.jsonl \
  --max-problems 6000 \
  --k 4 --keep-lo 1 --keep-hi 3 --max-tokens 1536
```

If this keeps too few problems, relax to `--keep-lo 0 --keep-hi 3` for the
first smoke run, then tighten later.

## 5. GRPO smoke

Start without an init adapter:

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3.5-2B \
  --data data/grpo_train_probe.jsonl --limit 256 \
  --num-generations 4 --per-device-batch 4 --grad-accum 4 \
  --max-prompt 1536 --max-completion 1024 --lr 5e-6 \
  --max-steps 100 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_grpo_smoke
```

Success criterion: reward mean should not collapse, `reward_std` should not be
zero for nearly every group, and completions should not grow uncontrollably.

## 6. GRPO first real run

Use the filtered training pool:

```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3.5-2B \
  --data data/grpo_train.jsonl \
  --num-generations 4 --per-device-batch 4 --grad-accum 8 \
  --max-prompt 1536 --max-completion 1536 --lr 5e-6 \
  --epochs 1 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_grpo
```

If reward improves but training is noisy, rerun with `--num-generations 8
--per-device-batch 8` on a larger GPU.

## 7. Final eval

Evaluate the GRPO adapter against the same dev set:

```bash
python scripts/eval_model.py --model Qwen/Qwen3.5-2B \
  --lora outputs/qwen3_5_2b_grpo \
  --data data/dev_internal.jsonl \
  --n 5 --temperature 0.8 --ks 1,5 \
  --out outputs/eval/qwen3_5_2b_grpo_dev1000_n5.json
```

EvalPlus sanity:

```bash
python scripts/eval_evalplus.py --model Qwen/Qwen3.5-2B \
  --lora outputs/qwen3_5_2b_grpo \
  --dataset humaneval --out outputs/eval/he_grpo.jsonl
evalplus.evaluate --dataset humaneval --samples outputs/eval/he_grpo.jsonl
```

## Optional: SFT ablation

Only do this as a separate experiment. Keep it clearly named and do not mix it
with the Base adapter you already trained.

```bash
python rlcoder/train/sft_trl.py --model Qwen/Qwen3.5-2B \
  --data data/sft_train.jsonl --limit 1000 \
  --per-device-batch 2 --grad-accum 8 --max-length 4096 \
  --lr 5e-5 --epochs 1 \
  --bf16 --gradient-checkpointing \
  --output outputs/qwen3_5_2b_sft_ablation
```

## Notes

- If vLLM shows Triton/JIT compilation messages on first run, that is usually
  warm-up. Check GPU utilization before killing it.
- If GRPO OOMs, reduce `--max-completion`, `--per-device-batch`, or `--lora-r`.
- Stop GPU billing between sessions. Keep the disk and upload important
  adapters/checkpoints to the Hub.
