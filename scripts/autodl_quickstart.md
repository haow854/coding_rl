# AutoDL quickstart

Local dev is CPU-only (sandbox/reward/data are stdlib and already tested);
everything below runs on the rented GPU box.

## 0. Pick model by GPU (the GPU decides what fits)
| GPU (1×)              | model            | train flags                                  |
|----------------------|------------------|----------------------------------------------|
| A100 / A800 80G      | `Qwen/Qwen3-14B` | `--bf16 --gradient-checkpointing`            |
| A100 40G             | `Qwen/Qwen3-14B` | add `--per-device-batch 4 --max-completion 2048` |
| 4090 / 3090 24G      | `Qwen/Qwen3-8B`  | `--per-device-batch 4 --max-completion 2048` |

## 1. Environment (Python 3.10/3.11 image, CUDA 12.1)
```bash
pip install -r requirements-gpu.txt
# torch usually ships with the image; else: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 2. Build the clean, gold-verified pool (~10–20 min for the full set)
```bash
python scripts/build_dataset.py --source hf --limit 6000 \
    --max-tests 10 --concurrency 64 --out data/clean_problems.jsonl
```

## 3. Difficulty-filter with the base model (DAPO dynamic sampling, one pass)
```bash
python scripts/difficulty_filter.py --model Qwen/Qwen3-14B \
    --in data/clean_problems.jsonl --out data/train_problems.jsonl \
    --k 8 --keep-lo 1 --keep-hi 7 --max-tokens 3072
# keeps problems the base solves 1..7 / 8 times -> learnable reward signal
```

## 4. Smoke run — prove the reward rises (small model, cheap, ~30 min)
```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-1.7B \
    --data data/train_problems.jsonl --limit 256 \
    --num-generations 8 --max-steps 60 --max-completion 1536 \
    --use-vllm --bf16 --output outputs/smoke
# success = logged mean reward trends up over steps
```

## 5. Real run — Qwen3-14B on a single 80G card
```bash
python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-14B \
    --data data/train_problems.jsonl \
    --num-generations 8 --max-completion 3072 --epochs 2 \
    --use-vllm --bf16 --gradient-checkpointing --output outputs/qwen3_14b_grpo
```

## 6. Evaluate (base vs RL — the numbers for your writeup)
```bash
# hold out an eval set disjoint from training
python scripts/split_data.py --in data/clean_problems.jsonl \
    --train-out data/train_problems.jsonl --holdout-out data/holdout.jsonl --holdout 200

# in-house competitive held-out: baseline vs RL (same judge as training)
python scripts/eval_model.py --model Qwen/Qwen3-14B --data data/holdout.jsonl \
    --n 5 --temperature 0.8 --ks 1,5 --out outputs/eval/base.json
python scripts/eval_model.py --model Qwen/Qwen3-14B --lora outputs/qwen3_14b_grpo \
    --data data/holdout.jsonl --n 5 --temperature 0.8 --ks 1,5 --out outputs/eval/rl.json

# HumanEval+/MBPP+ regression sanity
python scripts/eval_evalplus.py --model Qwen/Qwen3-14B --dataset humaneval --out outputs/eval/he.jsonl
evalplus.evaluate --dataset humaneval --samples outputs/eval/he.jsonl

# LiveCodeBench / BigCodeBench (headline, contamination-free): merge then run the
# OFFICIAL harness on the merged model, over a post-cutoff window.
python scripts/merge_lora.py --base Qwen/Qwen3-14B --adapter outputs/qwen3_14b_grpo --out outputs/merged
# pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git  # then run its runner on outputs/merged
```

## Notes
- **Checkpoints**: AutoDL sessions can drop — `--save-steps` writes the LoRA
  adapter; `huggingface-cli upload` it so a restart resumes.
- **OOM**: lower `--max-completion` / `--per-device-batch` / `--lora-r`; keep
  `--gradient-checkpointing`.
- **Stop GPU billing** between sessions with AutoDL 关机 (disk kept).
- DAPO defaults are on (`--beta 0 --loss-type dr_grpo --epsilon-high 0.28`); for
  the vanilla-GRPO ablation: `--beta 0.04 --loss-type grpo --epsilon-high 0.2`.
