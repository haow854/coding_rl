#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python rl/scripts/eval_model.py \
  --model Qwen/Qwen3-1.7B \
  --no-thinking \
  --data data/dev_internal.jsonl \
  --n 1 \
  --temperature 0.6 \
  --max-tokens 2048 \
  --out outputs/eval/base_nothink.json

python rl/scripts/eval_livecodebench.py \
  --model Qwen/Qwen3-1.7B \
  --no-think \
  --model-repr Qwen3-1.7B-nothink \
  --release-version release_v5 \
  --start-date 2024-10-01 \
  --end-date 2025-02-28 \
  --n 1 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 8192 \
  --max-model-len 16384 \
  --gpu-mem-util 0.95 \
  --timeout 10

python rl/scripts/difficulty_filter.py \
  --model Qwen/Qwen3-1.7B \
  --no-thinking \
  --in data/rl_pool.jsonl \
  --out data/grpo_train_nothink.jsonl \
  --k 8 \
  --keep-lo 1 \
  --keep-hi 7 \
  --max-tokens 2048
