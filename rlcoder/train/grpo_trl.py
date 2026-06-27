"""Single-turn GRPO (RLVR) training with TRL + LoRA — runs on the AutoDL GPU box.

Smoke run (prove the reward rises; small model, cheap):
    python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-1.7B --limit 256 \
        --num-generations 8 --max-steps 60 --use-vllm --bf16

Real run (Qwen3-14B, single A100-80G):
    python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-14B --limit 6000 \
        --num-generations 8 --max-completion 3072 --epochs 2 \
        --use-vllm --bf16 --gradient-checkpointing

Targets TRL >= 0.15 (GRPOConfig / GRPOTrainer). A few defaults are already
DAPO-flavoured: beta=0 (no KL), loss_type="dr_grpo" (no length bias), and
epsilon_high>epsilon (clip-higher). Field names evolve across TRL versions —
verify against the one you install.

Batch math: per_device_train_batch_size counts *generations* and must be
divisible by num_generations; unique prompts per optimizer step =
per_device_batch * grad_accum * world_size / num_generations.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_hf_dataset(data_path: str, limit):
    from datasets import Dataset

    from rlcoder.data.load import load_clean_jsonl
    from rlcoder.rollout.prompt import build_messages

    problems = load_clean_jsonl(data_path, limit=limit)
    rows = [{"prompt": build_messages(p), "tests": p.tests, "mode": p.mode} for p in problems]
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/clean_problems.jsonl")
    ap.add_argument("--output", default="outputs/grpo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=8)      # G (group size)
    ap.add_argument("--per-device-batch", type=int, default=8)     # counts generations
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--max-completion", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coeff; 0 = DAPO-style")
    ap.add_argument("--loss-type", default="dr_grpo", help="grpo | dr_grpo | bnpo")
    ap.add_argument("--epsilon-high", type=float, default=0.28, help="DAPO clip-higher")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--reward-timeout", type=float, default=10.0)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--log-steps", type=int, default=1)
    args = ap.parse_args()

    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from rlcoder.rollout.single_turn import make_reward_fn

    train_ds = build_hf_dataset(args.data, args.limit)
    print(f"training on {len(train_ds)} verified problems from {args.data}")

    # score the whole generation batch concurrently in the sandbox
    reward_fn = make_reward_fn(
        timeout=args.reward_timeout,
        concurrency=max(8, args.per_device_batch),
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Build the config tolerantly: keep only kwargs the installed TRL supports,
    # so this runs across TRL versions without crashing on renamed/new fields.
    from dataclasses import fields as _fields

    desired = dict(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,
        epsilon_high=args.epsilon_high,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        log_completions=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        report_to="none",
    )
    valid = {f.name for f in _fields(GRPOConfig)}
    dropped = sorted(set(desired) - valid)
    if dropped:
        import trl
        print(f"[warn] TRL {trl.__version__} ignores: {dropped}")
    cfg = GRPOConfig(**{k: v for k, v in desired.items() if k in valid})

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved LoRA adapter -> {args.output}")


if __name__ == "__main__":
    main()
