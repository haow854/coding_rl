"""Optional LoRA SFT script for ablations.

The main first-stage route now starts from the post-trained Qwen3.5-2B model
and runs GRPO directly. Use this script only when you intentionally want a
small format/data ablation; do not reuse a Base-trained SFT adapter on the
post-trained checkpoint.

Example on the GPU box:
    python rlcoder/train/sft_trl.py --model Qwen/Qwen3.5-2B \
        --data data/sft_train.jsonl --limit 1000 \
        --bf16 --gradient-checkpointing --output outputs/qwen3_5_2b_sft_ablation
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
    from rlcoder.prompting import build_messages

    problems = load_clean_jsonl(data_path, limit=limit)
    rows = []
    for p in problems:
        if not p.gold_solution:
            continue
        rows.append(
            {
                "prompt": build_messages(p),
                "completion": [
                    {"role": "assistant", "content": p.gold_solution.strip()}
                ],
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--data", default="data/train_problems.jsonl")
    ap.add_argument("--output", default="outputs/sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--eos-token", default="<|im_end|>")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--log-steps", type=int, default=5)
    args = ap.parse_args()

    import inspect
    from dataclasses import fields as _fields

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from rlcoder.prompting import load_processing_class
    from rlcoder.train.model_loading import load_base_model

    train_ds = build_hf_dataset(args.data, args.limit)
    print(f"SFT on {len(train_ds)} verified gold solutions from {args.data}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    desired = dict(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        packing=args.packing,
        completion_only_loss=True,
        eos_token=args.eos_token,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        report_to="none",
    )
    valid = {f.name for f in _fields(SFTConfig)}
    dropped = sorted(set(desired) - valid)
    if dropped:
        import trl

        print(f"[warn] TRL {trl.__version__} ignores: {dropped}")
    cfg = SFTConfig(**{k: v for k, v in desired.items() if k in valid})

    model = load_base_model(args.model, bf16=args.bf16)
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    processing_class = load_processing_class(args.model)

    trainer_kwargs = dict(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = processing_class
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = processing_class

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved SFT LoRA adapter -> {args.output}")


if __name__ == "__main__":
    main()
