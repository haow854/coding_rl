"""Stage-1 distillation SFT: teach the base model to reason then code.

This is now the MAIN first stage, not an ablation. We SFT on competitive-code
reasoning traces (scripts/build_sft_data.py -> OpenCodeReasoning) so the policy
learns the <think>...</think> + fenced-program format before any GRPO. Loss is
masked to the completion only; prompts are pre-rendered in thinking mode so the
target keeps its reasoning block (some chat templates strip <think> from
message-form assistant turns).

Example on the GPU box (LoRA):
    python rlcoder/train/sft_trl.py --model Qwen/Qwen3-4B \
        --data data/sft_ocr.jsonl --max-length 16384 \
        --bf16 --gradient-checkpointing --packing \
        --output outputs/qwen3_4b_sft

Full-parameter (needs more VRAM; lower the LR):
    python rlcoder/train/sft_trl.py --model Qwen/Qwen3-4B --data data/sft_ocr.jsonl \
        --full-ft --lr 1e-5 --max-length 16384 --bf16 --gradient-checkpointing \
        --output outputs/qwen3_4b_sft_full
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def build_hf_dataset(data_path: str, limit, processing_class, enable_thinking: bool = True):
    """Prompt-completion dataset with PRE-RENDERED prompt strings.

    Rendering the prompt ourselves (rather than handing TRL message lists) keeps
    the target's <think> block intact: Qwen3's chat template strips reasoning
    from message-form assistant turns, which would silently delete the very
    thing we are distilling. completion_only_loss then masks the prompt tokens,
    and TRL appends the EOS so the model learns to stop.
    """
    from datasets import Dataset

    from rlcoder.data.load import load_clean_jsonl
    from rlcoder.prompting import build_messages, render_chat_prompt

    problems = load_clean_jsonl(data_path, limit=limit)
    rows = []
    for p in problems:
        if not p.gold_solution:
            continue
        prompt = render_chat_prompt(processing_class, build_messages(p),
                                    enable_thinking=enable_thinking)
        rows.append({"prompt": prompt, "completion": p.gold_solution.strip()})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="data/sft_ocr.jsonl")
    ap.add_argument("--output", default="outputs/sft")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=16384,
                    help="R1 traces are long; too small truncates the target "
                         "mid-reasoning and teaches the model never to stop.")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA default; pass ~1e-5 with --full-ft.")
    ap.add_argument("--optim", default="adamw_torch",
                    help="Use adamw_8bit (needs bitsandbytes) if full-FT OOMs.")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--full-ft", action="store_true",
                    help="Full-parameter SFT instead of LoRA (needs more VRAM).")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Render prompts in non-thinking mode (format ablation).")
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

    processing_class = load_processing_class(args.model)
    train_ds = build_hf_dataset(args.data, args.limit, processing_class,
                                enable_thinking=not args.no_thinking)
    print(f"SFT on {len(train_ds)} reasoning traces from {args.data}")

    if args.full_ft:
        peft_config = None
        print("full-parameter SFT (no LoRA adapter)")
    else:
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
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim=args.optim,
        max_grad_norm=1.0,
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
    print(f"saved SFT {'model' if args.full_ft else 'LoRA adapter'} -> {args.output}")


if __name__ == "__main__":
    main()
