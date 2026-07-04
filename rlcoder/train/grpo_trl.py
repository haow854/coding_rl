"""Single-turn GRPO (RLVR) training with TRL on the GPU box.

Stage-2 route: start from the Stage-1 SFT checkpoint. Pass the SFT LoRA adapter
via --init-adapter to continue training it, or --full-ft for full-parameter RL.

Smoke run (continue the SFT adapter):
    python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-4B \
        --init-adapter outputs/qwen3_4b_sft \
        --data data/grpo_train.jsonl --limit 256 \
        --num-generations 4 --per-device-batch 4 --max-completion 4096 \
        --max-steps 100 --bf16 --gradient-checkpointing

First real run:
    python rlcoder/train/grpo_trl.py --model Qwen/Qwen3-4B \
        --init-adapter outputs/qwen3_4b_sft --data data/grpo_train.jsonl \
        --num-generations 8 --per-device-batch 8 --max-completion 4096 \
        --epochs 1 --bf16 --gradient-checkpointing \
        --output outputs/qwen3_4b_grpo

Targets TRL >= 0.15 (GRPOConfig / GRPOTrainer). Defaults are DAPO-flavoured:
loss_type="dr_grpo" (no length bias) and epsilon_high>epsilon (clip-higher),
plus a small KL (beta) as a stability anchor for a small student — set --beta 0
for pure DAPO. Field names evolve across TRL versions, so we keep only fields
supported by the installed version.

Batch math: per_device_train_batch_size counts generations and must be
divisible by num_generations; unique prompts per optimizer step =
per_device_batch * grad_accum * world_size / num_generations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _prompt_token_len(processing_class, messages) -> int:
    from rlcoder.prompting import render_chat_prompt

    rendered = render_chat_prompt(processing_class, messages)
    ids = processing_class(rendered, add_special_tokens=False)["input_ids"]
    return len(ids)


def build_hf_dataset(data_path: str, limit, processing_class=None,
                     max_prompt_tokens=None):
    from datasets import Dataset

    from rlcoder.data.load import load_clean_jsonl
    from rlcoder.prompting import build_messages

    problems = load_clean_jsonl(data_path, limit=limit)
    rows = []
    dropped_long = 0
    for p in problems:
        messages = build_messages(p)
        if processing_class is not None and max_prompt_tokens is not None:
            if _prompt_token_len(processing_class, messages) > max_prompt_tokens:
                dropped_long += 1
                continue
        rows.append({"prompt": messages, "tests": p.tests, "mode": p.mode})
    if dropped_long:
        print(f"dropped {dropped_long} problems with prompt > {max_prompt_tokens} tokens")
    return Dataset.from_list(rows)


def _plot_metrics(metrics_path: str, png_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not plot metrics: {e}")
        return

    rows = []
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return

    def series(key):
        xs, ys = [], []
        for row in rows:
            if key in row:
                try:
                    ys.append(float(row[key]))
                    xs.append(int(row.get("step", len(xs))))
                except (TypeError, ValueError):
                    pass
        return xs, ys

    plots = [
        ("reward", "reward"),
        ("reward_std", "reward std"),
        ("frac_reward_zero_std", "zero-std frac"),
        ("completions/mean_length", "mean completion len"),
        ("completions/clipped_ratio", "clipped ratio"),
        ("loss", "loss"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flatten(), plots):
        xs, ys = series(key)
        ax.set_title(title)
        if xs:
            ax.plot(xs, ys)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"wrote metrics plot -> {png_path}")


def make_jsonl_metrics_callback(base_cls, path: str, plot_path: str | None = None):
    class JsonlMetricsCallback(base_cls):
        def __init__(self):
            self.path = path
            self.plot_path = plot_path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8"):
                pass

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {"step": state.global_step}
            row.update(logs)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        def on_train_end(self, args, state, control, **kwargs):
            if self.plot_path:
                _plot_metrics(self.path, self.plot_path)

    return JsonlMetricsCallback()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--init-adapter", default=None,
                    help="Optional LoRA adapter to continue from; leave unset "
                         "for the main post-trained direct-GRPO route.")
    ap.add_argument("--data", default="data/train_problems.jsonl")
    ap.add_argument("--output", default="outputs/grpo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=1536)
    ap.add_argument("--max-completion", type=int, default=4096,
                    help="Thinking traces are long; 1024 truncated ~half of them "
                         "and destroyed the reward signal. Raise to 8192 if VRAM "
                         "allows, or lower --num-generations to afford it.")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.001,
                    help="KL coeff; small anchor for a small student. 0 = pure DAPO.")
    ap.add_argument("--loss-type", default="dr_grpo", help="grpo | dr_grpo | bnpo | dapo")
    ap.add_argument("--epsilon-high", type=float, default=0.28, help="DAPO clip-higher")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--full-ft", action="store_true",
                    help="Full-parameter GRPO instead of LoRA (needs more VRAM).")
    ap.add_argument("--reward-timeout", type=float, default=10.0)
    ap.add_argument("--use-vllm", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--log-steps", type=int, default=1)
    ap.add_argument("--log-completions", action="store_true",
                    help="Print TRL's rich completion table during training.")
    ap.add_argument("--metrics-jsonl", default=None,
                    help="Path for per-log metrics JSONL. Defaults to "
                         "<output>/metrics.jsonl.")
    ap.add_argument("--metrics-plot", default=None,
                    help="Path for an automatic metrics PNG. Defaults to "
                         "<output>/metrics.png.")
    args = ap.parse_args()

    import inspect
    from dataclasses import fields as _fields

    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from rlcoder.prompting import load_processing_class
    from rlcoder.rollout.single_turn import make_reward_fn
    from rlcoder.train.model_loading import load_base_model, load_peft_model

    processing_class = load_processing_class(args.model)
    train_ds = build_hf_dataset(
        args.data,
        args.limit,
        processing_class=processing_class,
        max_prompt_tokens=args.max_prompt,
    )
    print(f"training on {len(train_ds)} verified problems from {args.data}")

    reward_fn = make_reward_fn(
        timeout=args.reward_timeout,
        concurrency=max(8, args.per_device_batch),
    )

    if args.init_adapter:
        model = load_peft_model(args.init_adapter, args.model, bf16=args.bf16,
                                is_trainable=True)
        peft_config = None
        print(f"continuing GRPO from adapter: {args.init_adapter}")
    else:
        model = load_base_model(args.model, bf16=args.bf16)
        if args.full_ft:
            peft_config = None
            print("full-parameter GRPO (no LoRA adapter)")
        else:
            peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=2 * args.lora_r,
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    desired = dict(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
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
        log_completions=args.log_completions,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        report_to="none",
    )
    valid = {f.name for f in _fields(GRPOConfig)}
    if "max_prompt_length" in valid:
        desired["max_prompt_length"] = args.max_prompt
    else:
        print("[info] installed TRL has no max_prompt_length; "
              "using dataset-side prompt filtering instead")
    dropped = sorted(set(desired) - valid)
    if dropped:
        import trl

        print(f"[warn] TRL {trl.__version__} ignores: {dropped}")
    cfg = GRPOConfig(**{k: v for k, v in desired.items() if k in valid})

    trainer_kwargs = dict(
        model=model,
        reward_funcs=[reward_fn],
        args=cfg,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    params = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = processing_class
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = processing_class

    trainer = GRPOTrainer(**trainer_kwargs)
    metrics_jsonl = args.metrics_jsonl or os.path.join(args.output, "metrics.jsonl")
    metrics_plot = args.metrics_plot or os.path.join(args.output, "metrics.png")
    callback = make_jsonl_metrics_callback(TrainerCallback, metrics_jsonl, metrics_plot)
    trainer.add_callback(callback)
    trainer.train()
    trainer.save_model(args.output)
    print(f"saved LoRA adapter -> {args.output}")


if __name__ == "__main__":
    main()
