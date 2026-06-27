"""Merge a trained LoRA adapter into the base model, so external benchmark
harnesses (LiveCodeBench, BigCodeBench) can load it as a normal HF model.

    python scripts/merge_lora.py --base Qwen/Qwen3-14B \
        --adapter outputs/qwen3_14b_grpo --out outputs/qwen3_14b_merged
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
