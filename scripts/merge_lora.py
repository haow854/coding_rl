"""Merge a trained LoRA adapter into the base model.

Most first-stage runs can evaluate the adapter directly with scripts/eval_model.py.
Merge only when an external benchmark harness cannot load LoRA adapters.

Example:
    python scripts/merge_lora.py --base Qwen/Qwen3.5-2B-Base \
        --adapter outputs/qwen3_5_2b_grpo --out outputs/qwen3_5_2b_merged
"""
import argparse


def _load_base_model(base: str):
    import torch

    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    except Exception as first_error:  # noqa: BLE001
        try:
            from transformers import AutoModelForMultimodalLM

            return AutoModelForMultimodalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        except Exception:  # noqa: BLE001
            raise first_error


def _save_processor_or_tokenizer(base: str, out: str) -> None:
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(base).save_pretrained(out)
        return
    except Exception:  # noqa: BLE001
        pass

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from peft import PeftModel

    base = _load_base_model(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    _save_processor_or_tokenizer(args.base, args.out)
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
