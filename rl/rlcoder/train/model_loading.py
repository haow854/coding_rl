"""Model loading helpers for Qwen-style text and multimodal checkpoints."""
from __future__ import annotations

from typing import Optional


def _dtype_kwargs(bf16: bool) -> dict:
    if not bf16:
        return {}
    import torch

    return {"torch_dtype": torch.bfloat16}


def _flash_attn_implementation() -> Optional[str]:
    """TRL packing/padding-free needs a flash-attention variant to keep packed
    samples from attending to each other; SDPA silently cross-contaminates
    them (and is far slower on packed batches). Prefer locally compiled
    flash-attn; fall back to the hub-prebuilt kernel (`pip install kernels`),
    which needs no matching wheel or nvcc build."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        pass
    try:
        import kernels  # noqa: F401

        return "kernels-community/flash-attn2"
    except ImportError:
        return None


def load_base_model(model_name: str, bf16: bool = False):
    kwargs = _dtype_kwargs(bf16)
    impl = _flash_attn_implementation()
    if impl:
        kwargs["attn_implementation"] = impl
        print(f"attention implementation: {impl}")

    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as first_error:  # noqa: BLE001
        try:
            from transformers import AutoModelForMultimodalLM

            return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
        except Exception:  # noqa: BLE001
            raise first_error


def load_peft_model(adapter_path: str, fallback_base: str, bf16: bool = False,
                    is_trainable: bool = True):
    from peft import PeftConfig, PeftModel

    cfg = PeftConfig.from_pretrained(adapter_path)
    base_name: Optional[str] = getattr(cfg, "base_model_name_or_path", None)
    base = load_base_model(base_name or fallback_base, bf16=bf16)
    return PeftModel.from_pretrained(base, adapter_path, is_trainable=is_trainable)
