"""Model loading helpers for Qwen-style text and multimodal checkpoints."""
from __future__ import annotations

from typing import Optional


def _dtype_kwargs(bf16: bool) -> dict:
    if not bf16:
        return {}
    import torch

    return {"torch_dtype": torch.bfloat16}


def load_base_model(model_name: str, bf16: bool = False):
    kwargs = _dtype_kwargs(bf16)

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
