"""vLLM generation for eval / probing (GPU box). Supports a base model with an
optional LoRA adapter. Returns, per prompt, a list of `n` completion strings.
Heavy imports are deferred so the module stays importable on a CPU-only box."""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def generate(
    model: str,
    prompts_messages: List[List[Dict[str, str]]],
    n: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = -1,
    max_tokens: int = 4096,
    lora_path: Optional[str] = None,
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.90,
    max_lora_rank: int = 32,
    seed: int = 0,
) -> List[List[str]]:
    # FlashInfer's sampler crashes on Blackwell/sm120 (e.g. RTX 5090) but is MUCH
    # faster than the native Triton top-k/top-p kernel on A100/H100 (the native
    # path JIT-recompiles per shape and tanks batched throughput). Disable it
    # only on the broken arch; an explicit env override always wins.
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        try:
            import torch
            if torch.cuda.get_device_capability()[0] == 12:
                os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        except Exception:  # noqa: BLE001
            pass

    from vllm import LLM, SamplingParams
    from rlcoder.prompting import load_processing_class, render_chat_prompt

    proc = load_processing_class(model)
    prompts = [
        render_chat_prompt(proc, m)
        for m in prompts_messages
    ]

    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enable_lora=lora_path is not None,
        max_lora_rank=max_lora_rank,
        seed=seed,
    )
    sp_kwargs = dict(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if top_k > 0:
        sp_kwargs["top_k"] = top_k
    sp = SamplingParams(**sp_kwargs)

    lora_req = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, lora_path)

    outs = llm.generate(prompts, sp, lora_request=lora_req)
    return [[c.text for c in o.outputs] for o in outs]
