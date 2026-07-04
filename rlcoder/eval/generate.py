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
    max_tokens: int = 4096,
    lora_path: Optional[str] = None,
    max_model_len: int = 8192,
    gpu_mem_util: float = 0.90,
    max_lora_rank: int = 32,
    seed: int = 0,
) -> List[List[str]]:
    # vLLM 0.24's FlashInfer sampler path mis-detects RTX 5090/sm120 and can
    # also crash when flashinfer is absent. Triton/native sampling is slower but
    # reliable for eval and probing.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

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
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    lora_req = None
    if lora_path:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, lora_path)

    outs = llm.generate(prompts, sp, lora_request=lora_req)
    return [[c.text for c in o.outputs] for o in outs]
