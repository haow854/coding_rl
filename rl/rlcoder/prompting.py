"""Prompt and chat-template helpers shared by SFT, GRPO, and evaluation.

These utilities are not RL rollout logic. They define the default text
interface used across all stages: problem statement in, one fenced Python
stdin/stdout program out. The default route is thinking-first — the policy
reasons inside a <think>...</think> block (Qwen3 thinking mode), then emits the
final fenced program. rewards.extract_code() judges only that final block, so
the same interface serves SFT distillation, GRPO, and eval without divergence.
"""
from __future__ import annotations

from typing import Dict, List

from rlcoder.data.schema import Problem

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Reason about the problem, then "
    "give your final answer as one complete Python 3 program inside a single "
    "```python code block. The program must read input from standard input "
    "(stdin) and write the answer to standard output (stdout), exactly matching "
    "the output format described in the problem."
)

QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def build_messages(problem: Problem) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.statement},
    ]


def _has_template(obj) -> bool:
    return bool(getattr(obj, "chat_template", None))


def _ensure_template(obj):
    if not _has_template(obj):
        print("No chat template found, using default")
        obj.chat_template = QWEN_CHATML_TEMPLATE
    return obj


def load_processing_class(model_name: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if hasattr(tok, "apply_chat_template"):
            print("Using tokenizer with apply_chat_template")
            return _ensure_template(tok)
    except Exception:  # noqa: BLE001
        pass

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_name)
    if hasattr(proc, "apply_chat_template"):
        return _ensure_template(proc)
    return proc


def render_chat_prompt(
    processing_class,
    messages: List[Dict[str, str]],
    enable_thinking: bool = True,
) -> str:
    """Render [system, user] into a generation prompt string.

    Defaults to thinking-first (Qwen3): the prompt ends at the assistant tag and
    the model generates its own <think>...</think>. Kept explicit so SFT, GRPO
    probing, and eval all render prompts the same way.
    """
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    try:
        return processing_class.apply_chat_template(messages, **kwargs)
    except TypeError:
        print("Error applying chat template, using default")
        # Older tokenizers/processors may not accept Qwen's enable_thinking kwarg.
        kwargs.pop("enable_thinking", None)
        return processing_class.apply_chat_template(messages, **kwargs)
