"""Prompt and chat-template helpers shared by SFT, GRPO, and evaluation.

These utilities are not RL rollout logic. They define the default text
interface used across all stages: problem statement in, one fenced Python
stdin/stdout program out. The first-stage route is deliberately non-thinking;
reasoning traces are an optional later ablation, not the default contract.
"""
from __future__ import annotations

from typing import Dict, List

from rlcoder.data.schema import Problem

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Read the problem and output "
    "one complete Python 3 program inside a single ```python code block. "
    "The program must read input from standard input (stdin) and write the "
    "answer to standard output (stdout), exactly matching the output format "
    "described in the problem. Do not include analysis or extra text."
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
        obj.chat_template = QWEN_CHATML_TEMPLATE
    return obj


def load_processing_class(model_name: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if hasattr(tok, "apply_chat_template"):
            return _ensure_template(tok)
    except Exception:  # noqa: BLE001
        pass

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_name)
    if hasattr(proc, "apply_chat_template"):
        return _ensure_template(proc)
    return proc


def render_chat_prompt(processing_class, messages: List[Dict[str, str]]) -> str:
    return processing_class.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
