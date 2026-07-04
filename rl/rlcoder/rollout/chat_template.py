"""Compatibility wrapper; chat-template helpers live in rlcoder.prompting."""
from rlcoder.prompting import (
    QWEN_CHATML_TEMPLATE,
    load_processing_class,
    render_chat_prompt,
)

__all__ = ["QWEN_CHATML_TEMPLATE", "load_processing_class", "render_chat_prompt"]
