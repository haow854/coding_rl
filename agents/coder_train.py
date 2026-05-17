import asyncio
import json
import art
from art import TrainableModel, gather_trajectory_groups
from art.local.backend import LocalBackend
from art.utils import limit_concurrency
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel
from datasets import load_dataset
import re
import multiprocessing as mp
from art.dev import InternalModelConfig, InitArgs, EngineArgs


CODER_SYSTEM_PROMPT = """
You are a Python coding assistant specializing in incremental development.

You are given:
1) A full problem description.
2) Sample testcase.
3) The current state of the Python code.
4) The NEXT SPECIFIC STEP to implement.

Your task:
- Extend the "Current Code" by implementing the "Next Step",
  but respect the requirements of BOTH original problem description and sample testcase.

OUTPUT RULES:
- ONLY output code. NO explanations, markdown notes, or comments.
- The function name and signature MUST match the testcase exactly.
"""

CODER_SYSTEM_PROMPT_NO_INCRE = """
You are a Python coding expert.
"""


class Coder:
    def __init__(self, model):
        self.model = model
        print("coder initialized")

    async def code(self, problem, test, steps: list[str], traj):
        coder = self.model.openai_client()
        code = ""
        for step in steps:
            user_prompt = (
                f"The problem is as follows.\n{problem}\n\n"
                f"Sample testcase: {test}\n\n"
                f"Current code: {code}"
                f"Next step: {step}"
            )
            
            traj.messages_and_choices.append({
            "role": "user", 
            "content": user_prompt
            })

            messages = [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            chat_completion = await coder.chat.completions.create(
                messages=messages,
                model=self.model.get_inference_name(),
                timeout=30,
                temperature=0.9,
                top_p=0.9,
                # temperature=0.0,
                # top_p=1.0,
            )
            choice = chat_completion.choices[0]
            traj.messages_and_choices.append(choice)
            code = choice.message.content.strip()

        return code, traj


    async def code_test(self, problem, test, steps: list[str]):
        coder = self.model.openai_client()
        code = ""
        for step in steps:
            user_prompt = (
                f"The problem is as follows.\n{problem}\n\n"
                f"Sample testcase: {test}\n\n"
                f"Current code: {code}"
                f"Next step: {step}"
            )

            messages = [
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            chat_completion = await coder.chat.completions.create(
                messages=messages,
                model=self.model.get_inference_name(),
                temperature=0.9,
                top_p=0.9,
                # temperature=0.0,
                # top_p=1.0,
            )
            choice = chat_completion.choices[0]
            code = choice.message.content.strip()

        return code

    async def code_ni(self, problem, test, steps):
        coder = self.model.openai_client()
      
        user_prompt = (
            f"\n{problem}\n\n"
            f"{test}\n\n"
            f"Here's an implementation plan for your reference: {steps}"
        )

        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT_NO_INCRE},
            {"role": "user", "content": user_prompt}
        ]

        chat_completion = await coder.chat.completions.create(
            messages=messages,
            model=self.model.get_inference_name(),
            timeout = 60,
            max_tokens=8192,
            temperature=0.9,
            top_p=0.9,
            # temperature=0.0,
            # top_p=1.0,
        )
        choice = chat_completion.choices[0]
        code = choice.message.content.strip()
        tokens = chat_completion.usage.completion_tokens
        print("done")

        return code, choice, tokens
