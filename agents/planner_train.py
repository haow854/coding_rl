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


# Planner System Prompt
PLANNER_SYSTEM_PROMPT = """
You are an coding planner for programming problems.

You are given:
1) A full problem description.
2) Sample testcase.

Your task:
- Produce a precise coding plan that a coder model can implement directly.

OUTPUT RULES:
- Provide 1 to 4 steps.
- Output ONLY a JSON list of strings.
- Each string must follow the format: "step X: ..."
- No markdown, no code blocks, no explanations.

Example:
["step 1: ...", "step 2: ..."]


"""

# Planner

class Planner:
    def __init__(self, model):
        self.model = model
        print("planner initialized")

    async def plan(self, problem: str, testcase: str) -> str:
        planner = self.model.openai_client()

        user_prompt = f"The problem is as follows.\n{problem}\n\nSample testcase:\n{testcase}"

        messages = [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

        try:
            chat_completion = await planner.chat.completions.create(
                messages=messages,
                model=self.model.get_inference_name(),
                timeout=30,
                temperature=0.9,
                top_p=0.9,
                logprobs=True,
                # temperature=0.0,
                # top_p=1.0,
            )
            print("done")
        except Exception as e:
            print(f"plan generation failed，error: {e}")
            return None

        plan_raw = chat_completion.choices[0]
        return plan_raw
