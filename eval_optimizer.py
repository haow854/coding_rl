import asyncio
import json
import re
import os
import sys
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
from evalplus.data import get_mbpp_plus
from art import TrainableModel
from art.local.backend import LocalBackend
from art.dev import InternalModelConfig, InitArgs, EngineArgs

from agents.planner_mbpp import Planner
from agents.coder_mbpp import Coder

PREFIX = """import math
from math import tan, pi, radians, sin, cos, acos
import cmath, re, heapq, collections, sys
from typing import *
from collections import Counter, defaultdict
from itertools import chain, groupby, combinations_with_replacement
from operator import itemgetter
from copy import deepcopy
from array import array
"""

OPTIMIZER_SYSTEM_PROMPT = """You are a Python code reviewer and optimizer.

Given a problem, a sample testcase, and a code attempt, you must:

Step 1: Judge whether the code is correct or buggy.
Step 2: If buggy, analyze the issue and provide a fixed version.
        If correct, return the code unchanged.

Format your response EXACTLY as:

If the code is correct:
Verdict: CORRECT
```python
<return the original code unchanged>
```

If the code is buggy:
Verdict: BUGGY
Bug: <brief description of the issue>
```python
<fixed code>
```

Rules:
- Keep the function name and signature exactly as in the testcase.
- Do NOT add test code or assertions.
- You MUST start with "Verdict: CORRECT" or "Verdict: BUGGY".
"""


class Scenario:
    def __init__(self, task_id, prompt, test_list):
        self.task_id = task_id
        self.prompt = prompt
        self.test_list = test_list


class DebuggerART:
    def __init__(self, model: TrainableModel):
        self.model = model

    async def debug(self, problem: str, testcase: str, code: str) -> str:
        user_prompt = (
            f"Problem: {problem}\n\n"
            f"Testcase: {testcase}\n\n"
            f"Code:\n```python\n{code}\n```"
        )
        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        client = self.model.openai_client()
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=self.model.get_inference_name(),
            temperature=0.0,
            top_p=1.0,
        )
        return chat_completion.choices[0].message.content.strip()


def extract_python_code(text: str) -> str:
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_plan(text: str) -> str:
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


SEM = asyncio.Semaphore(16)

async def evaluate_one(planner, coder, debugger, s):
    try:
        async with SEM:
            plan_raw = await planner.plan(s.prompt, s.test_list[0])
            plan_text = extract_plan(plan_raw)

            code_raw = await coder.code_no_incre(plan_text, s)
            code_text = extract_python_code(code_raw)

            fixed_raw = await debugger.debug(s.prompt, s.test_list[0], code_text)
            fixed_code = extract_python_code(fixed_raw)

        return {"task_id": s.task_id, "solution": PREFIX + fixed_code}

    except Exception as e:
        print(f"Error on {s.task_id}: {e}")
        return None


async def eval_debugger(debugger_model: TrainableModel, planner_engine, planner_tokenizer,
                        coder_engine, coder_tokenizer, step: int):
    planner = Planner(planner_engine, planner_tokenizer)
    coder = Coder(coder_engine, coder_tokenizer)
    debugger = DebuggerART(debugger_model)

    print("Loading MBPP dataset...")
    raw_dataset = get_mbpp_plus()
    scenarios = [Scenario(tid, row["prompt"], row["assertion"]) for tid, row in raw_dataset.items()]
    print(f"Loaded {len(scenarios)} problems.")

    results = await asyncio.gather(*[evaluate_one(planner, coder, debugger, s) for s in scenarios])
    results = [r for r in results if r is not None]

    output_file = f"debugger_mbpp_{step}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            r["solution"] = r["solution"].replace('\xa0', ' ')
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_file}")
    print(f"python -m evalplus.evaluate --dataset mbpp --samples {output_file}")
    return output_file