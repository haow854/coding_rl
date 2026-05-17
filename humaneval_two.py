import asyncio
import json
import re
from evalplus.data import get_human_eval_plus
from agents.planner_h import Planner
from agents.coder_h import Coder

from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer

from config import TRAINED_PLANNER_ID, TRAINED_CODER_ID, TRAINED_OPT_ID, HF_TOKEN

PLANNER_HF_MODEL_ID = TRAINED_PLANNER_ID
CODER_HF_MODEL_ID = TRAINED_CODER_ID
OPTIMIZER_HF_MODEL_ID = TRAINED_OPT_ID
HF_TOKEN = HF_TOKEN

PREFIX = '''
import math
import cmath
import re
import heapq
import collections
import sys
import string
import functools
import itertools
import random
import operator
import bisect
import numpy as np
from math import *
from typing import *
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import chain, groupby, combinations, permutations, combinations_with_replacement, product
from functools import reduce, lru_cache
from operator import itemgetter
from copy import deepcopy, copy
from array import array
'''


class Scenario:
    def __init__(self, task_id, prompt):
        self.task_id = task_id
        self.prompt = prompt


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


async def evaluate_with_resampling(planner, coder, s, n=10):
    try:
        async def single_sample():
            plan_raw = await planner.plan(s.prompt)
            plan_text = extract_plan(plan_raw)
            code_text = await coder.code_no_incre(plan_text, s)
            code_text = extract_python_code(code_text)
            return PREFIX + code_text

        tasks = [single_sample() for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        solutions = []
        for res in results:
            if isinstance(res, Exception):
                print(f"Error on {s.task_id}: {res}")
                continue
            solutions.append(res)

        if not solutions:
            return None

        return [{"task_id": s.task_id, "solution": sol} for sol in solutions]

    except Exception as e:
        print(f"Error on {s.task_id}: {e}")
        return None


async def main():
    planner_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=PLANNER_HF_MODEL_ID,
        gpu_memory_utilization=0.25,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    ))
    planner_tokenizer = AutoTokenizer.from_pretrained(PLANNER_HF_MODEL_ID)

    coder_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=CODER_HF_MODEL_ID,
        gpu_memory_utilization=0.50,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    ))
    coder_tokenizer = AutoTokenizer.from_pretrained(CODER_HF_MODEL_ID)

    planner = Planner(planner_engine, planner_tokenizer)
    coder = Coder(coder_engine, coder_tokenizer)

    print("Loading HumanEval+ dataset...")
    raw_dataset = get_human_eval_plus()

    scenarios = [
        Scenario(tid, row["prompt"])
        for tid, row in raw_dataset.items()
    ]

    print(f"Loaded {len(scenarios)} problems.")

    n = 10
    tasks = [evaluate_with_resampling(planner, coder, s, n=n) for s in scenarios]
    results = await asyncio.gather(*tasks)

    successful_data = []
    for res in results:
        if res is not None:
            successful_data.extend(res)

    output_file = "humaneval_pass1_2.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in successful_data:
            entry["solution"] = entry["solution"].replace('\xa0', ' ')
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(successful_data)} solutions to {output_file}")
    print(f"python -m evalplus.evaluate --dataset humaneval --samples {output_file} --pass-k {n}")


if __name__ == "__main__":
    asyncio.run(main())