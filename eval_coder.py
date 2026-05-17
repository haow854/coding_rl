from evalplus.data import get_mbpp_plus
import asyncio
import json
import re
import nest_asyncio
from agents.planner_mbpp import Planner
from agents.coder_train import Coder
from art import TrainableModel
from art.local.backend import LocalBackend
from openai import AsyncOpenAI
from art.dev import InternalModelConfig, InitArgs, EngineArgs
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

class Scenario:
    def __init__(self, task_id, prompt, test_list):
        self.task_id = task_id
        self.prompt = prompt
        self.test_list = test_list

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


async def evaluate_with_resampling(planner, coder, s):

    try:
        plan_raw = await planner.plan(s.prompt, s.test_list[0]) 
        plan_text = extract_plan(plan_raw)

        code_text, choice,tokens = await coder.code_ni(s.prompt, s.test_list[0], plan_text)
        code_text = extract_python_code(code_text)

        prefix = '''
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

        full_solution = prefix + code_text

        return {
            "task_id": s.task_id,
            "solution": full_solution
        }

    except Exception as e:
        return None

async def eval_coder(coder_model, engine, tokenizer, step):
    print("Loading MBPP dataset...")
    raw_dataset = get_mbpp_plus() 

    scenarios = [Scenario(task_id, row["prompt"], row["assertion"]) 
        for task_id, row in raw_dataset.items()]
        
    total = len(scenarios)
    print(f"{total} problems in total.")

    coder = Coder(coder_model)
    planner = Planner(engine, tokenizer)

    tasks = [evaluate_with_resampling(planner, coder, s) for s in scenarios]
    results = await asyncio.gather(*tasks)

    successful_data = [res for res in results if res is not None]

    output_file = f"qwen_coder_mbpp_{step}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in successful_data:
            if "solution" in entry:
                entry["solution"] = entry["solution"].replace('\xa0', ' ')
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{len(successful_data)} EvalPlus data in total")
    print(f"Saved to: {output_file}")
    print(f"python -m evalplus.evaluate --dataset mbpp --samples {output_file}")
    
    return output_file
