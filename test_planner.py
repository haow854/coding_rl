import asyncio
import json
import re
import os
import sys
import base64
import nest_asyncio
from agents.planner_train import Planner
from agents.coder_mbpp import Coder
from art import TrainableModel
from art.local.backend import LocalBackend
from openai import AsyncOpenAI
from art.dev import InternalModelConfig, InitArgs, EngineArgs
from query import load_test, load_coding_problem
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer

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

async def execute_with_timeout_async(code_text, test_list, timeout=10.0):
    payload = {"code": code_text, "tests": test_list}
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

    runner_script = f"""
import base64
import json
import sys

payload_str = '{payload_b64}'
data = json.loads(base64.b64decode(payload_str).decode())
code = data['code']
tests = data['tests']

_env = {{"__builtins__": __builtins__}}
_setup = '''import math
from math import tan, pi, radians, sin, cos, acos
import cmath
import re
import heapq
import collections
import sys
from typing import *
from collections import Counter, defaultdict
from itertools import chain, groupby, combinations_with_replacement
from operator import itemgetter
from copy import deepcopy
from array import array
'''
try:
    exec(_setup, _env)
    exec(code, _env, _env)
    passed = False
    if len(tests) > 0:
        for t in tests:
            try:
                exec(t, _env, _env)
            except Exception:
                print("RUN_OK_TEST_FAIL")
                sys.exit(0)
        passed = True
    if passed:
        print("ALL_PASS")
    else:
        print("RUN_OK_TEST_FAIL")
except Exception as e:
    print("RUN_FAIL")
"""

    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-c', runner_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode().strip()
        
        if "ALL_PASS" in out:
            return True, True, False      
        elif "RUN_OK_TEST_FAIL" in out:
            return True, False, False     
        else:
            return False, False, False    
            
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return False, False, True


async def evaluate_planner(planner, coder, s, sem):
    try:
        print("generating plan")
        plan_raw = await planner.plan(s.prompt, s.test_list[0])
        plan_text = extract_plan(plan_raw.message.content.strip())
        
        print("generating code")
        code_text = await coder.code_no_incre(plan_text, s)
        code_text = extract_python_code(code_text)

        try:
            async with sem: 
                # 注意这里没有传入 verify_code 函数名了
                runs, passed, is_timeout = await execute_with_timeout_async(
                    code_text, s.test_list, timeout=10.0
                )
            
            if is_timeout:
                print(f"⏰ Task {s.task_id} timed out (Process Killed).")
                print(f"代码片段:\n{code_text[:150]}...")
                return False, False
                
            return runs, passed

        except Exception as e:
            print(f"Error executing: {e}")
            return False, False

    except Exception as e:
        return False, False

async def planner_evaluation(planner_model, engine, tokenizer):
    scenarios = load_test()
    total = len(scenarios)

    planner = Planner(planner_model)
    coder_fixed = Coder(engine, tokenizer)

    sem = asyncio.Semaphore(16)
    print(f"start testing，total: {total}")
    tasks = [evaluate_planner(planner, coder_fixed, s, sem) for s in scenarios]
    
    results = await asyncio.gather(*tasks)

    run_count = sum(1 for run, passed in results if run is True)
    passed_count = sum(1 for run, passed in results if passed is True)

    run_rate = run_count/total
    pass_rate = passed_count/ total
    return run_rate, pass_rate

