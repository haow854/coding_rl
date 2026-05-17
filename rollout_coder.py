import asyncio, json, re, os, sys, base64
import art
from art import Trajectory
from dataclasses import dataclass
from typing import Dict
from agents.coder_train import Coder, CODER_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT_NO_INCRE
from query import PythonScenario
from art.local.backend import LocalBackend
from art.utils import limit_concurrency
import re

CODE_SEM = asyncio.Semaphore(16)

async def execute_with_timeout_async(code_text, test_list, timeout=10.0):
    payload = {"code": code_text, "tests": test_list}
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

    runner_script = f"""
import base64, json, sys
data = json.loads(base64.b64decode('{payload_b64}').decode())
code, tests = data['code'], data['tests']

_setup = '''
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

_env = {{}}
try:
    exec(_setup, _env)
    exec(code, _env)
    
    passed_count = 0
    total_tests = len(tests)
    
    if total_tests > 0:
        for t in tests:
            try:
                exec(t, _env)
                passed_count += 1
            except:
                pass
    print(f"RESULT:OK:{{passed_count}}:{{total_tests}}")
except Exception as e:
    print("RESULT:RUN_FAIL:0:0")
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-c', runner_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode().strip()
        
        if "RESULT:OK" in out:
            parts = out.split(":")
            passed = int(parts[2])
            total = int(parts[3])
            return True, passed, total, False
        else:
            return False, 0, 0, False
            
    except asyncio.TimeoutError:
        try: proc.kill()
        except: pass
        return False, 0, 0, True


@dataclass
class CodingRubric:
    code_runs: bool = False
    passed_tests: int = 0
    total_tests: int = 0
    time_out: bool = False
    token_count: int = 0

    def to_metrics(self) -> Dict[str, int]:
        return {
            "code_runs": int(self.code_runs),
            "pass_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            "time_out": int(self.time_out),
        }

def calculate_reward(rubric: CodingRubric) -> float:
    if rubric.time_out:
        return -2.0
    
    reward = 0.0
    pass_rate = rubric.passed_tests / rubric.total_tests if rubric.total_tests > 0 else 0.0

    if rubric.code_runs:
        execution_bonus = 0.1
        accuracy_bonus = pass_rate * 1.0
        completion_bonus = 0.5 if pass_rate == 1.0 else 0.0
        reward = execution_bonus + accuracy_bonus + completion_bonus
    else:
        reward = -1.0 

    return round(reward, 2)



#----------------------extract------------------------------
def extract_python_code(text: str) -> str:
    pattern = r"```(?:python)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()
#-----------------------------------------------------------



async def rollout_coder_ni(
    model: art.Model,
    scenario: PythonScenario
) -> Trajectory:

    rubric = CodingRubric()
    
    user_prompt = (
            f"\n{scenario.prompt}\n\n"
            f"{scenario.test_list[0]}\n\n"
            f"Here's an implementation plan for your reference: {scenario.plan}"
        )

    traj = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": CODER_SYSTEM_PROMPT_NO_INCRE},
            {"role": "user", "content": user_prompt}
        ],
        reward=0.0,
        metadata={"task_id": scenario.task_id},
    )

    coder = Coder(model)
    try:
        code_text, choice, tokens= await coder.code_ni(scenario.prompt, scenario.test_list[0], 
                          scenario.plan)
        rubric.token_count = tokens 
        traj.messages_and_choices.append(choice)
    except Exception as e:
        traj.reward = calculate_reward(rubric)
        traj.metrics = rubric.to_metrics()
        return traj
    
    print("generated code")
    code_text = extract_python_code(code_text)

    # verify code
    try:
        async with CODE_SEM:
            run, passed, total, is_timeout = await execute_with_timeout_async(
                code_text, scenario.test_list, timeout=10.0
            )
        
        rubric.code_runs = run
        rubric.passed_tests = passed
        rubric.total_tests = total
        rubric.time_out = is_timeout

        if is_timeout:
            print(f"Task {scenario.task_id} KILLED.")

    except Exception as e:
        print(f"Verify Error: {e}")

    traj.reward = calculate_reward(rubric)
    traj.metrics = rubric.to_metrics()
    return traj
    


