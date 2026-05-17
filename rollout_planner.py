import asyncio
import json
import re
import nest_asyncio
from typing import Dict
from dataclasses import dataclass
from agents.planner_train import Planner, PLANNER_SYSTEM_PROMPT
from art import Trajectory
import art
from art.dev import InternalModelConfig, InitArgs, EngineArgs
from art.utils import limit_concurrency
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import sys
import base64

CODE_SEM = asyncio.Semaphore(16)

async def execute_with_timeout_async(code_text, test_list, timeout=10.0):
    payload = {"code": code_text, "tests": test_list}
    payload_b64 = base64.b64encode(json.dumps(payload).encode()).decode()

    runner_script = f"""
import base64, json, sys
data = json.loads(base64.b64decode('{payload_b64}').decode())
code, tests = data['code'], data['tests']
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
    code_runs = True
    
    passed_count = 0
    total_tests = len(tests)
    
    if total_tests > 0:
        for t in tests:
            try:
                exec(t, _env, _env)
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
    length: int = 0
    code_runs: bool = False
    passed_tests: int = 0
    total_tests: int = 0
    time_out: bool = False

    def to_metrics(self) -> Dict[str, int]:
        return {
            "length": int(self.length),
            "code_runs": int(self.code_runs),
            "pass_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            "time_out": int(self.time_out),
        }

def calculate_reward(rubric: CodingRubric) -> float:
    if rubric.time_out:
        return -2.0
    
    reward = 0.0
    pass_rate = rubric.passed_tests / rubric.total_tests if rubric.total_tests > 0 else 0.0

    if rubric.length == 0:
        return -1.5
    if rubric.length == 4 or rubric.length == 1:
        struct_reward = 0.0
    elif 2 <= rubric.length <= 3:
        struct_reward = 0.4
    else:
        struct_reward = -0.2 * (rubric.length - 4)

    if rubric.code_runs:
        execution_bonus = 0.1
        accuracy_bonus = pass_rate * 1.0
        completion_bonus = 0.5 if pass_rate == 1.0 else 0.0
        reward = struct_reward + execution_bonus + accuracy_bonus + completion_bonus
    else:
        reward = struct_reward - 1.0 

    return round(reward, 2)

#----------------------extract------------------------------
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
#-----------------------------------------------------------


async def rollout_no_incre(
    planner: art.Model,
    coder_fixed,
    s
) -> Trajectory:

    rubric = CodingRubric()
    user_prompt = f"The problem is as follows.\n{s.prompt}\n\nSample testcase:\n{s.test_list[0]}"
    traj = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        reward=0.0,
        metadata={"task_id": s.task_id},
    )

    try:
        plan_raw = await planner.plan(s.prompt, s.test_list[0])
        if not plan_raw:
            raise ValueError("Plan generation failed")
        traj.messages_and_choices.append(plan_raw)
        plan_text = extract_plan(plan_raw.message.content.strip())
        plan_list = json.loads(plan_text)
        rubric.length = len(plan_list)
    except Exception:
        traj.reward = calculate_reward(rubric)
        traj.metrics = rubric.to_metrics()
        return traj

    try:
        code_text_raw = await coder_fixed.code_no_incre(plan_text, s)
        code_text = extract_python_code(code_text_raw)
    except Exception:
        traj.reward = calculate_reward(rubric)
        traj.metrics = rubric.to_metrics()
        return traj

    try:
        async with CODE_SEM:
            run, passed, total, is_timeout = await execute_with_timeout_async(
                code_text, s.test_list, timeout=10.0
            )
        
        rubric.code_runs = run
        rubric.passed_tests = passed
        rubric.total_tests = total
        rubric.time_out = is_timeout

        if is_timeout:
            print(f"Task {s.task_id} KILLED.")

    except Exception as e:
        print(f"Verify Error: {e}")

    traj.reward = calculate_reward(rubric)
    traj.metrics = rubric.to_metrics()
    return traj
