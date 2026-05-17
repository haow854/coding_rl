import asyncio
from agents.planner_mbpp import Planner
from agents.coder_mbpp import Coder
from query import load_test
import base64
import json
import re
import sys

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
import cmath, re, heapq, collections, sys
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
    passed_count = 0
    total_tests = len(tests)
    for t in tests:
        try:
            exec(t, _env, _env)
            passed_count += 1
        except: pass
    print(f"RESULT:OK:{{passed_count}}:{{total_tests}}")
except Exception as e:
    print("RESULT:RUN_FAIL:0:0")
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-c', runner_script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode().strip()

        result_line = None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("RESULT:"):
                result_line = line
        
        if result_line is None:
            return False, 0, 0, False
        
        parts = result_line.split(":")
        if parts[1] == "OK":
            passed = int(parts[2])
            total = int(parts[3])
            return True, passed, total, False
        else:
            return False, 0, 0, False
        
    except asyncio.TimeoutError:
        try: proc.kill()
        except: pass
        return False, 0, 0, True

async def check_pass_partial(code: str, test_list: list):
    async with CODE_SEM:
        runs, passed, total, timed_out = await execute_with_timeout_async(
            code, test_list, timeout=10.0
        )
    if timed_out or not runs:
        return False, False, 0.0
    full = (passed == total and total > 0)
    partial = passed / total if total > 0 else 0.0
    return True, full, partial

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

class DebuggerART:
    def __init__(self, model):
        self.model = model

    async def debug(self, problem, testcase, code):
        user_prompt = (
            f"Problem: {problem}\n\n"
            f"Testcase: {testcase}\n\n"
            f"Code:\n```python\n{code}\n```"
        )
        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        client = self.model.openai_client()
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=self.model.get_inference_name(),
            temperature=0.0,
            top_p=1.0,
            timeout=60,
        )
        return chat_completion.choices[0].message.content.strip()


async def evaluate_one(planner, coder, debugger, s, sem):
    try:
        async with sem:
            plan_raw = await planner.plan(s.prompt, s.test_list[0])
            plan_text = extract_plan(plan_raw)

            code_raw = await coder.code_no_incre(plan_text, s)
            code_text = extract_python_code(code_raw)

        before_runs, before_pass, _ = await check_pass_partial(code_text, s.test_list)

        async with sem:
            fixed_raw = await debugger.debug(s.prompt, s.test_list[0], code_text)
            fixed_code = extract_python_code(fixed_raw)

        after_runs, after_pass, _ = await check_pass_partial(fixed_code, s.test_list)

        return {
            "task_id": s.task_id,
            "before_runs": before_runs,
            "before_pass": before_pass,
            "after_runs": after_runs,
            "after_pass": after_pass,
        }
    except Exception as e:
        print(f"[test] error on {s.task_id}: {type(e).__name__}: {e}")
        return None


async def optimizer_evaluation(
    planner_engine, planner_tokenizer,
    coder_engine, coder_tokenizer,
    debugger_model,
):
    scenarios = load_test()
    total = len(scenarios)

    planner = Planner(planner_engine, planner_tokenizer)
    coder = Coder(coder_engine, coder_tokenizer)
    debugger = DebuggerART(debugger_model)
    sem = asyncio.Semaphore(16)

    print(f"[test] start evaluating on {total} scenarios")
    tasks = [evaluate_one(planner, coder, debugger, s, sem) for s in scenarios]
    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    done = len(results)
    before_run_count = sum(1 for r in results if r["before_runs"])
    before_pass_count = sum(1 for r in results if r["before_pass"])
    after_run_count = sum(1 for r in results if r["after_runs"])
    after_pass_count = sum(1 for r in results if r["after_pass"])

    fix_count = sum(1 for r in results if (not r["before_pass"]) and r["after_pass"])
    reg_count = sum(1 for r in results if r["before_pass"] and (not r["after_pass"]))

    stats = {
        "total": total,
        "evaluated": done,
        "before_run_rate": before_run_count / total,
        "before_pass_rate": before_pass_count / total,
        "after_run_rate": after_run_count / total,
        "after_pass_rate": after_pass_count / total,
        "fix_count": fix_count,
        "regression_count": reg_count,
    }
    return stats