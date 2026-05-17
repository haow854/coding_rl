import asyncio
import json
import re
import sys
import base64
import random
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
from art import TrainableModel, TrajectoryGroup, Trajectory
from art.local.backend import LocalBackend
from art.dev import InternalModelConfig, InitArgs, EngineArgs
import art

from test_optimizer import optimizer_evaluation
from eval_optimizer import eval_debugger
from agents.planner_mbpp import Planner
from agents.coder_mbpp import Coder
from query_full import load_coding_problem
from config import TRAINED_PLANNER_ID, TRAINED_CODER_ID
PLANNER_MODEL_ID = TRAINED_PLANNER_ID   # frozen
CODER_MODEL_ID   = TRAINED_CODER_ID     # frozen



# ═══════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════
BATCH_SIZE = 15
ROLLOUTS = 8
MAX_STEPS = 500
LEARNING_RATE = 5e-6
REGEN_INTERVAL = 48
EVAL_INTERVAL = 24
CODER_SAMPLES = 8
CODE_SEM = asyncio.Semaphore(16)


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


def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


def extract_plan(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


def parse_verdict(text: str) -> str:
    text_upper = text.strip().upper()
    match = re.search(r'VERDICT\s*:\s*(CORRECT|BUGGY)', text_upper)
    if match:
        return match.group(1)
    if text_upper.startswith("CORRECT") or "VERDICT: CORRECT" in text_upper:
        return "CORRECT"
    if text_upper.startswith("BUGGY") or "VERDICT: BUGGY" in text_upper:
        return "BUGGY"
    if re.search(r'^Bug:', text, re.MULTILINE):
        return "BUGGY"
    return "UNKNOWN"


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
        except:
            pass
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
            return True, int(parts[2]), int(parts[3]), False
        else:
            return False, 0, 0, False
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except:
            pass
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


def calculate_reward(
    actually_correct: bool,
    verdict: str,
    after_pass: bool,
    after_partial: float,
    before_partial: float,
):
    reward = 0.0

    if verdict == "UNKNOWN":
        reward += -0.1
    elif actually_correct and verdict == "CORRECT":
        reward += 0.3
    elif actually_correct and verdict == "BUGGY":
        reward += -0.3
    elif not actually_correct and verdict == "BUGGY":
        reward += 0.3
    elif not actually_correct and verdict == "CORRECT":
        reward += -0.3

    if actually_correct:
        if after_pass:
            reward += 0.1
        else:
            reward += -0.5
    else:
        if after_pass:
            reward += 1.0
        else:
            delta = after_partial - before_partial
            if delta > 0:
                reward += round(min(0.5, delta * 1.5), 2)
            elif delta < 0:
                reward += round(max(-0.5, delta * 1.0), 2)
            else:
                reward += -0.05

    return round(reward, 2)


# ═══════════════════════════════════════════════════
#  CODE POOL
# ═══════════════════════════════════════════════════
async def _single_coder_attempt(planner, coder, scenario):
    try:
        plan_raw = await planner.plan(scenario.prompt, scenario.test_list[0])
        plan_text = extract_plan(plan_raw)
        code_raw = await coder.code_no_incre(plan_text, scenario)
        code_text = extract_python_code(code_raw)
    except Exception:
        return None

    runs, full_pass, partial = await check_pass_partial(code_text, scenario.test_list)
    return {
        "code": code_text,
        "runs": runs,
        "full_pass": full_pass,
        "partial": partial,
    }


async def fill_scenario_codes(planner, coder, scenario):
    results = await asyncio.gather(
        *[_single_coder_attempt(planner, coder, scenario)
          for _ in range(CODER_SAMPLES)]
    )
    results = [r for r in results if r is not None]

    scenario.passing_codes = []
    scenario.buggy_codes = []

    for r in results:
        if r["full_pass"]:
            scenario.passing_codes.append(r["code"])
        else:
            scenario.buggy_codes.append({
                "code": r["code"],
                "partial": r["partial"],
            })


async def refresh_all_code(planner, coder, scenarios):
    await asyncio.gather(
        *[fill_scenario_codes(planner, coder, s) for s in scenarios]
    )

    has_buggy = sum(1 for s in scenarios if s.buggy_codes)
    has_pass = sum(1 for s in scenarios if s.passing_codes)
    has_both = sum(1 for s in scenarios if s.buggy_codes and s.passing_codes)
    only_pass = sum(1 for s in scenarios if s.passing_codes and not s.buggy_codes)
    only_buggy = sum(1 for s in scenarios if s.buggy_codes and not s.passing_codes)
    neither = sum(1 for s in scenarios
                  if not s.buggy_codes and not s.passing_codes)

    print(f"[pool] total={len(scenarios)}")
    print(f"both={has_both} "
          f"only_pass={only_pass} only_buggy={only_buggy} neither={neither}")
    print(f"trainable: {has_buggy + only_pass}/{len(scenarios)} "
          f"({(has_buggy + only_pass)/len(scenarios):.0%})")


def select_batch(scenarios, batch_size):
    has_buggy = [s for s in scenarios if s.buggy_codes]
    only_pass = [s for s in scenarios if s.passing_codes and not s.buggy_codes]
    
    random.shuffle(has_buggy)
    random.shuffle(only_pass)

    n_buggy = min(len(has_buggy), int(batch_size * 0.8))
    n_pass = min(len(only_pass), batch_size - n_buggy)
    
    return has_buggy[:n_buggy] + only_pass[:n_pass]


def pick_input(scenario):
    has_buggy = bool(scenario.buggy_codes)
    has_pass = bool(scenario.passing_codes)

    if has_buggy and has_pass:
        if random.random() < 0.7:
            chosen = random.choice(scenario.buggy_codes)
            return {"code": chosen["code"], "partial": chosen["partial"],
                    "actually_correct": False}
        else:
            return {"code": random.choice(scenario.passing_codes),
                    "partial": 1.0, "actually_correct": True}
    elif has_buggy:
        chosen = random.choice(scenario.buggy_codes)
        return {"code": chosen["code"], "partial": chosen["partial"],
                "actually_correct": False}
    elif has_pass:
        return {"code": random.choice(scenario.passing_codes),
                "partial": 1.0, "actually_correct": True}
    else:
        return None


# ═══════════════════════════════════════════════════
#  ROLLOUT
# ═══════════════════════════════════════════════════
async def rollout_optimizer(optimizer_model, scenario, chosen_input):

    if chosen_input is None:
        traj = Trajectory(
            messages_and_choices=[
                {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": "No code available."},
            ],
            reward=0.0,
            metadata={"task_id": scenario.task_id},
        )
        traj.metrics = {
            "actually_correct": 0,
            "verdict_correct": 0,
            "after_pass": 0,
            "fix_success": 0,
            "regression": 0,
            "valid": 0,
        }
        return traj

    chosen_code = chosen_input["code"]
    before_partial = chosen_input["partial"]
    actually_correct = chosen_input["actually_correct"]

    user_prompt = (
        f"Problem: {scenario.prompt}\n\n"
        f"Testcase: {scenario.test_list[0]}\n\n"
        f"Code:\n```python\n{chosen_code}\n```"
    )

    traj = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        reward=0.0,
        metadata={"task_id": scenario.task_id},
    )

    try:
        client = optimizer_model.openai_client()
        chat_completion = await client.chat.completions.create(
            messages=traj.messages_and_choices,
            model=optimizer_model.get_inference_name(),
            temperature=0.9,
            top_p=0.9,
            timeout=60,
        )
        choice = chat_completion.choices[0]
        traj.messages_and_choices.append(choice)
        raw_output = choice.message.content.strip()
    except Exception as e:
        print(f"[opt fail] {scenario.task_id}: {type(e).__name__}: {e}")
        traj.reward = -0.2
        traj.metrics = {
            "actually_correct": int(actually_correct),
            "verdict_correct": 0,
            "after_pass": 0,
            "fix_success": 0,
            "regression": 0,
            "valid": 0,
        }
        return traj

    verdict = parse_verdict(raw_output)
    fixed_code = extract_python_code(raw_output)

    _, after_pass, after_partial = await check_pass_partial(
        fixed_code, scenario.test_list
    )

    reward = calculate_reward(
        actually_correct=actually_correct,
        verdict=verdict,
        after_pass=after_pass,
        after_partial=after_partial,
        before_partial=before_partial,
    )

    if actually_correct:
        verdict_correct = (verdict == "CORRECT")
    else:
        verdict_correct = (verdict == "BUGGY")

    traj.reward = reward
    traj.metrics = {
        "actually_correct": int(actually_correct),
        "verdict_correct": int(verdict_correct),
        "after_pass": int(after_pass),
        "fix_success": int(not actually_correct and after_pass),
        "regression": int(actually_correct and not after_pass),
        "valid": 1,
    }
    return traj


def log_step_summary(step, train_groups, batch):
    total_fix = 0
    total_reg = 0
    total_verdict_correct = 0
    total_verdict_count = 0
    trainable_count = 0

    for gi, group in enumerate(train_groups):
        rewards = [t.reward for t in group]
        std = float(np.std(rewards))
        if std > 0.001:
            trainable_count += 1
        print(f"[group {gi}] rewards={rewards} std={std:.3f}")

        for t in group:
            m = t.metrics
            total_fix += m.get("fix_success", 0)
            total_reg += m.get("regression", 0)
            if m.get("valid", 0) == 1:
                total_verdict_count += 1
                total_verdict_correct += m.get("verdict_correct", 0)

    verdict_acc = (total_verdict_correct / total_verdict_count * 100
                   if total_verdict_count > 0 else 0)

    print(f"[step {step}] "
          f"trainable={trainable_count}/{len(train_groups)} "
          f"fix={total_fix} reg={total_reg} "
          f"verdict_acc={verdict_acc:.0f}% "
          f"({total_verdict_correct}/{total_verdict_count})")

    return {
        "trainable": trainable_count,
        "fix": total_fix,
        "reg": total_reg,
        "verdict_acc": verdict_acc,
    }


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
async def train():
    # ─── frozen planner ───
    planner_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=PLANNER_MODEL_ID,
        gpu_memory_utilization=0.25,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    ))
    planner_tokenizer = AutoTokenizer.from_pretrained(
        PLANNER_MODEL_ID
    )

    # ─── frozen coder ───
    coder_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=CODER_MODEL_ID,
        gpu_memory_utilization=0.25,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    ))
    coder_tokenizer = AutoTokenizer.from_pretrained(CODER_MODEL_ID)

    # ─── optimizer (trainable) ───
    backend = LocalBackend(path='./optimizer_7B')
    optimizer_model = TrainableModel(
        name="v2.0",
        project="code_optimizer",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config=InternalModelConfig(
            init_args=InitArgs(max_seq_length=8192),
            engine_args=EngineArgs(
                max_num_seqs=32,
                gpu_memory_utilization=0.25,
            ),
        ),
    )
    await optimizer_model.register(backend)

    planner = Planner(planner_engine, planner_tokenizer)
    coder = Coder(coder_engine, coder_tokenizer)

    scenarios = load_coding_problem()
    print(f"Loaded {len(scenarios)} scenarios")

    step = await optimizer_model.get_step()

    print(f"\n{'='*60}")
    print(f"  Building code pool ({CODER_SAMPLES} samples per problem)")
    print(f"{'='*60}")
    await refresh_all_code(planner, coder, scenarios)

    while step < MAX_STEPS:
        if step > 0 and step % REGEN_INTERVAL == 0:
            print(f"\n=== Refreshing code pool at step {step} ===")
            await refresh_all_code(planner, coder, scenarios)

        # ── eval ──
        if step % EVAL_INTERVAL == 0:
            try:
                stats = await optimizer_evaluation(
                    planner_engine, planner_tokenizer,
                    coder_engine, coder_tokenizer,
                    optimizer_model,
                )
                line = (
                    f"step:{step}, "
                    f"before_run:{stats['before_run_rate']:.2%}, "
                    f"before_pass:{stats['before_pass_rate']:.2%}, "
                    f"after_run:{stats['after_run_rate']:.2%}, "
                    f"after_pass:{stats['after_pass_rate']:.2%}, "
                    f"fix:{stats['fix_count']}, reg:{stats['regression_count']}"
                )
                with open("debugger_test_eval.txt", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(f"[test-eval] {line}")
            except Exception as e:
                print(f"debugger_evaluation: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

            try:
                output_file = await eval_debugger(
                    optimizer_model, planner_engine, planner_tokenizer,
                    coder_engine, coder_tokenizer, step
                )
                print(f"[eval] step={step} saved to {output_file}")
            except Exception as e:
                print(f"Eval error: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

        batch = select_batch(scenarios, BATCH_SIZE)

        n_has_both = sum(1 for s in batch if s.buggy_codes and s.passing_codes)
        n_only_buggy = sum(1 for s in batch if s.buggy_codes and not s.passing_codes)
        n_only_pass = sum(1 for s in batch if s.passing_codes and not s.buggy_codes)
        print(f"\n{'='*60}")
        print(f"  STEP {step} | batch={len(batch)} "
              f"(both={n_has_both} buggy_only={n_only_buggy} pass_only={n_only_pass})")
        print(f"{'='*60}")

        # ── rollouts ──
        train_groups = await art.gather_trajectory_groups(
            (
                TrajectoryGroup(
                    rollout_optimizer(optimizer_model, s, chosen_input)
                    for _ in range(ROLLOUTS)
                )
                for s, chosen_input in [
                    (s, pick_input(s)) for s in batch
                ]
            ),
            pbar_desc="gather",
        )

        summary = log_step_summary(step, train_groups, batch)

        result = await backend.train(
            optimizer_model,
            train_groups,
            learning_rate=LEARNING_RATE,
        )
        await optimizer_model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split='train',
        )

        print(f"✓ step {step} done")
        step = await optimizer_model.get_step()


if __name__ == "__main__":
    import psutil
    for proc in psutil.process_iter(['pid', 'name']):
        if 'model-service' in proc.info['name'] or 'vllm' in proc.info['name']:
            proc.kill()
    asyncio.run(train())