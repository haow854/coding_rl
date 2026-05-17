from evalplus.data import get_mbpp_plus
import asyncio
import json
import re
from agents.planner_mbpp import Planner
from agents.coder_mbpp import Coder
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer
from config import TRAINED_PLANNER_ID, TRAINED_CODER_ID, TRAINED_OPT_ID, HF_TOKEN

PLANNER_HF_MODEL_ID = TRAINED_PLANNER_ID
CODER_HF_MODEL_ID = TRAINED_CODER_ID
HF_TOKEN = HF_TOKEN


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

import ast

def clean_code(code: str) -> str:
    try:
        tree = ast.parse(code)

        class RemoveAsserts(ast.NodeTransformer):
            def visit_Assert(self, node):
                return None

        tree = RemoveAsserts().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except SyntaxError:
        lines = []
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith("assert"):
                continue
            lines.append(line)
        return "\n".join(lines)


async def single_pipeline(planner, coder, s):
    try:
        plan_raw = await planner.plan(s.prompt, s.test_list[1])
        plan_text = extract_plan(plan_raw)

        code_raw = await coder.code_no_incre(plan_text, s)
        code_text = extract_python_code(code_raw)
        code_text = clean_code(code_text)

        return {"task_id": s.task_id, "solution": code_text}
    except Exception as e:
        print(f"Error on {s.task_id}: {e}")
        return None

async def evaluate_with_resampling(planner, coder, s, n=10):
    try:
        tasks = [single_pipeline(planner, coder, s) for _ in range(n)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        solutions = []
        for res in results:
            if isinstance(res, Exception):
                print(f"Sample error on {s.task_id}: {res}")
                continue
            if res is not None:
                solutions.append(res)

        if not solutions:
            return None

        return solutions

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
    planner_tokenizer = AutoTokenizer.from_pretrained(PLANNER_HF_MODEL_ID, token=HF_TOKEN)

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

    print("Loading MBPP dataset...")
    raw_dataset = get_mbpp_plus()
    scenarios = [Scenario(tid, row["prompt"], row["assertion"]) for tid, row in raw_dataset.items()]
    print(f"Loaded {len(scenarios)} problems.")

    n = 10
    tasks = [evaluate_with_resampling(planner, coder, s, n=n) for s in scenarios]
    results = await asyncio.gather(*tasks)

    successful_data = []
    for res in results:
        if res is not None:
            successful_data.extend(res)

    output_file = "planner_coder_mbpp_pass10.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in successful_data:
            entry["solution"] = entry["solution"].replace('\xa0', ' ')
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(successful_data)} solutions to {output_file}")
    print(f"python -m evalplus.evaluate --dataset mbpp --samples {output_file} --pass-k {n}")


if __name__ == "__main__":
    asyncio.run(main())