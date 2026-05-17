import asyncio
import json
import re
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer
from evalplus.data import get_mbpp_plus

from agents.planner_mbpp import Planner
from agents.coder_mbpp import Coder
from agents.optimizer_mbpp import Optimizer

from config import TRAINED_PLANNER_ID, TRAINED_CODER_ID, TRAINED_OPT_ID, HF_TOKEN

PLANNER_HF_MODEL_ID = TRAINED_PLANNER_ID
CODER_HF_MODEL_ID = TRAINED_CODER_ID
OPTIMIZER_HF_MODEL_ID = TRAINED_OPT_ID
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

async def single_pipeline(planner, coder, optimizer, s):
    try:
        plan_raw = await planner.plan(s.prompt, s.test_list[1])
        plan_text = extract_plan(plan_raw)

        code_raw = await coder.code_no_incre(plan_text, s)
        code_text = extract_python_code(code_raw)
        code_text = clean_code(code_text)

        fixed_raw = await optimizer.debug(s.prompt, s.test_list[1], code_text)
        fixed_code = extract_python_code(fixed_raw)

        return {"task_id": s.task_id, "solution": fixed_code}
    except Exception as e:
        print(f"Error on {s.task_id}: {e}")
        return None


async def evaluate_with_resampling(planner, coder, optimizer, s, n=10):
    tasks = [single_pipeline(planner, coder, optimizer, s) for _ in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    solutions = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        solutions.append(r)

    return solutions if solutions else None


async def eval_debugger_resampling(
    optimizer_engine, optimizer_tokenizer,
    planner_engine, planner_tokenizer,
    coder_engine, coder_tokenizer,
    step: int, n: int = 10,
):
    planner = Planner(planner_engine, planner_tokenizer)
    coder = Coder(coder_engine, coder_tokenizer)
    optimizer = Optimizer(optimizer_engine, optimizer_tokenizer)

    print("Loading MBPP dataset...")
    raw_dataset = get_mbpp_plus()
    scenarios = [Scenario(tid, row["prompt"], row["assertion"]) for tid, row in raw_dataset.items()]
    print(f"Loaded {len(scenarios)} problems.")

    tasks = [evaluate_with_resampling(planner, coder, optimizer, s, n=n) for s in scenarios]
    results = await asyncio.gather(*tasks)

    successful_data = []
    for res_list in results:
        if res_list is not None:
            successful_data.extend(res_list)

    output_file = f"all{n}_step{step}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in successful_data:
            entry["solution"] = entry["solution"].replace('\xa0', ' ')
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(successful_data)} solutions to {output_file}")
    print(f"python -m evalplus.evaluate --dataset mbpp --samples {output_file} --pass-k {n}")
    return output_file


if __name__ == "__main__":
    async def main():
        planner_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=PLANNER_HF_MODEL_ID,
            gpu_memory_utilization=0.25,
            max_model_len=8192,
            dtype="bfloat16",
            trust_remote_code=True,
        ))
        planner_tokenizer = AutoTokenizer.from_pretrained(
            PLANNER_HF_MODEL_ID, token=HF_TOKEN
        )

        coder_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=CODER_HF_MODEL_ID,
            gpu_memory_utilization=0.25,
            max_model_len=8192,
            dtype="bfloat16",
            trust_remote_code=True,
        ))
        coder_tokenizer = AutoTokenizer.from_pretrained(CODER_HF_MODEL_ID)

        optimizer_engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=OPTIMIZER_HF_MODEL_ID,
            gpu_memory_utilization=0.25,
            max_model_len=16384,
            dtype="bfloat16",
            trust_remote_code=True,
        ))
        optimizer_tokenizer = AutoTokenizer.from_pretrained(OPTIMIZER_HF_MODEL_ID)

        await eval_debugger_resampling(
            optimizer_engine, optimizer_tokenizer,
            planner_engine, planner_tokenizer,
            coder_engine, coder_tokenizer,
            step=1,
            n=10,
        )

    asyncio.run(main())