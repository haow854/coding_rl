import asyncio
import art
import numpy as np
from art import TrainableModel, TrajectoryGroup
from art.local.backend import LocalBackend
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from rollout_coder import rollout_coder_ni, execute_with_timeout_async, extract_python_code
from eval_coder import eval_coder
from test_coder import coder_evaluation
from query_full import load_coding_problem
from art.dev import InternalModelConfig, InitArgs, EngineArgs
from transformers import AutoTokenizer
from agents.planner_mbpp import Planner
from agents.coder_train import Coder
from collections import Counter
import re
from config import TRAINED_PLANNER_ID
HF_MODEL_ID = TRAINED_PLANNER_ID   # frozen planner


def extract_plan(text: str) -> str:
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


async def filter_by_difficulty(coder_model, scenarios, k=8, keep_lo=1, keep_hi=7):

    coder = Coder(coder_model)

    async def count_passes(s):
        async def one_try():
            try:
                code_text, _ = await coder.code_ni(s.prompt, s.test_list[0], s.plan)
                code_text = extract_python_code(code_text)
                run, passed, total, _ = await execute_with_timeout_async(
                    code_text, s.test_list, timeout=10.0
                )
                return run and total > 0 and passed == total
            except Exception as e:
                print(f"[filter] task {s.task_id} error: {type(e).__name__}: {e}")
                return False

        results = await asyncio.gather(*[one_try() for _ in range(k)])
        return sum(results)

    pass_counts = await asyncio.gather(*[count_passes(s) for s in scenarios])

    kept = [s for s, c in zip(scenarios, pass_counts) if keep_lo <= c <= keep_hi]

    dist = Counter(pass_counts)
    print(f"[filter] pass-count distribution (out of {k}): "
          f"{dict(sorted(dist.items()))}")
    print(f"[filter] kept {len(kept)} / {len(scenarios)} "
          f"(dropped {len(scenarios) - len(kept)})")
    return kept


async def train():
    # ---- planner ----
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=HF_MODEL_ID,
        gpu_memory_utilization=0.25,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True,
    ))
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    # ---- coder ----
    backend = LocalBackend(path='./coder_7B')
    coder_model = TrainableModel(
        name="v7.0",
        project="code_generation",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config=InternalModelConfig(
            init_args=InitArgs(max_seq_length=16384),
            engine_args=EngineArgs(max_num_seqs=32, gpu_memory_utilization=0.6),
        ),
    )
    await coder_model.register(backend)

    planner_fixed = Planner(engine, tokenizer)
    max_steps = 500
    batch_size = 15

    force_plan = True
    print("=== Building training pool ===")
    scenarios_pool = load_coding_problem()
    print(len(scenarios_pool))

    step = await coder_model.get_step()
    while step < max_steps:
        if step % 24 == 0 or force_plan:
            print(f"[replan] refreshing plans for {len(scenarios_pool)} scenarios")
            new_plans = await asyncio.gather(*[
                planner_fixed.plan(s.prompt, s.test_list[0]) for s in scenarios_pool
            ])
            for s, raw in zip(scenarios_pool, new_plans):
                new_plan = extract_plan(raw)
                if new_plan:
                    s.plan = new_plan
            force_plan = False

        if step % 24 == 0:
            try:
                run_rate, pass_rate = await coder_evaluation(engine, tokenizer, coder_model)
                with open("coder_7B_pass_rate.txt", "a", encoding="utf-8") as f:
                    f.write(f"step:{step}, run_rate:{run_rate:.2%}, "
                            f"pass_rate:{pass_rate:.2%}\n")
                print(f"[eval] step={step} run={run_rate:.2%} pass={pass_rate:.2%}")
            except Exception as e:
                print(f"coder_evaluation: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

            try:
                await eval_coder(coder_model, engine, tokenizer, step)
            except Exception as e:
                print(f"eval_coder: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

        n = len(scenarios_pool)
        start = (step * batch_size) % n
        end = start + batch_size
        if end <= n:
            batch_scenarios = scenarios_pool[start:end]
        else:
            batch_scenarios = scenarios_pool[start:] + scenarios_pool[:end - n]

        print(f"\n=== TRAIN STEP {step} (batch={len(batch_scenarios)}) ===")

        # ---- rollout ----
        train_groups = await art.gather_trajectory_groups(
            (
                TrajectoryGroup(
                    rollout_coder_ni(coder_model, s) for _ in range(8)
                )
                for s in batch_scenarios
            ),
            pbar_desc="gather",
        )

        n_signal = 0
        for gi, group in enumerate(train_groups):
            rewards = [t.reward for t in group]
            std = float(np.std(rewards))
            if std > 1e-3:
                n_signal += 1
            print(f"[group {gi}] rewards={rewards} std={std:.3f}")
        print(f"groups with signal: {n_signal}/{len(train_groups)}")

        result = await backend.train(
            coder_model, train_groups, learning_rate=1e-5
        )
        await coder_model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split='train',
        )

        print(f"✓ finished step {step}")
        step = await coder_model.get_step()


if __name__ == "__main__":
    import psutil
    for proc in psutil.process_iter(['pid', 'name']):
        if 'model-service' in proc.info['name'] or 'vllm' in proc.info['name']:
            proc.kill()
    asyncio.run(train())