import asyncio
import art
from art import TrainableModel, TrajectoryGroup
from art.local.backend import LocalBackend
from rollout_planner import rollout_no_incre
from query_full import load_coding_problem
from art.dev import InternalModelConfig, InitArgs, EngineArgs
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from agents.planner_train import Planner
from agents.coder_mbpp import Coder
from eval_planner import eval_planner
from test_planner import planner_evaluation
from openai import AsyncOpenAI
from transformers import AutoTokenizer
import nest_asyncio
from config import QWEN_7B
HF_MODEL_ID = QWEN_7B


async def train():
    #-------------planner-------------------------
    backend = LocalBackend(path='./art_planner')

    planner_model = TrainableModel(
        name="v1.0",
        project="code_generation",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        _internal_config=InternalModelConfig(
            init_args=InitArgs(
                max_seq_length=8192,
            ),
            engine_args=EngineArgs(
                max_num_seqs=32,
                gpu_memory_utilization=0.6,
            ),
        ),
    )

    await planner_model.register(backend)
    #---------------------------------------------

    #-------------coder-------------------------
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=HF_MODEL_ID,
        gpu_memory_utilization=0.25,
        max_model_len=8192,
        dtype="bfloat16",
        trust_remote_code=True
    ))
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    #-------------------------------------------

    # model
    planner = Planner(planner_model)
    coder_fixed = Coder(engine, tokenizer)

    scenarios = load_coding_problem(shuffle = True)
    n = len(scenarios)
    print(f"train dataset: {n}")
    max_steps = 500
    batch_size = 15

    step = await planner_model.get_step()
    while step < max_steps:
        if step % 24 == 0:
            try:
                run_rate, pass_rate = await planner_evaluation(planner_model, engine,tokenizer)
                output_file = "planner_pass_rate_3B.txt"
                with open(output_file, "a", encoding="utf-8") as f:
                    log_entry = f"step:{step}, run_rate:{run_rate:.2%}, pass_rate:{pass_rate:.2%}\n"
                    f.write(log_entry)
                print(f"Run {run_rate:.2%}, Pass {pass_rate:.2%}")
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

            try:
                output_file = await eval_planner(planner_model, engine, tokenizer, step)
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n=== TRAIN STEP {step} ===")

        start = (step * batch_size) % n
        end = start + batch_size
        if end <= n:
            batch_scenarios = scenarios[start:end]
        else:
            batch_scenarios = scenarios[start:] + scenarios[:end - n]

        train_groups = await art.gather_trajectory_groups(
            (
                TrajectoryGroup(
                    rollout_no_incre(planner, coder_fixed, scenario) for _ in range(8)
                )
                for scenario in batch_scenarios
            ),
            pbar_desc="gather",
        )
        
        print("num train groups:", len(train_groups))
        print("group size:", len(train_groups[0]))

        for gi, group in enumerate(train_groups):
            rewards = [traj.reward for traj in group]
            print(f"[group {gi}] rewards = {rewards}")

        result = await backend.train(
            planner_model, 
            train_groups, 
            learning_rate=5e-6
        )

        await planner_model.log(
            train_groups, 
            metrics=result.metrics, 
            step=result.step, 
            split='train'
        )

        step = await planner_model.get_step()


if __name__ == "__main__":
    import psutil, os
    for proc in psutil.process_iter(['pid', 'name']):
        if 'model-service' in proc.info['name'] or 'vllm' in proc.info['name']:
            proc.kill()
    asyncio.run(train())
       
