# llm_rl_coding - Code RLVR with Qwen3.5-2B

Train a single Qwen3.5 policy to solve Python coding problems with verifiable
rewards: the model writes code, the sandbox runs tests, and the test result is
the reward signal.

## Stage-1 Design

- **Starting model:** `Qwen/Qwen3.5-2B` post-trained checkpoint.
- **Main training path:** LoRA GRPO directly from the post-trained model.
- **No mandatory thinking:** the default prompt asks for one fenced Python
  stdin/stdout program and no analysis text.
- **SFT:** optional ablation only. Do not reuse adapters trained from
  `Qwen/Qwen3.5-2B-Base` on the post-trained checkpoint.
- **RL data:** `open-r1/verifiable-coding-problems-python_decontaminated-tested`,
  normalized to this repo's stdin/stdout judge format.
- **Reward:** dense test pass fraction + full-pass bonus + small fenced-code
  format bonus; timeouts and code that fails to run are penalized. Thinking
  reward is off by default.
- **Eval:** same sandbox judge on a held-out set, with EvalPlus
  HumanEval+/MBPP+ as a regression sanity check.

This replaces the old multi-agent and Base+SFT-first direction. A single policy
keeps credit assignment clear: every generated token belongs to the same model
trajectory.

## Recommended Experiment

1. Build a clean normalized pool from the Open-R1 decontaminated/tested dataset.
2. Split off `dev_internal.jsonl`; keep the rest as `rl_pool.jsonl`.
3. Evaluate `Qwen/Qwen3.5-2B` on `dev_internal.jsonl`.
4. Difficulty-filter `rl_pool.jsonl` with the same post-trained policy.
5. Run a short GRPO smoke test.
6. If reward variance and reward mean look sane, run the first real GRPO job.
7. Evaluate the GRPO adapter against the post-trained baseline.

Headline table:

| model | pass@1 | pass@5 | timeout rate | notes |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B | TBD | TBD | TBD | post-trained baseline |
| + GRPO LoRA | TBD | TBD | TBD | RLVR |
| Qwen3.5-2B-Base | TBD | TBD | TBD | research reference only |

## Layout

```text
rlcoder/
  data/       dataset loading, parsing, gold verification
  sandbox/    subprocess execution and stdin/stdout judging
  rewards/    RLVR reward function
  rollout/    TRL reward bridge
  train/      SFT and GRPO training entrypoints
  eval/       pass@k generation/evaluation helpers
scripts/
  build_dataset.py       build verified JSONL from HF or local sample
  split_stages.py        make dev / optional SFT / RL splits
  difficulty_filter.py   keep problems with non-zero pass-count variance
  eval_model.py          in-house sandbox pass@k eval
  eval_evalplus.py       HumanEval+/MBPP+ output generation
```

## Local Checks

CPU-only checks:

```bash
python scripts/sanity_check.py
python -m py_compile rlcoder/train/sft_trl.py rlcoder/train/grpo_trl.py
```

`scripts/test_rollout.py` needs a local `data/clean_problems.jsonl`, so run it
after building data.

RunPod/GPU workflow is in `scripts/runpod_quickstart.md`.

## Notes

- Keep GRPO as a LoRA adapter for the first experiment; merge only when an
  external harness cannot load adapters.
- Use 2B for the first run. Larger Qwen3.5 checkpoints are expansion targets,
  not the first smoke test.
- The GRPO script keeps DAPO-flavoured defaults (`beta=0`, `dr_grpo`,
  `epsilon_high=0.28`) while keeping the project story simple: post-trained
  baseline versus GRPO.
