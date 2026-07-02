# llm_rl_coding - Code RLVR with Qwen3.5-2B

Train a single Qwen3.5 policy to solve Python coding problems with verifiable
rewards: the model writes code, the sandbox runs tests, and the test result is
the reward signal.

## Stage-1 Design

- **Base model:** `Qwen/Qwen3.5-2B-Base`.
- **Training:** LoRA SFT first (reasoning-trace imitation, not answer-only),
  then LoRA GRPO from the SFT adapter.
- **SFT data:** `open-r1/codeforces-cots` (`solutions_py_decontaminated`
  config) — `<think>...</think>` reasoning + fenced solution distilled from
  DeepSeek-R1 on Codeforces problems, so SFT teaches the reasoning pattern,
  not just final-answer code. Verified against this repo's judge before use
  (only ~84% of upstream solutions actually pass).
- **RL data:** `open-r1/verifiable-coding-problems-python_decontaminated-tested`,
  filtered again by running each gold solution against its own stdin/stdout
  tests under this repo's judge.
- **Prompt:** one problem in, `<think>` reasoning then one fenced Python 3
  stdin/stdout program out.
- **Reward:** dense test pass fraction + full-pass bonus + small format bonus
  + small think-bonus; timeouts and code that fails to run are penalized.
- **Eval:** same sandbox judge on a held-out set, with EvalPlus
  HumanEval+/MBPP+ as a regression sanity check.

This replaces the old multi-agent direction. A single policy keeps credit
assignment clear: every generated token belongs to the same model trajectory.

## Recommended Experiment

1. Build a clean verified RL pool (`verifiable-coding-problems-python`).
2. Build a clean verified SFT pool (`codeforces-cots`, reasoning traces).
3. Split `train_problems.jsonl` and `holdout.jsonl` from the RL pool.
4. Evaluate the base model on the holdout set.
5. SFT on CoT reasoning traces (not gold answers) from the SFT pool.
6. Evaluate the SFT adapter.
7. Difficulty-filter the RL pool using the SFT adapter.
8. Run GRPO from the SFT adapter.
9. Evaluate SFT+GRPO and compare against base/SFT/official Instruct.

Headline table:

| model | pass@1 | pass@5 | timeout rate | notes |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-2B-Base | TBD | TBD | TBD | base |
| + SFT LoRA | TBD | TBD | TBD | reasoning warm-up |
| + SFT -> GRPO LoRA | TBD | TBD | TBD | RLVR |
| Qwen3.5-2B-Instruct | TBD | TBD | TBD | official reference |

## Layout

```text
rlcoder/
  data/       dataset loading, parsing, gold verification
  sandbox/    safe-ish subprocess execution and stdin/stdout judging
  rewards/    RLVR reward function
  rollout/    prompt construction and TRL reward bridge
  train/      SFT and GRPO training entrypoints
  eval/       pass@k generation/evaluation helpers
scripts/
  build_dataset.py       build verified JSONL from HF or local sample
  split_data.py          make disjoint train/holdout files
  difficulty_filter.py   keep problems with non-zero reward variance
  eval_model.py          in-house sandbox pass@k eval
  eval_evalplus.py       HumanEval+/MBPP+ output generation
```

## Local Checks

CPU-only checks:

```bash
python scripts/sanity_check.py
python scripts/test_rollout.py
python -m py_compile rlcoder/train/sft_trl.py rlcoder/train/grpo_trl.py
```

GPU workflow is in `scripts/autodl_quickstart.md`.

## Notes

- Keep SFT and GRPO as LoRA adapters for the first experiment; merging is only
  needed when an external harness cannot load adapters.
- Use 2B for the first run. `Qwen/Qwen3.5-9B-Base` is an expansion target, not
  the first experiment.
- The GRPO script keeps DAPO-flavoured defaults (`beta=0`, `dr_grpo`,
  `epsilon_high=0.28`), but the project story should stay simple until the
  base/SFT/GRPO comparison is complete.
