# llm_rl_coding — Code reasoning with Qwen3-4B (SFT distillation → GRPO)

Improve a small model's competitive-coding ability in two stages: first
**distill** reasoning traces into it with SFT, then optionally sharpen it with
**GRPO/RLVR** where the sandbox test result is the reward.

## Why this shape

Pure GRPO on a small base barely moves — multiple reproductions find no
significant gain from RL alone at this scale, and DeepSeek-R1's own finding is
that for small models *distillation beats RL*. So Stage 1 (SFT) does the heavy
lifting; Stage 2 (GRPO) is the refinement, not the main event.

Two earlier mistakes this repo now avoids:

- **Non-thinking on reasoning problems.** Competitive problems need a
  `<think>...</think>` derivation. Everything (SFT target, GRPO rollout,
  difficulty probe, eval) now runs thinking-first and consistently.
- **Truncation.** A 1024-token completion cap truncated ~half of all rollouts,
  turning the reward into a length signal. Completion/eval budgets are now sized
  for real reasoning traces (`max-completion` 4096+, SFT `max-length` 16384).

## Design

- **Base model:** `Qwen/Qwen3-4B` (text-native, thinking mode). The old
  `Qwen3.5-2B` was multimodal + non-thinking, which capped code ability and hit
  immature tooling support for its hybrid-attention architecture.
- **Stage 1 — SFT distillation (main):** `nvidia/OpenCodeReasoning` R1 traces,
  curated by `scripts/build_sft_data.py` (coverage-first, long-tail trimmed).
  Loss is masked to the completion; the `<think>` block is kept in the target.
- **Stage 2 — GRPO/RLVR (optional):** continue the SFT adapter
  (`--init-adapter`) on the verified stdin/stdout pool, difficulty-filtered by
  the *same* SFT policy.
- **Reward:** dense test-pass fraction + full-pass bonus + small fenced-code
  format bonus; timeouts and non-running code penalized. `extract_code()` judges
  only the final fenced block after `</think>`.
- **Eval:** same sandbox judge on a held-out set (multi-sample pass@1/pass@5),
  with EvalPlus HumanEval+/MBPP+ as a regression sanity check. LiveCodeBench is
  the recommended external headline metric.

## Recommended experiment

1. `build_sft_data.py` → curate ~30k OpenCodeReasoning traces to `data/sft_ocr.jsonl`.
2. `sft_trl.py` → SFT `Qwen/Qwen3-4B` (thinking) → `outputs/qwen3_4b_sft`.
3. `eval_model.py` → eval base vs SFT with **identical** sampling (this is the
   number that should move most).
4. (Optional GRPO) build a clean RL pool, `difficulty_filter.py` with the SFT
   adapter, then `grpo_trl.py --init-adapter outputs/qwen3_4b_sft`.
5. Eval the GRPO adapter against the SFT checkpoint.

Headline table:

| model | pass@1 | pass@5 | timeout rate | notes |
| --- | ---: | ---: | ---: | --- |
| Qwen3-4B | TBD | TBD | TBD | base, thinking |
| + SFT (OpenCodeReasoning) | TBD | TBD | TBD | Stage 1, main lift |
| + GRPO LoRA | TBD | TBD | TBD | Stage 2, refinement |

## Layout

```text
rlcoder/
  data/       dataset loading, parsing (parse_ocr = SFT source), gold verification
  sandbox/    subprocess execution and stdin/stdout judging
  rewards/    RLVR reward function (think-aware code extraction)
  rollout/    TRL reward bridge
  train/      SFT (Stage 1) and GRPO (Stage 2) entrypoints
  eval/       pass@k generation/evaluation helpers
scripts/
  build_sft_data.py      curate OpenCodeReasoning -> SFT JSONL (Stage 1)
  build_dataset.py       build verified JSONL for the GRPO pool
  split_stages.py        make dev / RL splits
  difficulty_filter.py   keep problems with non-zero pass-count variance
  eval_model.py          in-house sandbox pass@k eval
  eval_livecodebench.py  official LiveCodeBench harness wrapper
  eval_livecodebench_subset.py  local stdin/stdout-only LCB subset judge
  eval_evalplus.py       HumanEval+/MBPP+ output generation
```

## Local checks

CPU-only:

```bash
python scripts/sanity_check.py
python -m py_compile rlcoder/train/sft_trl.py rlcoder/train/grpo_trl.py \
    scripts/build_sft_data.py rlcoder/data/parse_ocr.py
```

`scripts/test_rollout.py` needs a local `data/clean_problems.jsonl`, so run it
after building data. RunPod/GPU workflow is in `scripts/runpod_quickstart.md`.

## Notes

- Keep both stages as LoRA adapters first; `--full-ft` is available on both
  `sft_trl.py` and `grpo_trl.py` when VRAM allows (lower the LR for full FT).
- Keep the difficulty probe, GRPO completion length, and eval length in sync;
  a mismatch makes the kept difficulty band meaningless.
- GRPO keeps DAPO-flavoured defaults (`dr_grpo`, `epsilon_high=0.28`) plus a
  small KL (`beta=1e-3`) as a stability anchor; set `--beta 0` for pure DAPO.
