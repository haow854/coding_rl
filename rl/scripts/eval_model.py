"""Evaluate a model on a normalised problem JSONL (our sandbox judge) — use this
for the in-house competitive held-out set and for base-vs-RL comparison.

    # baseline (base model), pass@1 and pass@5 estimated from n samples
    python scripts/eval_model.py --model Qwen/Qwen3-4B \
        --data data/dev_internal.jsonl --out outputs/eval/base.json

    # after SFT/GRPO (LoRA adapter) — keep sampling identical to the baseline
    python scripts/eval_model.py --model Qwen/Qwen3-4B --lora outputs/qwen3_4b_grpo \
        --data data/dev_internal.jsonl --out outputs/eval/rl.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.data.load import load_clean_jsonl   # noqa: E402
from rlcoder.eval.run_eval import evaluate         # noqa: E402
from rlcoder.rewards import extract_code            # noqa: E402


def _default_answers_path(args) -> str:
    """Keep raw generations next to the metric file, or use a stable fallback."""
    if args.answers_out:
        return args.answers_out
    if args.out:
        stem, _ = os.path.splitext(args.out)
        return stem + ".answers.jsonl"
    model_name = os.path.basename(args.lora or args.model).replace(" ", "_")
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    return os.path.join("outputs", "eval", f"{data_name}_{model_name}.answers.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--data", default="data/dev_internal.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8,
                    help="Samples/problem. >1 gives a low-variance pass@1 estimate; "
                         "single greedy pass@1 is too noisy for small deltas.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=-1,
                    help="Set to 20 for Qwen3 report-like thinking eval.")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="Matches GRPO --max-completion; raise for very "
                         "long-reasoning problems (slower, less batching).")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM context length. Use 32768+ for report-like long thinking.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Render prompts in non-thinking mode; use the same "
                         "setting as training for base-vs-RL comparisons.")
    ap.add_argument("--ks", default="1,5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--answers-out", default=None,
                    help="JSONL destination for every raw completion and extracted code. "
                         "Defaults next to --out, or under outputs/eval/.")
    args = ap.parse_args()

    problems = load_clean_jsonl(args.data, limit=args.limit)
    ks = [int(x) for x in args.ks.split(",")]
    answers_out = _default_answers_path(args)
    os.makedirs(os.path.dirname(answers_out) or ".", exist_ok=True)

    with open(answers_out, "w", encoding="utf-8") as answers_file:
        def save_answer(problem, sample_index, completion, breakdown) -> None:
            record = {
                "problem_id": problem.problem_id,
                "source": problem.source,
                "sample_index": sample_index,
                "completion": completion,
                "code": extract_code(completion),
                "passed": breakdown.all_passed,
                "pass_rate": breakdown.pass_rate,
                "ran": breakdown.ran,
                "timed_out": breakdown.timed_out,
            }
            answers_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        res, _ = evaluate(args.model, problems, n=args.n, temperature=args.temperature,
                          top_p=args.top_p, top_k=args.top_k,
                          max_tokens=args.max_tokens,
                          max_model_len=args.max_model_len,
                          gpu_mem_util=args.gpu_mem_util,
                          lora_path=args.lora, ks=ks,
                          enable_thinking=not args.no_thinking,
                          sample_callback=save_answer,
                          show_judge_progress=True)
    res["answers_out"] = answers_out
    print(json.dumps(res, indent=2))
    print("wrote", answers_out)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "lora": args.lora, "data": args.data,
                       "no_thinking": args.no_thinking, **res}, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
