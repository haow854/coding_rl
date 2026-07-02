"""HumanEval+/MBPP+ regression sanity (function completion; evalplus judge).
These are function-completion (assert-judged), NOT stdin/stdout — so they use a
different prompt from training. This is only a "did we break the basics?" check.

    python scripts/eval_evalplus.py --model Qwen/Qwen3.5-2B \
        --dataset humaneval --out outputs/eval/he_base.jsonl
    evalplus.evaluate --dataset humaneval --samples outputs/eval/he_base.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcoder.eval.generate import generate   # noqa: E402
from rlcoder.rewards import extract_code      # noqa: E402

FN_SYSTEM = (
    "You are an expert Python programmer. Complete the function. "
    "Return ONLY the complete function inside one ```python code block."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="outputs/eval/evalplus.jsonl")
    args = ap.parse_args()

    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    data = get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    items = list(data.items())
    msgs = [[{"role": "system", "content": FN_SYSTEM},
             {"role": "user", "content": v["prompt"]}] for _, v in items]

    gens = generate(args.model, msgs, n=args.n, temperature=args.temperature,
                    max_tokens=args.max_tokens, lora_path=args.lora)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for (task_id, _), samples in zip(items, gens):
            for s in samples:
                f.write(json.dumps({"task_id": task_id, "solution": extract_code(s)}) + "\n")
    print(f"wrote {args.out}")
    print(f"now run:  evalplus.evaluate --dataset {args.dataset} --samples {args.out}")


if __name__ == "__main__":
    main()
