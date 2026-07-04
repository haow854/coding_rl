"""Official LiveCodeBench code-generation evaluation wrapper.

This script delegates generation and judging to the official LiveCodeBench
`lcb_runner` package. The project-local stdin/stdout-only evaluator is kept as
`scripts/eval_livecodebench_subset.py`.

RunPod setup:

    cd /workspace/coding_rl
    python -m pip install -r requirements.txt

Qwen3 report-ish run:

    export VLLM_USE_FLASHINFER_SAMPLER=1
    python scripts/eval_livecodebench.py --model Qwen/Qwen3-4B \
        --release-version release_v5 --start-date 2024-10-01 \
        --end-date 2025-02-28 --model-style QwQ \
        --n 1 --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-tokens 32768 --max-model-len 40960 --gpu-mem-util 0.95
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


INSTALL_HELP = """Official LiveCodeBench is not importable.

The wrapper needs the official LiveCodeBench source tree, because upstream's
package metadata currently omits subpackages like `lcb_runner.runner`.
Clone it next to this repo, for example:

    cd /workspace/coding_rl
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git

Or pass --lcb-root /path/to/LiveCodeBench.
"""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id or the model name registered with LCB.")
    ap.add_argument("--local-model-path", "--local_model_path", default=None,
                    help="Local merged model path for official LCB/vLLM.")
    ap.add_argument("--lora", default=None,
                    help="Not supported by official LCB. Merge first with scripts/merge_lora.py.")
    ap.add_argument("--lcb-root", default=None,
                    help="Path to a cloned LiveCodeBench repo, if not pip-installed.")

    ap.add_argument("--release-version", "--release_version", "--version-tag",
                    dest="release_version", default="release_v5")
    ap.add_argument("--start-date", "--start_date", default=None)
    ap.add_argument("--end-date", "--end_date", default=None)
    ap.add_argument("--not-fast", "--not_fast", action="store_true")

    ap.add_argument("--n", type=int, default=10,
                    help="Official LiveCodeBench default is 10.")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="Official LiveCodeBench default is 0.2; Qwen3 thinking recommends 0.6.")
    ap.add_argument("--top-p", "--top_p", type=float, default=0.95)
    ap.add_argument("--top-k", "--top_k", type=int, default=-1,
                    help="Injected into vLLM SamplingParams; official LCB has no CLI flag for this.")
    ap.add_argument("--max-tokens", "--max_tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="Injected into vLLM LLM(...). Use 40960 for 32K output plus prompt.")
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="Injected as vLLM gpu_memory_utilization.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", "--tensor_parallel_size", type=int, default=None)
    ap.add_argument("--enable-prefix-caching", "--enable_prefix_caching", action="store_true")
    ap.add_argument("--trust-remote-code", "--trust_remote_code", action="store_true")

    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--num-process-evaluate", "--num_process_evaluate", type=int, default=12)
    ap.add_argument("--no-evaluate", action="store_true",
                    help="Only generate official LCB outputs; do not run the judge.")
    ap.add_argument("--use-cache", "--use_cache", action="store_true")
    ap.add_argument("--continue-existing", "--continue_existing", action="store_true")
    ap.add_argument("--continue-existing-with-eval", "--continue_existing_with_eval",
                    action="store_true")
    ap.add_argument("--stop", default=None,
                    help="Comma-separated stop strings passed to official LCB.")

    ap.add_argument("--model-style", default="auto",
                    choices=["auto", "CodeQwenInstruct", "QwQ", "DeepSeekR1",
                             "GenericBase", "LLaMa3"],
                    help="Style to register if official LCB does not know --model.")
    ap.add_argument("--model-repr", default=None,
                    help="Display name for a model dynamically registered with LCB.")
    ap.add_argument("--release-date", default="2024-06-30",
                    help="Release date used when dynamically registering --model.")
    ap.add_argument("--link", default=None)

    # Accepted for backward compatibility with the old subset script.
    ap.add_argument("--out", default=None,
                    help="Ignored: official LCB writes into its own output directory.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Ignored: official LCB has no exact equivalent.")
    ap.add_argument("--max-tests", type=int, default=None,
                    help="Ignored: official LCB controls lite/full tests via --not-fast.")
    ap.add_argument("--ks", default=None,
                    help="Ignored: official LCB reports its own pass@k metrics.")

    ap.add_argument("--official-arg", action="append", default=[],
                    help="Extra raw argument(s) appended to lcb_runner.runner.main.")
    return ap.parse_args()


def _purge_lcb_modules() -> None:
    for name in list(sys.modules):
        if name == "lcb_runner" or name.startswith("lcb_runner."):
            del sys.modules[name]


def _candidate_lcb_roots(lcb_root: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    if lcb_root:
        roots.append(Path(lcb_root).expanduser())

    script_path = Path(__file__).resolve()
    search_bases = [Path.cwd(), *script_path.parents]
    for base in search_bases:
        roots.append(base / "LiveCodeBench")
        if base.name == "coding_rl":
            roots.append(base.parent / "LiveCodeBench")
    roots.append(Path("/workspace/coding_rl/LiveCodeBench"))
    roots.append(Path("/workspace/LiveCodeBench"))

    seen = set()
    out = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def _looks_like_lcb_source(root: Path) -> bool:
    return (
        (root / "lcb_runner" / "runner" / "main.py").is_file()
        and (root / "lcb_runner" / "prompts" / "few_shot_examples"
             / "generation" / "func.json").is_file()
    )


def _ensure_lcb_importable(lcb_root: Optional[str]) -> Path:
    for root in _candidate_lcb_roots(lcb_root):
        if not _looks_like_lcb_source(root):
            continue
        _purge_lcb_modules()
        sys.path.insert(0, str(root))
        spec = importlib.util.find_spec("lcb_runner.runner.main")
        if spec is not None:
            print(f"[lcb-official] using LiveCodeBench source: {root}")
            return root

    _purge_lcb_modules()
    spec = importlib.util.find_spec("lcb_runner.runner.main")
    if spec is not None and spec.origin:
        root = Path(spec.origin).resolve().parents[2]
        print(f"[lcb-official] using installed LiveCodeBench source: {root}")
        return root

    searched = "\n".join(f"  - {p}" for p in _candidate_lcb_roots(lcb_root))
    raise SystemExit(
        INSTALL_HELP
        + "\nSearched these source roots:\n"
        + searched
        + "\n\nNote: `pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git` "
          "can install a partial package that lacks `lcb_runner.runner`; the source "
          "checkout is required."
    )


def _auto_style(model: str) -> str:
    lower = model.lower()
    if "qwen3" in lower or "qwq" in lower:
        return "QwQ"
    if "deepseek-r1" in lower:
        return "DeepSeekR1"
    if "qwen" in lower:
        return "CodeQwenInstruct"
    if "llama-3" in lower or "llama3" in lower:
        return "LLaMa3"
    return "GenericBase"


def _register_model_if_needed(args: argparse.Namespace) -> None:
    from lcb_runner.lm_styles import LMStyle, LanguageModel, LanguageModelList, LanguageModelStore

    if args.model in LanguageModelStore and args.model_style == "auto":
        return

    style_name = _auto_style(args.model) if args.model_style == "auto" else args.model_style
    style = getattr(LMStyle, style_name)
    release_date = datetime.fromisoformat(args.release_date)
    model_repr = args.model_repr or args.model.split("/")[-1]
    link = args.link or (f"https://huggingface.co/{args.model}" if "/" in args.model else None)
    model = LanguageModel(args.model, model_repr, style, release_date, link=link)
    action = "overrode" if args.model in LanguageModelStore else "registered"
    LanguageModelList.append(model)
    LanguageModelStore[args.model] = model
    print(f"[lcb-official] {action} {args.model!r} as LMStyle.{style_name}")


def _patch_vllm(args: argparse.Namespace) -> None:
    if args.top_k <= 0 and args.max_model_len is None and args.gpu_mem_util is None:
        return

    try:
        import vllm
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "vLLM is required for official LiveCodeBench open-model generation.\n"
            "Install the project environment first:\n\n"
            "    cd /workspace/coding_rl\n"
            "    python -m pip install -r requirements.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    original_sampling_params = vllm.SamplingParams
    # Import LLM before monkey-patching SamplingParams. vLLM imports its own
    # SamplingParams in type annotations such as `SamplingParams | None`; if the
    # class has already been replaced by our wrapper function, that import fails.
    original_llm = vllm.LLM

    def sampling_params(*pos, **kwargs):
        if args.top_k > 0:
            kwargs.setdefault("top_k", args.top_k)
        return original_sampling_params(*pos, **kwargs)

    vllm.SamplingParams = sampling_params

    def llm(*pos, **kwargs):
        if args.max_model_len is not None and not kwargs.get("max_model_len"):
            kwargs["max_model_len"] = args.max_model_len
        if args.gpu_mem_util is not None and not kwargs.get("gpu_memory_utilization"):
            kwargs["gpu_memory_utilization"] = args.gpu_mem_util
        return original_llm(*pos, **kwargs)

    vllm.LLM = llm
    patched = []
    if args.top_k > 0:
        patched.append(f"top_k={args.top_k}")
    if args.max_model_len is not None:
        patched.append(f"max_model_len={args.max_model_len}")
    if args.gpu_mem_util is not None:
        patched.append(f"gpu_memory_utilization={args.gpu_mem_util}")
    print("[lcb-official] patched vLLM " + ", ".join(patched))


def _build_official_argv(args: argparse.Namespace) -> List[str]:
    argv = [
        "lcb_runner.runner.main",
        "--model", args.model,
        "--scenario", "codegeneration",
        "--release_version", args.release_version,
        "--n", str(args.n),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--max_tokens", str(args.max_tokens),
        "--dtype", args.dtype,
        "--timeout", str(args.timeout),
        "--num_process_evaluate", str(args.num_process_evaluate),
    ]
    if not args.no_evaluate:
        argv.append("--evaluate")
    if args.local_model_path:
        argv += ["--local_model_path", args.local_model_path]
    if args.start_date:
        argv += ["--start_date", args.start_date]
    if args.end_date:
        argv += ["--end_date", args.end_date]
    if args.tensor_parallel_size is not None:
        argv += ["--tensor_parallel_size", str(args.tensor_parallel_size)]
    if args.stop is not None:
        argv += ["--stop", args.stop]
    for flag_name, enabled in [
        ("--not_fast", args.not_fast),
        ("--trust_remote_code", args.trust_remote_code),
        ("--enable_prefix_caching", args.enable_prefix_caching),
        ("--use_cache", args.use_cache),
        ("--continue_existing", args.continue_existing),
        ("--continue_existing_with_eval", args.continue_existing_with_eval),
    ]:
        if enabled:
            argv.append(flag_name)
    for raw in args.official_arg:
        argv.extend(shlex.split(raw))
    return argv


def _warn_ignored(args: argparse.Namespace) -> None:
    if args.lora:
        raise SystemExit(
            "Official LiveCodeBench does not load PEFT LoRA adapters directly.\n"
            "Merge first, then pass --local-model-path, e.g.\n\n"
            "    python scripts/merge_lora.py --base Qwen/Qwen3-4B "
            "--adapter outputs/qwen3_4b_sft --out outputs/qwen3_4b_sft_merged\n"
            "    python scripts/eval_livecodebench.py --model qwen3_4b_sft_merged "
            "--local-model-path outputs/qwen3_4b_sft_merged ..."
        )
    ignored = []
    for name in ["out", "limit", "max_tests", "ks"]:
        if getattr(args, name) not in (None, ""):
            ignored.append("--" + name.replace("_", "-"))
    if ignored:
        print("[lcb-official] ignored old subset-only args: " + ", ".join(ignored))


def main() -> None:
    args = _parse_args()
    _warn_ignored(args)
    lcb_root = _ensure_lcb_importable(args.lcb_root)
    if args.local_model_path:
        args.local_model_path = str(Path(args.local_model_path).expanduser().resolve())
    _register_model_if_needed(args)
    _patch_vllm(args)

    argv = _build_official_argv(args)
    print("[lcb-official] running: " + shlex.join(argv))
    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        os.chdir(lcb_root)
        sys.argv = argv
        runpy.run_module("lcb_runner.runner.main", run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


if __name__ == "__main__":
    main()
