"""Run official LiveCodeBench from a source checkout.

Why this wrapper exists:
- The upstream pip/Git package can install an incomplete `lcb_runner` package
  that lacks subpackages such as `lcb_runner.runner`.
- The official runner opens prompt JSON files by relative path, so it must run
  with the LiveCodeBench source root as the working directory.
- Qwen3 report-style eval needs knobs official LCB does not expose directly,
  notably vLLM `top_k`, `max_model_len`, and GPU memory utilization.
- Two official LCB defaults are wrong for thinking models and are overridden
  here: `--stop` defaults to "###", which truncates Qwen3-style answers at
  "### Approach" before the final code block (grading them 0), so it is
  dropped for chat/thinking model styles unless --stop is passed explicitly;
  and the vLLM runner hardcodes `enforce_eager=True` (CUDA graphs off, several
  times slower decode), which is disabled unless --enforce-eager is passed.

Example:

    cd /workspace/coding_rl
    export VLLM_USE_FLASHINFER_SAMPLER=1
    python rl/scripts/eval_livecodebench.py \
      --model Qwen/Qwen3-1.7B \
      --release-version release_v5 \
      --start-date 2024-10-01 --end-date 2025-02-28 \
      --n 1 --temperature 0.6 --top-p 0.95 --top-k 20 \
      --max-tokens 32768 --max-model-len 40960 --gpu-mem-util 0.95
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import runpy
import shlex
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--local-model-path", "--local_model_path", default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lcb-root", default=None,
                    help="Path to official LiveCodeBench source checkout.")

    ap.add_argument("--release-version", "--release_version", "--version-tag",
                    dest="release_version", default="release_v5")
    ap.add_argument("--start-date", "--start_date", default=None)
    ap.add_argument("--end-date", "--end_date", default=None)
    ap.add_argument("--not-fast", "--not_fast", action="store_true")

    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", "--top_p", type=float, default=0.95)
    ap.add_argument("--top-k", "--top_k", type=int, default=-1,
                    help="Injected into vLLM SamplingParams.")
    ap.add_argument("--max-tokens", "--max_tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="Injected into vLLM LLM(...).")
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="Injected as vLLM gpu_memory_utilization.")
    ap.add_argument("--max-num-seqs", "--max_num_seqs", type=int, default=None,
                    help="Injected as vLLM max_num_seqs. Cap concurrency so "
                         "long thinking rollouts fit in KV cache without "
                         "preemption thrash (see 'Maximum concurrency' in the "
                         "vLLM startup log for a guide).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel-size", "--tensor_parallel_size",
                    type=int, default=None)
    ap.add_argument("--enable-prefix-caching", "--enable_prefix_caching",
                    action="store_true")
    ap.add_argument("--enforce-eager", "--enforce_eager", action="store_true",
                    help="Keep official LCB's enforce_eager=True (disables "
                         "CUDA graphs; several times slower decode).")
    ap.add_argument("--trust-remote-code", "--trust_remote_code",
                    action="store_true")

    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--num-process-evaluate", "--num_process_evaluate",
                    type=int, default=12)
    ap.add_argument("--no-evaluate", action="store_true")
    ap.add_argument("--use-cache", "--use_cache", action="store_true")
    ap.add_argument("--continue-existing", "--continue_existing",
                    action="store_true")
    ap.add_argument("--continue-existing-with-eval",
                    "--continue_existing_with_eval", action="store_true")
    ap.add_argument("--stop", default=None,
                    help="Comma-separated stop strings passed to official LCB.")

    ap.add_argument("--no-think", "--no-thinking", "--no_think",
                    dest="no_think", action="store_true",
                    help="Append an empty <think></think> block to ChatML "
                         "prompts (Qwen3 non-thinking mode). Pair with "
                         "--model-repr to keep output dirs separate from "
                         "thinking runs.")
    ap.add_argument("--model-style", default="auto",
                    choices=["auto", "CodeQwenInstruct", "QwQ",
                             "DeepSeekR1", "GenericBase", "LLaMa3"])
    ap.add_argument("--model-repr", default=None)
    ap.add_argument("--release-date", default="2024-06-30")
    ap.add_argument("--link", default=None)

    # Backward-compatible no-ops from the old local subset evaluator.
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tests", type=int, default=None)
    ap.add_argument("--ks", default=None)

    ap.add_argument("--official-arg", action="append", default=[],
                    help="Extra raw argument(s) appended to official LCB.")
    return ap.parse_args()


def _repo_root() -> Path:
    # .../coding_rl/rl/scripts/eval_livecodebench.py -> .../coding_rl
    return Path(__file__).resolve().parents[2]


def _candidate_lcb_roots(explicit_root: Optional[str]) -> List[Path]:
    roots: List[Path] = []
    if explicit_root:
        roots.append(Path(explicit_root).expanduser())

    cwd = Path.cwd()
    repo = _repo_root()
    roots.extend([
        repo / "LiveCodeBench",
        cwd / "LiveCodeBench",
        cwd.parent / "LiveCodeBench",
        Path("/workspace/coding_rl/LiveCodeBench"),
        Path("/workspace/LiveCodeBench"),
    ])
    for parent in Path(__file__).resolve().parents:
        roots.append(parent / "LiveCodeBench")

    seen = set()
    unique: List[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _is_lcb_source(root: Path) -> bool:
    return (
        (root / "lcb_runner" / "runner" / "main.py").is_file()
        and (root / "lcb_runner" / "runner" / "vllm_runner.py").is_file()
        and (root / "lcb_runner" / "prompts" / "few_shot_examples"
             / "generation" / "func.json").is_file()
    )


def _purge_lcb_modules() -> None:
    for name in list(sys.modules):
        if name == "lcb_runner" or name.startswith("lcb_runner."):
            del sys.modules[name]


def _install_lcb_namespace(root: Path) -> None:
    """Make this source checkout win over any broken pip-installed package."""
    _purge_lcb_modules()
    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)

    pkg_dir = root / "lcb_runner"
    pkg = types.ModuleType("lcb_runner")
    pkg.__package__ = "lcb_runner"
    pkg.__path__ = [str(pkg_dir)]
    pkg.__file__ = None
    spec = importlib.machinery.ModuleSpec("lcb_runner", loader=None,
                                          is_package=True)
    spec.submodule_search_locations = [str(pkg_dir)]
    pkg.__spec__ = spec
    sys.modules["lcb_runner"] = pkg


def _find_lcb_source(explicit_root: Optional[str]) -> Path:
    for root in _candidate_lcb_roots(explicit_root):
        if _is_lcb_source(root):
            _install_lcb_namespace(root)
            print(f"[lcb-official] using LiveCodeBench source: {root}")
            return root

    searched = "\n".join(f"  - {p}" for p in _candidate_lcb_roots(explicit_root))
    raise SystemExit(
        "Could not find a complete official LiveCodeBench source checkout.\n"
        "Clone it into the project root:\n\n"
        "    cd /workspace/coding_rl\n"
        "    git clone https://github.com/LiveCodeBench/LiveCodeBench.git\n\n"
        "Or pass --lcb-root /path/to/LiveCodeBench.\n\n"
        "Searched:\n" + searched
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


def _register_model(args: argparse.Namespace) -> None:
    from lcb_runner.lm_styles import (
        LMStyle,
        LanguageModel,
        LanguageModelList,
        LanguageModelStore,
    )

    if args.model in LanguageModelStore and args.model_style == "auto":
        return

    style_name = _auto_style(args.model) if args.model_style == "auto" else args.model_style
    style = getattr(LMStyle, style_name)
    model_repr = args.model_repr or args.model.split("/")[-1]
    link = args.link or (f"https://huggingface.co/{args.model}" if "/" in args.model else None)
    model = LanguageModel(
        args.model,
        model_repr,
        style,
        datetime.fromisoformat(args.release_date),
        link=link,
    )
    action = "overrode" if args.model in LanguageModelStore else "registered"
    LanguageModelList.append(model)
    LanguageModelStore[args.model] = model
    print(f"[lcb-official] {action} {args.model!r} as LMStyle.{style_name}")


def _patch_vllm(args: argparse.Namespace) -> None:
    _patch_nvrtc_library_path()

    try:
        import vllm  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "vLLM is required for official LiveCodeBench open-model generation.\n"
            "Install dependencies first:\n\n"
            "    cd /workspace/coding_rl\n"
            "    python -m pip install -r requirements.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    from lcb_runner.lm_styles import LanguageModelStore, LMStyle
    from lcb_runner.runner import vllm_runner

    original_sampling_params = vllm_runner.SamplingParams
    original_llm = vllm_runner.LLM

    # Official LCB defaults --stop to "###". Chat/thinking models (Qwen3, QwQ,
    # R1) emit "### Approach" headers before the final code block, so that stop
    # truncates the answer and the extractor grades it 0. GenericBase few-shot
    # prompts genuinely rely on "###" as the example separator, so keep it there.
    style = LanguageModelStore[args.model].model_style
    drop_default_stop = args.stop is None and style != LMStyle.GenericBase

    def sampling_params(*pos, **kwargs):
        if drop_default_stop:
            stop = kwargs.get("stop")
            if isinstance(stop, str):
                stop = [stop]
            if stop:
                stop = [s for s in stop if s != "###"]
                kwargs["stop"] = stop or None
        if args.top_k > 0:
            kwargs.setdefault("top_k", args.top_k)
        return original_sampling_params(*pos, **kwargs)

    def llm(*pos, **kwargs):
        if args.max_model_len is not None and not kwargs.get("max_model_len"):
            kwargs["max_model_len"] = args.max_model_len
        if args.gpu_mem_util is not None and not kwargs.get("gpu_memory_utilization"):
            kwargs["gpu_memory_utilization"] = args.gpu_mem_util
        if args.max_num_seqs is not None and not kwargs.get("max_num_seqs"):
            kwargs["max_num_seqs"] = args.max_num_seqs
        if not args.enforce_eager:
            kwargs["enforce_eager"] = False
        return original_llm(*pos, **kwargs)

    # Patch only official LCB's imported references. Do not replace
    # vllm.SamplingParams globally: vLLM engine warmup uses classmethods such as
    # SamplingParams.for_sampler_warmup() in worker processes.
    vllm_runner.SamplingParams = sampling_params
    vllm_runner.LLM = llm

    patched = []
    if drop_default_stop:
        patched.append("dropped default stop '###'")
    if not args.enforce_eager:
        patched.append("enforce_eager=False (CUDA graphs on)")
    if args.top_k > 0:
        patched.append(f"top_k={args.top_k}")
    if args.max_model_len is not None:
        patched.append(f"max_model_len={args.max_model_len}")
    if args.gpu_mem_util is not None:
        patched.append(f"gpu_memory_utilization={args.gpu_mem_util}")
    if args.max_num_seqs is not None:
        patched.append(f"max_num_seqs={args.max_num_seqs}")
    print("[lcb-official] patched vLLM: " + ", ".join(patched))


def _patch_no_think(args: argparse.Namespace) -> None:
    """Qwen3 non-thinking mode: the official chat template renders an empty
    <think> block inside the assistant turn when enable_thinking=False. LCB's
    handwritten ChatML prompts never do this, so inject it here."""
    if not args.no_think:
        return

    from lcb_runner.prompts import code_generation
    from lcb_runner.runner import scenario_router

    original = code_generation.format_prompt_generation

    def format_prompt_no_think(question, style):
        prompt = original(question, style)
        if isinstance(prompt, str) and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "<think>\n\n</think>\n\n"
        return prompt

    code_generation.format_prompt_generation = format_prompt_no_think
    scenario_router.format_prompt_generation = format_prompt_no_think
    print("[lcb-official] no-think: appending empty <think> block to ChatML prompts")


def _patch_nvrtc_library_path() -> None:
    try:
        import nvidia.cuda_nvrtc as cuda_nvrtc
    except Exception as exc:  # noqa: BLE001
        print(f"[lcb-official] warning: nvidia-cuda-nvrtc not importable: {exc}")
        return

    lib_dir = str(Path(cuda_nvrtc.__file__).resolve().parent / "lib")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in old.split(":") if p]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + old if old else "")
        print(f"[lcb-official] added nvrtc lib path: {lib_dir}")


def _raw_lcb_jsonl_names(version_tag: str) -> List[str]:
    try:
        n = int(version_tag.rsplit("v", 1)[-1])
    except Exception:  # noqa: BLE001
        n = 1
    return ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, n + 1)]


def _load_lcb_raw_jsonl(repo_id: str, version_tag: str) -> List[dict]:
    from huggingface_hub import hf_hub_download

    rows: List[dict] = []
    got: List[str] = []
    for name in _raw_lcb_jsonl_names(version_tag):
        try:
            path = hf_hub_download(repo_id, name, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"[lcb-official] skipped {repo_id}/{name}: {exc}")
            continue
        got.append(name)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(
            f"Could not load raw LiveCodeBench JSONL files for {repo_id} "
            f"{version_tag}. Tried: {_raw_lcb_jsonl_names(version_tag)}"
        )
    print(f"[lcb-official] loaded {len(rows)} rows from {repo_id} raw JSONL {got}")
    return rows


def _patch_datasets_for_lcb() -> None:
    """Make official LCB work with datasets>=4, which removed script loading."""
    try:
        import datasets
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "datasets is required for LiveCodeBench. Install dependencies first:\n\n"
            "    python -m pip install -r requirements.txt\n\n"
            f"Original import error: {exc}"
        ) from exc

    original_load_dataset = datasets.load_dataset

    def load_dataset_compat(path, *args, **kwargs):
        if path == "livecodebench/code_generation_lite":
            version_tag = kwargs.get("version_tag")
            if version_tag is None and args:
                version_tag = args[0]
            version_tag = version_tag or "release_latest"
            if version_tag == "release_latest":
                version_tag = "release_v6"
            return _load_lcb_raw_jsonl(path, str(version_tag))
        return original_load_dataset(path, *args, **kwargs)

    datasets.load_dataset = load_dataset_compat
    print("[lcb-official] patched datasets.load_dataset for LCB raw JSONL")


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

    flags = [
        ("--not_fast", args.not_fast),
        ("--trust_remote_code", args.trust_remote_code),
        ("--enable_prefix_caching", args.enable_prefix_caching),
        ("--use_cache", args.use_cache),
        ("--continue_existing", args.continue_existing),
        ("--continue_existing_with_eval", args.continue_existing_with_eval),
    ]
    for flag, enabled in flags:
        if enabled:
            argv.append(flag)
    for raw in args.official_arg:
        argv.extend(shlex.split(raw))
    return argv


def _check_args(args: argparse.Namespace) -> None:
    if args.lora:
        raise SystemExit(
            "Official LiveCodeBench does not load PEFT LoRA adapters directly.\n"
            "Merge first, then pass --local-model-path:\n\n"
            "    python rl/scripts/merge_lora.py --base Qwen/Qwen3-4B "
            "--adapter outputs/qwen3_4b_sft --out outputs/qwen3_4b_sft_merged\n"
        )
    ignored = []
    for name in ("out", "limit", "max_tests", "ks"):
        if getattr(args, name) not in (None, ""):
            ignored.append("--" + name.replace("_", "-"))
    if ignored:
        print("[lcb-official] ignored old subset-only args: " + ", ".join(ignored))


def main() -> None:
    args = _parse_args()
    _check_args(args)
    lcb_root = _find_lcb_source(args.lcb_root)
    if args.local_model_path:
        args.local_model_path = str(Path(args.local_model_path).expanduser().resolve())

    main_path = lcb_root / "lcb_runner" / "runner" / "main.py"
    argv = _build_official_argv(args)

    old_cwd = Path.cwd()
    old_argv = sys.argv
    try:
        os.chdir(lcb_root)
        # Official LCB imports prompt modules that open files with paths relative
        # to the LiveCodeBench repo root, so all LCB imports must happen after
        # this chdir. Patch datasets before importing benchmark modules, because
        # they bind `from datasets import load_dataset` at import time.
        _patch_datasets_for_lcb()
        _register_model(args)
        _patch_no_think(args)
        _patch_vllm(args)
        print("[lcb-official] running: " + shlex.join(argv))
        sys.argv = argv
        runpy.run_path(str(main_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
