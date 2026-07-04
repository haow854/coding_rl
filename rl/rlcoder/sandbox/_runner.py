"""Standalone subprocess runner. Reads one JSON job from stdin, prints one JSON
result line to stdout. Dependency-free (stdlib only) so it can be copied to a
remote box and run under a hardened sandbox (nsjail / firejail).

Two judging modes:
  - "assert"        : exec the candidate code once, then exec each assert snippet
                      in the same namespace (MBPP / HumanEval style).
  - "stdin_stdout"  : for each test, run the candidate program fresh with the
                      test's stdin and compare its stdout to the expected output
                      (competitive style: APPS / Codeforces / TACO).

Job:    {"code": str, "tests": [...], "mode": "assert"|"stdin_stdout", "setup": str}
        assert       -> tests is a list of code strings
        stdin_stdout -> tests is a list of {"input": str, "output": str}
"""
import contextlib
import io
import json
import sys

# Permissive preamble for assert-mode solutions that assume common imports.
SETUP = """
import math, cmath, re, heapq, collections, sys, string
import functools, itertools, random, operator, bisect
from math import *
from typing import *
from collections import Counter, defaultdict, deque, OrderedDict
from itertools import chain, groupby, combinations, permutations, product
from functools import reduce, lru_cache
from operator import itemgetter
from copy import deepcopy, copy
"""


def _norm(s: str) -> str:
    """Normalise program output for comparison: drop trailing spaces per line and
    leading/trailing blank lines (standard competitive-judge leniency)."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    while lines and lines[0] == "":
        lines.pop(0)
    return "\n".join(lines)


def _to_float(tok: str):
    try:
        return float(tok)
    except (ValueError, OverflowError):
        return None


def _cmp(got: str, want: str, rel: float = 1e-6, abs_: float = 1e-6) -> bool:
    """Competitive-judge comparison: exact after whitespace-normalisation, else
    token-by-token with float tolerance (recovers float precision / int-vs-float)."""
    if _norm(got) == _norm(want):
        return True
    gt, wt = got.split(), want.split()
    if len(gt) != len(wt):
        return False
    for a, b in zip(gt, wt):
        if a == b:
            continue
        fa, fb = _to_float(a), _to_float(b)
        if fa is None or fb is None:
            return False
        if abs(fa - fb) > abs_ + rel * abs(fb):
            return False
    return True


def run_assert(code: str, setup: str, tests: list) -> dict:
    env: dict = {}
    try:
        exec(SETUP, env)
        if setup:
            exec(setup, env)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, env)
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "stage": "build", "error": repr(e),
                "passed": 0, "total": len(tests), "tests": []}

    results, passed = [], 0
    for t in tests:
        rec = {"test": str(t)[:120], "passed": False, "error": None}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(t, env)
            rec["passed"] = True
            passed += 1
        except BaseException as e:  # noqa: BLE001
            rec["error"] = repr(e)
        results.append(rec)
    return {"ok": True, "stage": "run", "error": None,
            "passed": passed, "total": len(tests), "tests": results}


def run_stdin(code: str, tests: list) -> dict:
    try:
        compiled = compile(code, "<solution>", "exec")
    except BaseException as e:  # noqa: BLE001  (syntax error etc.)
        return {"ok": False, "stage": "build", "error": repr(e),
                "passed": 0, "total": len(tests), "tests": []}

    results, passed = [], 0
    for i, t in enumerate(tests):
        inp = t.get("input", "") or ""
        exp = t.get("output", "") or ""
        rec = {"test": f"stdin[{i}] {inp[:40]!r}", "passed": False, "error": None}
        sout = io.StringIO()
        old_in, old_out = sys.stdin, sys.stdout
        err = None
        try:
            sys.stdin, sys.stdout = io.StringIO(inp), sout
            exec(compiled, {"__name__": "__main__"})
        except BaseException as e:  # noqa: BLE001
            err = repr(e)
        finally:
            sys.stdin, sys.stdout = old_in, old_out

        if err is not None:
            rec["error"] = err
        elif _cmp(sout.getvalue(), exp):
            rec["passed"] = True
            passed += 1
        else:
            rec["error"] = f"want {_norm(exp)[:60]!r} got {_norm(sout.getvalue())[:60]!r}"
        results.append(rec)
    return {"ok": True, "stage": "run", "error": None,
            "passed": passed, "total": len(tests), "tests": results}


def main() -> None:
    try:
        job = json.loads(sys.stdin.read())
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "stage": "decode", "error": repr(e),
                          "passed": 0, "total": 0, "tests": []}))
        return

    code = job.get("code", "")
    tests = job.get("tests", []) or []
    mode = job.get("mode", "assert")
    setup = job.get("setup", "") or ""

    out = run_stdin(code, tests) if mode == "stdin_stdout" else run_assert(code, setup, tests)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
