"""Async executor around the subprocess runner.

On POSIX we additionally apply RLIMITs (CPU seconds, address space, no core
dumps) via a preexec hook, and run the child in its own session so a timeout can
kill the whole process group. On Windows these are skipped (local dev only); the
real training / eval runs on Linux (for example RunPod) where the limits apply. For stronger
isolation in production, wrap `_runner` in nsjail / firejail.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union

IS_POSIX = os.name == "posix"
# Run the runner by absolute file path: it is self-contained (stdlib only), so
# this works regardless of the launch directory / PYTHONPATH.
_RUNNER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runner.py")


@dataclass
class TestRec:
    test: str                 # a short label for the test (assert text / stdin preview)
    passed: bool
    error: Optional[str] = None


@dataclass
class ExecResult:
    ran: bool                 # candidate code built (imported / compiled) without error
    passed: int               # number of tests that passed
    total: int                # number of tests
    timed_out: bool = False
    stage: str = "run"        # decode | build | run
    error: Optional[str] = None
    tests: List[TestRec] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def feedback(self, max_chars: int = 800) -> str:
        """Compact execution feedback for multi-turn self-debug (P3)."""
        if self.timed_out:
            return "Execution timed out."
        if not self.ran:
            return f"Code failed to run ({self.stage}): {self.error}"
        fails = [f"FAIL {r.test}\n  -> {r.error}" for r in self.tests if not r.passed]
        return ("\n".join(fails) or "All tests passed.")[:max_chars]


def _posix_preexec(cpu_s: int, mem_mb: int):
    import resource

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        nbytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()  # own session/group so we can kill the whole tree on timeout

    return _apply


async def execute_async(
    code: str,
    tests: Union[List[str], List[dict]],
    mode: str = "assert",          # "assert" | "stdin_stdout"
    setup: str = "",
    timeout: float = 10.0,
    cpu_s: int = 10,
    mem_mb: int = 1024,
) -> ExecResult:
    job = json.dumps({"code": code, "tests": tests, "mode": mode, "setup": setup})

    kwargs = {}
    if IS_POSIX:
        kwargs["preexec_fn"] = _posix_preexec(cpu_s, mem_mb)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, _RUNNER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )

    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(input=job.encode()), timeout=timeout
        )
    except asyncio.TimeoutError:
        try:
            if IS_POSIX:
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await proc.wait()  # reap the killed child so its pipes close cleanly
        except Exception:  # noqa: BLE001
            pass
        return ExecResult(ran=False, passed=0, total=len(tests),
                          timed_out=True, stage="run", error="timeout")

    out = stdout.decode(errors="replace").strip()
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return ExecResult(ran=False, passed=0, total=len(tests), stage="decode",
                          error=f"bad runner output: {e}; raw={out[:200]!r}")

    recs = [TestRec(test=r.get("test", ""), passed=r["passed"], error=r.get("error"))
            for r in data.get("tests", [])]
    return ExecResult(
        ran=bool(data.get("ok", False)),
        passed=int(data.get("passed", 0)),
        total=int(data.get("total", len(tests))),
        stage=data.get("stage", "run"),
        error=data.get("error"),
        tests=recs,
    )
