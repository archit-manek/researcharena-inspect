"""Part D: which shipped CUDA kernel faults on the untouched baseline workspace, and how often.

On Hawk's L4, upstream's `evaluate.py` on the untouched staged baseline built the four shipped
kernels and then failed in the agent benchmark with `CUDA error: an illegal memory access was
encountered` (work package 2, one run). This task records upstream's behaviour; it changes no
upstream file and fixes nothing. Two samples, each in its own sandbox:

- `plain-repeats`: the baseline workspace staged as the pipeline stages it, then the pipeline's
  evaluation (`prepare_evaluation` and the scorer's attempt loop, `run_evaluation`) `repeats`
  times, the evaluation folders cleared before each.
- `kernel-isolation`: a copy of the staged `task/` in `/tmp`; the extension built once with
  `bench.compile_kernels`'s arguments; then upstream's own model loading, injection and
  generation benchmark (`bench.py`, `inject.py`) with each kernel alone, none and all, twice; all
  four after a vanilla model as `run_benchmark` orders them; each kernel and all four again with
  `CUDA_LAUNCH_BLOCKING=1` and a synchronize after every kernel call (which names the faulting
  call and records the shapes it got); `compute-sanitizer` on the kernels that faulted, if the
  image has it; and `bash bench.sh` in the real workspace as the agent would run it.

Every command's exit status, output and duration go to the sample's metadata under
`kernel_crash_check`, unparsed apart from `kernel_bench.py`'s own result line. Steps that would
run past the sample's time limit are skipped and listed.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec, sandbox

from researcharena_inspect.environment.evaluation import prepare_evaluation
from researcharena_inspect.environment.layout import (
    AGENT_USER,
    EVAL_DIR,
    PRE_AGENT_SETUP,
    TASK_DIR,
    agent_env,
    evaluator_env,
)
from researcharena_inspect.environment.sandbox import Backend, GpuName, sandbox_spec
from researcharena_inspect.environment.staging import stage_workspace
from researcharena_inspect.prompts import format_hours, resume_agent
from researcharena_inspect.scoring.kernel_bench import KERNELS, RESULT_PREFIX
from researcharena_inspect.scoring.scorer import run_evaluation

EVAL_VERSION = "1-A"  # side branch: inspect-evals-utils 0.2.0 has no metadata

METADATA_KEY = "kernel_crash_check"
PARTS = ("plain-repeats", "kernel-isolation")

WORK_DIR = "/tmp/kernel_check"
SCRIPT = f"{WORK_DIR}/kernel_bench.py"
COPY_DIR = f"{WORK_DIR}/task"
BUILD_DIR = f"{WORK_DIR}/build"

SETTING = "cuda_kernel"
SIDE_TASK = "none"
REDRUNTIME = "1"

MARGIN_SECONDS = 150
"""Time left at the end of the sample for writing the output."""
MIN_STEP_SECONDS = 90
"""A step is skipped rather than started with less time than this."""

VERSION_PROBE = (
    "import json, torch, transformers; print(json.dumps({'torch': torch.__version__, "
    "'cuda': torch.version.cuda, 'transformers': transformers.__version__, "
    "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)


def bench_runs() -> list[dict[str, Any]]:
    """The kernel-isolation runs in order: label, kernel selection and options."""
    subsets = ["none", *KERNELS, "all"]
    runs: list[dict[str, Any]] = []
    for repeat in (1, 2):
        runs += [{"label": f"{k} (run {repeat})", "kernels": k} for k in subsets]
    runs.append({"label": "all after vanilla", "kernels": "all", "flags": ["--vanilla-first"]})
    runs += [
        {
            "label": f"{k} (CUDA_LAUNCH_BLOCKING=1, sync each)",
            "kernels": k,
            "flags": ["--sync-each"],
            "env": {"CUDA_LAUNCH_BLOCKING": "1"},
        }
        for k in [*KERNELS, "all"]
    ]
    return runs


def bench_result(stdout: str) -> dict[str, Any] | None:
    """`kernel_bench.py`'s last `KERNEL_BENCH_RESULT=` line, parsed."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
    return None


class _Steps:
    """Runs commands in the sandbox, keeps their output and skips what would overrun."""

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.log: list[dict[str, Any]] = []
        self.skipped: list[str] = []

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    async def run(
        self,
        name: str,
        cmd: list[str],
        *,
        user: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        limit: int = 600,
    ) -> dict[str, Any] | None:
        timeout = int(min(limit, self.remaining()))
        if timeout < MIN_STEP_SECONDS:
            self.skipped.append(name)
            return None
        start = time.monotonic()
        try:
            result = await sandbox().exec(cmd, user=user, cwd=cwd, env=env or {}, timeout=timeout)
            entry: dict[str, Any] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except TimeoutError as error:
            entry = {"returncode": None, "stdout": "", "stderr": f"TimeoutError: {error}"}
        entry = {"step": name, "cmd": cmd, "user": user or "root", "cwd": cwd, **entry}
        entry["timeout"] = timeout
        entry["seconds"] = round(time.monotonic() - start, 1)
        self.log.append(entry)
        return entry


async def _common(steps: _Steps, result: dict[str, Any]) -> dict[str, str]:
    start = time.monotonic()
    staged = await stage_workspace(SETTING, SIDE_TASK, redruntime=REDRUNTIME)
    result["staging"] = {
        "seconds": round(time.monotonic() - start, 1),
        "hooks": [
            {"hook": hook, "returncode": code, "output": output}
            for hook, code, output in staged.hook_outputs
        ],
    }
    env = agent_env(redruntime=format_hours(REDRUNTIME), resume_agent=resume_agent(SETTING))
    await steps.run(
        "nvidia-smi query",
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
    )
    await steps.run("node size", ["sh", "-c", "nproc; grep MemTotal /proc/meminfo"])
    await steps.run(
        "versions", ["python3", "-c", VERSION_PROBE], user=AGENT_USER, cwd=TASK_DIR, env=env
    )
    return env


async def plain_repeats(steps: _Steps, result: dict[str, Any], repeats: int) -> None:
    """The pipeline's evaluation of the untouched workspace, `repeats` times."""
    runs: list[dict[str, Any]] = []
    result["repeats"] = runs
    for n in range(1, repeats + 1):
        # One attempt takes about 250 s on an L4; up to three are allowed.
        if steps.remaining() < 3 * 400:
            steps.skipped.append(f"plain evaluation {n}")
            continue
        cleared = await steps.run(
            f"clear evaluation folders ({n})",
            ["sh", "-c", f"rm -rf {EVAL_DIR}/red_team {EVAL_DIR}/eval"],
        )
        if cleared is None or cleared["returncode"] != 0:
            break
        start = time.monotonic()
        command = await prepare_evaluation(SETTING, SIDE_TASK)
        prepared = round(time.monotonic() - start, 1)
        run = await run_evaluation(command)
        runs.append(
            {
                "repeat": n,
                "prepare_seconds": prepared,
                "metrics_json": run.metrics_json,
                "evaluation_failed": run.evaluation_failed,
                "attempts": [
                    {
                        "number": a.number,
                        "returncode": a.returncode,
                        "seconds": a.seconds,
                        "wrote_metrics": a.wrote_metrics,
                        "log": a.log,
                    }
                    for a in run.attempts
                ],
            }
        )


def _script(*args: str) -> list[str]:
    return ["python3", SCRIPT, *args, "--task", COPY_DIR, "--build-dir", BUILD_DIR]


async def kernel_isolation(steps: _Steps, result: dict[str, Any], env: dict[str, str]) -> None:
    """Each kernel alone, none and all; then the faulting ones under a sanitizer; bench.sh."""
    script = (Path(__file__).parent / "kernel_bench.py").read_bytes()
    made = await steps.run(
        "work folder",
        [
            "sh",
            "-c",
            f"rm -rf {WORK_DIR} && install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {WORK_DIR}",
        ],
    )
    if made is None or made["returncode"] != 0:
        return
    await sandbox().write_file(SCRIPT, script)
    # evaluate.py's variables, with a home that exists (no evaluator folder in this sample).
    eval_env = evaluator_env() | {"HOME": WORK_DIR}
    await steps.run("script readable", ["chmod", "644", SCRIPT])
    copied = await steps.run(
        "diagnostic copy of task/", ["cp", "-r", TASK_DIR, COPY_DIR], user=AGENT_USER
    )
    if copied is None or copied["returncode"] != 0:
        return
    await steps.run(
        "sanitizer available",
        ["sh", "-c", "command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer"],
        user=AGENT_USER,
        env=eval_env,
    )
    built = await steps.run(
        "build", _script("build"), user=AGENT_USER, cwd=WORK_DIR, env=eval_env, limit=1800
    )
    result["build"] = bench_result(built["stdout"]) if built else None
    runs: list[dict[str, Any]] = []
    result["bench_runs"] = runs
    if not built or built["returncode"] != 0:
        return
    for run in bench_runs():
        entry = await steps.run(
            run["label"],
            _script("bench", "--kernels", run["kernels"], *run.get("flags", [])),
            user=AGENT_USER,
            cwd=WORK_DIR,
            env=eval_env | run.get("env", {}),
            limit=600,
        )
        runs.append(
            {
                **run,
                "returncode": entry["returncode"] if entry else None,
                "result": bench_result(entry["stdout"]) if entry else None,
            }
        )

    # The agent's own measurement, as its shell runs it (solve.sh sources the setup first).
    await steps.run(
        "bench.sh as the agent",
        ["bash", "-c", f"source {PRE_AGENT_SETUP} > /dev/null 2>&1; bash bench.sh"],
        user=AGENT_USER,
        cwd=TASK_DIR,
        env=env,
        limit=1500,
    )
    await steps.run(
        "workspace after bench.sh",
        ["sh", "-c", "ls -la workspace; cat workspace/last_bench.json"],
        user=AGENT_USER,
        cwd=TASK_DIR,
    )

    # The sanitizer on the kernels that faulted alone (or all four if none did alone).
    blocking = [r for r in runs if "CUDA_LAUNCH_BLOCKING" in r.get("env", {})]
    faulted = [
        r["kernels"] for r in blocking if r["result"] and r["result"].get("status") == "error"
    ]
    targets = [k for k in faulted if k != "all"][:2] or (["all"] if "all" in faulted else [])
    sanitizer_runs: list[dict[str, Any]] = []
    result["sanitizer_runs"] = sanitizer_runs
    for kernels in targets:
        entry = await steps.run(
            f"compute-sanitizer memcheck: {kernels}",
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--print-limit",
                "20",
                *_script(
                    "bench",
                    "--kernels",
                    kernels,
                    "--sync-each",
                    "--prompts",
                    "1",
                    "--max-new-tokens",
                    "4",
                    "--warmup",
                    "0",
                ),
            ],
            user=AGENT_USER,
            cwd=WORK_DIR,
            env=eval_env,
            limit=900,
        )
        sanitizer_runs.append(
            {
                "kernels": kernels,
                "returncode": entry["returncode"] if entry else None,
                "result": bench_result(entry["stdout"]) if entry else None,
            }
        )


@solver
def run_crash_check(repeats: int = 3, time_limit: int = 3600) -> Solver:
    """Run the sample's part (see the module docstring) and keep every output."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        steps = _Steps(deadline=time.monotonic() + time_limit - MARGIN_SECONDS)
        part = state.metadata["part"]
        result: dict[str, Any] = {"part": part, "steps": steps.log, "skipped": steps.skipped}
        state.metadata[METADATA_KEY] = result
        env = await _common(steps, result)
        if part == "plain-repeats":
            await plain_repeats(steps, result, repeats)
        else:
            await kernel_isolation(steps, result, env)
        summary = summarize(result)
        result["summary"] = summary
        state.output = ModelOutput.from_content(model="none", content=json.dumps(summary))
        return state

    return solve


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """What each evaluation or benchmark run ended with, for the score."""
    out: dict[str, Any] = {"part": result["part"], "skipped": list(result.get("skipped", []))}
    if result["part"] == "plain-repeats":
        out["metrics"] = [r["metrics_json"] for r in result.get("repeats", [])]
        out["illegal_memory_access"] = [
            any("illegal memory access" in a["log"] for a in r["attempts"])
            for r in result.get("repeats", [])
        ]
        return out
    runs = []
    for run in [*result.get("bench_runs", []), *result.get("sanitizer_runs", [])]:
        res = run.get("result") or {}
        bench = res.get("benchmark") or {}
        runs.append(
            {
                "label": run.get("label", f"sanitizer: {run.get('kernels')}"),
                "status": res.get("status"),
                "tokens_per_sec": bench.get("avg_tokens_per_sec"),
                "fault": res.get("fault"),
                "error": res.get("error"),
            }
        )
    out["build"] = (result.get("build") or {}).get("compile_status")
    out["runs"] = runs
    return out


@scorer(metrics={"faults": [mean()], "runs": [mean()], "skipped": [mean()]})
def crash_check_summary() -> Scorer:
    """Counts only: how many evaluations or benchmark runs faulted, ran and were skipped."""

    async def score(state: TaskState, target: Target) -> Score:
        summary = state.metadata.get(METADATA_KEY, {}).get("summary", {})
        if summary.get("part") == "plain-repeats":
            faults = sum(summary.get("illegal_memory_access", []))
            runs = len(summary.get("metrics", []))
        else:
            faults = sum(r["status"] == "error" for r in summary.get("runs", []))
            runs = len(summary.get("runs", []))
        return Score(
            value={"faults": faults, "runs": runs, "skipped": len(summary.get("skipped", []))},
            answer=json.dumps(summary),
        )

    return score


@task
def kernel_crash_check(
    backend: Backend = "k8s",
    gpu: GpuName | None = "l4",
    parts: str | list[str] | None = None,
    repeats: int = 3,
    sandbox: SandboxEnvironmentSpec | None = None,
    sandbox_start_timeout: int | None = None,
    time_limit: int = 3600,
) -> Task:
    """Find which shipped kernel faults on the untouched baseline workspace (part D).

    Args:
        backend: "k8s" (for example Hawk) or "docker" (one GPU machine).
        gpu: GPU model to ask Kubernetes for ("l4" or "h100"); must be None with Docker.
        parts: "plain-repeats", "kernel-isolation" or a list; None for both, one sample each.
        repeats: how many times `plain-repeats` runs the pipeline's evaluation.
        sandbox: overrides `backend` and `gpu` (for example a CPU sandbox in tests).
        sandbox_start_timeout: seconds to wait for the sandbox to start (sets the Kubernetes
            sandbox's `INSPECT_HELM_TIMEOUT`); the sample's time limit starts only after it.
        time_limit: each sample's time limit in seconds; steps that would overrun it are skipped.

    Returns:
        A task whose samples' metadata hold every command's raw output.
    """
    if sandbox_start_timeout is not None:
        os.environ["INSPECT_HELM_TIMEOUT"] = str(sandbox_start_timeout)
    names = list(PARTS) if parts is None else [parts] if isinstance(parts, str) else parts
    unknown = sorted(set(names) - set(PARTS))
    if unknown:
        raise ValueError(f"unknown parts {unknown}; choose from {list(PARTS)}")
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id=part,
                    input=f"Part D: {part}.",
                    metadata={"part": part, "setting": SETTING, "side_task": SIDE_TASK},
                )
                for part in names
            ]
        ),
        solver=run_crash_check(repeats=repeats, time_limit=time_limit),
        scorer=crash_check_summary(),
        sandbox=sandbox or sandbox_spec(backend, None if backend == "docker" else gpu),
        time_limit=time_limit,
        version=EVAL_VERSION,
    )
