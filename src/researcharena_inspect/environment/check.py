"""Environment check on a GPU, with no agent and no model calls (pass rule 4 of work package 2).

In the eval's image and sandbox it records:

1. `nvidia-smi` (full output and the name, memory, driver and compute capability line), the
   CPU count, memory and disk the container sees;
2. what PyTorch sees of the GPU, as the agent's user with the agent's environment;
3. the agent's workspace staged for the setting and side task, as upstream stages it;
4. upstream's `pre_agent_setup.sh`, sourced as the agent's shell sources it;
5. upstream's build step, `python3 compile.py --check --verbose` in the agent's `task/`
   (`compile.py` writes only to `/tmp`, so the workspace stays as staged; the check records
   `git status` before and after);
6. upstream's `evaluate.py`, run once as the pipeline runs it, on the untouched workspace.

Every command's exit status, output and duration go to the sample's metadata under
`environment_check`, unparsed; `evaluate.py`'s log and `metrics.json` are kept as raw text.
The score says only whether `evaluate.py` wrote `metrics.json` (upstream's own test for a
finished evaluation); judging the numbers is the scorers' job.
"""

import time
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_evals_utils.metadata import load_eval_version

from researcharena_inspect.environment.evaluation import prepare_evaluation
from researcharena_inspect.environment.layout import (
    AGENT_USER,
    PRE_AGENT_SETUP,
    TASK_DIR,
    agent_env,
)
from researcharena_inspect.environment.sandbox import sandbox_spec
from researcharena_inspect.environment.staging import stage_workspace
from researcharena_inspect.prompts import format_hours, resume_agent

# eval.yaml sits in the package folder, one above this module.
EVAL_VERSION = load_eval_version(Path(__file__).parent.parent / "eval.yaml")

CHECK_METADATA_KEY = "environment_check"

# Upstream's own build step for the kernels: compile.py builds bindings.cpp and kernels/*.cu
# into one PyTorch extension under /tmp and reports COMPILE_STATUS.
BUILD_COMMAND = ["python3", "compile.py", "--check", "--verbose"]

TORCH_PROBE = (
    "import json, torch; ok = torch.cuda.is_available(); print(json.dumps({"
    "'torch': torch.__version__, 'torch.version.cuda': torch.version.cuda, "
    "'cuda_available': ok, 'device_count': torch.cuda.device_count(), "
    "'device_name': torch.cuda.get_device_name(0) if ok else None, "
    "'capability': torch.cuda.get_device_capability(0) if ok else None, "
    "'arch_list': torch.cuda.get_arch_list()}))"
)


async def _record(
    log: list[dict[str, Any]],
    name: str,
    cmd: list[str],
    *,
    user: str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    limit: int = 600,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        result = await sandbox().exec(cmd, user=user, cwd=cwd, env=env or {}, timeout=limit)
        entry: dict[str, Any] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except TimeoutError as error:
        entry = {"returncode": None, "stdout": "", "stderr": f"TimeoutError: {error}"}
    entry = {"step": name, "cmd": cmd, "user": user or "root", "cwd": cwd, **entry}
    entry["seconds"] = round(time.monotonic() - start, 1)
    log.append(entry)
    return entry


@solver
def run_environment_check(
    setting: str = "cuda_kernel",
    side_task: str = "none",
    redruntime: str = "1",
    build: bool = True,
    evaluate: bool = True,
    evaluate_timeout: int = 2700,
) -> Solver:
    """Run the checks in the module docstring and keep every output in the metadata."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        log: list[dict[str, Any]] = []
        result: dict[str, Any] = {"steps": log}
        state.metadata[CHECK_METADATA_KEY] = result
        env = agent_env(redruntime=format_hours(redruntime), resume_agent=resume_agent(setting))

        await _record(log, "nvidia-smi", ["nvidia-smi"])
        await _record(
            log,
            "nvidia-smi query",
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
        )
        await _record(
            log,
            "node size",
            ["sh", "-c", "nproc; grep MemTotal /proc/meminfo; df -h / /tmp; uname -a"],
        )

        start = time.monotonic()
        staged = await stage_workspace(setting, side_task, redruntime=redruntime)
        result["staging"] = {
            "seconds": round(time.monotonic() - start, 1),
            "hooks": [
                {"hook": hook, "returncode": code, "output": output}
                for hook, code, output in staged.hook_outputs
            ],
        }
        await _record(
            log,
            "staged home",
            [
                "sh",
                "-c",
                (
                    "cd /home/user && find . -path ./task/.git -prune -o "
                    "-printf '%M %u %s %p\\n' | sort -k4; "
                    "git -C task log --format='%H %an <%ae> %s' && git -C task status --porcelain"
                ),
            ],
            user=AGENT_USER,
            env=env,
        )
        await _record(
            log, "torch", ["python3", "-c", TORCH_PROBE], user=AGENT_USER, cwd=TASK_DIR, env=env
        )
        await _record(
            log,
            "pre_agent_setup.sh",
            ["bash", "-c", f"source {PRE_AGENT_SETUP}"],
            user=AGENT_USER,
            cwd=TASK_DIR,
            env=env,
        )
        if build:
            await _record(
                log, "build", BUILD_COMMAND, user=AGENT_USER, cwd=TASK_DIR, env=env, limit=1800
            )
            await _record(
                log,
                "workspace after build",
                ["sh", "-c", "git status --porcelain --untracked-files=all; ls -la"],
                user=AGENT_USER,
                cwd=TASK_DIR,
                env=env,
            )
        if evaluate:
            command = await prepare_evaluation(setting, side_task)
            result["evaluate_command"] = command.argv
            await _record(
                log,
                "evaluate.py attempt 1",
                command.attempt(1),
                user=command.user,
                cwd=command.cwd,
                env=command.env,
                limit=evaluate_timeout,
            )
            log_file = await _record(log, "evaluate log", ["cat", command.log_path(1)])
            metrics = await _record(log, "metrics.json", ["cat", command.metrics_path])
            result["evaluate_log"] = log_file["stdout"]
            result["metrics_json"] = metrics["stdout"] if metrics["returncode"] == 0 else None
        state.output = ModelOutput.from_content(
            model="none", content=str(result.get("metrics_json") or "no metrics.json")
        )
        return state

    return solve


@scorer(metrics=[accuracy()])
def evaluation_finished() -> Scorer:
    """CORRECT when upstream's `evaluate.py` wrote `metrics.json`; the text is kept raw."""

    async def score(state: TaskState, target: Target) -> Score:
        check = state.metadata.get(CHECK_METADATA_KEY, {})
        metrics = check.get("metrics_json")
        return Score(
            value=CORRECT if metrics is not None else INCORRECT,
            answer=metrics or "",
            explanation="evaluate.py wrote metrics.json"
            if metrics is not None
            else "no metrics.json",
        )

    return score


@task
def environment_check(
    setting: str = "cuda_kernel",
    side_task: str = "none",
    build: bool = True,
    evaluate: bool = True,
    time_limit: int = 3600,
) -> Task:
    """Check the agent's environment on a GPU without an agent or a model call.

    Runs in the built-in Docker sandbox (`environment.sandbox_spec`) on the local machine's
    GPU, in one container (no evaluator's service).

    Args:
        setting: setting whose workspace is staged and evaluated.
        side_task: side task, or "none" for the baseline.
        build: run upstream's kernel build step.
        evaluate: run upstream's `evaluate.py` once on the untouched workspace.
        time_limit: the sample's time limit in seconds.

    Returns:
        A one-sample task whose metadata holds every command's raw output.
    """
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    id=f"{setting}-{side_task}",
                    input="Check the agent's environment.",
                    metadata={"setting": setting, "side_task": side_task},
                )
            ]
        ),
        solver=run_environment_check(
            setting=setting, side_task=side_task, build=build, evaluate=evaluate
        ),
        scorer=evaluation_finished(),
        sandbox=sandbox_spec(evaluator=False),
        time_limit=time_limit,
        version=EVAL_VERSION,
    )
