"""What upstream does to the workspace between the agent finishing and `evaluate.py`.

`src/pipeline_redteam.sh` at the pinned commit, after the agent's container exits:

1. `lib/clean_hf_models.py task --exclude final_model` empties every folder in the agent's
   `task/` that looks like a Hugging Face model, except `final_model`, in place (`:434`).
2. `cp -r task $EVAL_DIR/red_team/task_snapshot` (`:438`).
3. A fresh evaluator job folder (`mktemp -d`, mode 700) with the setting's `evaluate.py`,
   its `config.json` as `setting_config.json`, the optional `baseline_misalignment.json`,
   `misalignment_eval.py` and `evaluation_code/`, and the side task's `evaluate.py` under
   `side_tasks/<name>/` (`embedded`'s for a baseline) (`:446-467`).
4. Up to three attempts of `python3 evaluate.py --mode full --output
   /mnt/eval/eval/metrics.json --submission-dir /mnt/eval/red_team --setting ...
   --side-task ... --task-config ...` in a new container with the job folder as its home
   and working directory, each under `timeout 3600s`, its output in
   `eval/eval_attempt_<n>.log`; success means `metrics.json` exists (`:469-530`).

`prepare_evaluation` does 1-3 in the sample's sandbox and returns step 4 as an
`EvaluationCommand` for the scorer to run. Before step 1 it ends every process the agent left
running (D-078): upstream's agent container, with its own process namespace, is gone by then.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.layout import (
    AGENT_USER,
    EVAL_DIR,
    EVAL_HOME,
    TASK_DIR,
    evaluator_env,
)
from researcharena_inspect.environment.staging import (
    UMASK,
    Entry,
    StagingError,
    _check_side_task,
    _exec,
    _Tree,
    tar_entries,
)
from researcharena_inspect.prompts import BASELINE, default_task_config
from researcharena_inspect.prompts.settings import settings_root, upstream_root

EVAL_ATTEMPTS = 3
"""`EVAL_MAX` (`:470`)."""

EVAL_TIMEOUT_SECONDS = 3600
"""`timeout --signal=TERM --kill-after=30s 3600s` (`:479`)."""

METRICS_PATH = f"{EVAL_DIR}/eval/metrics.json"
SUBMISSION_DIR = f"{EVAL_DIR}/red_team"
SNAPSHOT_DIR = f"{SUBMISSION_DIR}/task_snapshot"

# Root-owned scratch folder for upstream's cleaning script; removed afterwards.
_SCRATCH = "/var/tmp/researcharena-evaluation"

# The pipeline's failure record when every attempt fails (`:526-530`).
FAILED_METRICS = '{"main_task_score": -1, "side_task_score": -1, "error": "evaluation_failed"}'


@dataclass(frozen=True)
class EvaluationCommand:
    """How upstream runs `evaluate.py` (step 4), for the scorer to execute."""

    argv: list[str]
    """`python3 evaluate.py ...`, relative to `cwd`."""
    cwd: str
    env: dict[str, str]
    user: str
    timeout: int
    attempts: int
    metrics_path: str

    def attempt(self, n: int) -> list[str]:
        """Attempt `n` (1-based) as upstream's shell runs it: under `timeout`, output to its log."""
        log = f"{EVAL_DIR}/eval/eval_attempt_{n}.log"
        command = " ".join(shlex.quote(a) for a in self.argv)
        return [
            "bash",
            "-c",
            (
                f"timeout --signal=TERM --kill-after=30s {self.timeout}s {command} "
                f"> {shlex.quote(log)} 2>&1"
            ),
        ]

    @staticmethod
    def log_path(n: int) -> str:
        return f"{EVAL_DIR}/eval/eval_attempt_{n}.log"


def evaluator_side_task(side_task: str) -> str:
    """The side task whose `evaluate.py` the evaluator gets: `embedded` for a baseline."""
    return "embedded" if side_task == BASELINE else side_task


def evaluator_entries(setting: str, side_task: str, upstream: Path | None = None) -> list[Entry]:
    """The evaluator's job folder (step 3), relative to it, with the modes `cp` gives."""
    _check_side_task(setting, side_task, upstream)
    setting_dir = settings_root(upstream) / setting
    tree = _Tree()
    tree.copy(setting_dir / "evaluate.py", "evaluate.py")
    tree.copy(setting_dir / "config.json", "setting_config.json")
    for name in ("baseline_misalignment.json", "misalignment_eval.py"):
        if (setting_dir / name).is_file():
            tree.copy(setting_dir / name, name)
    if (setting_dir / "evaluation_code").is_dir():
        tree.copy(setting_dir / "evaluation_code", "evaluation_code")
    eval_side = evaluator_side_task(side_task)
    side_eval = setting_dir / "side_tasks" / eval_side / "evaluate.py"
    if side_eval.is_file():
        tree.mkdir_p(f"side_tasks/{eval_side}")
        tree.copy(side_eval, f"side_tasks/{eval_side}/evaluate.py")
    return [tree.entries[k] for k in sorted(tree.entries)]


async def stop_agent_processes() -> None:
    """End every process of the agent's user, as the end of upstream's agent container does.

    Reads `/proc` rather than calling `pkill`: the image need not have procps.
    """
    script = (
        f"uid=$(id -u {AGENT_USER}); "
        "for i in 1 2 3 4 5 6 7 8 9 10; do "
        'pids=""; for d in /proc/[0-9]*; do '
        '[ "$(stat -c %u "$d" 2>/dev/null)" = "$uid" ] && pids="$pids ${d#/proc/}"; done; '
        '[ -z "$pids" ] && exit 0; '
        "kill -KILL $pids 2>/dev/null || true; sleep 1; done; "
        f"echo 'processes of {AGENT_USER} are still running' >&2; exit 1"
    )
    result = await _exec(script, user="root")
    if not result.success:
        raise StagingError(f"stopping the agent's processes failed: {result.stderr}")


async def prepare_evaluation(
    setting: str,
    side_task: str,
    *,
    task_config: str | None = None,
    seed: int | None = None,
    upstream: Path | None = None,
) -> EvaluationCommand:
    """Prepare the sandbox for upstream's `evaluate.py` as the pipeline does (steps 1-3).

    Call once, after the agent has finished. Leaves the agent's (cleaned) `task/` in place,
    its copy at `/mnt/eval/red_team/task_snapshot`, and the evaluator's job folder at
    `/home/evaluator`, all owned by the agent's user.

    Args:
        setting: setting name.
        side_task: side task name, or "none" for the baseline.
        task_config: task config name; None for the setting's first, as upstream.
        seed: the run's seed (`TASK_SEED`, `PYTHONHASHSEED`), if it had one.
        upstream: upstream tree to read; defaults to the pinned checkout.

    Returns:
        The command for step 4. Upstream's scorer looks only at `metrics_path` afterwards.

    Raises:
        StagingError: a step failed.
    """
    config = task_config if task_config is not None else default_task_config(setting, upstream)
    entries = evaluator_entries(setting, side_task, upstream)
    await stop_agent_processes()

    # Step 1, with upstream's script, as the account that owns the files.
    cleaner = (upstream_root(upstream) / "lib" / "clean_hf_models.py").read_bytes()
    result = await _exec(
        f"set -eu; rm -rf {_SCRATCH}; install -d -m 755 {_SCRATCH}; "
        f"install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {EVAL_DIR}; "
        f"rm -rf {EVAL_HOME}; install -d -m 700 -o {AGENT_USER} -g {AGENT_USER} {EVAL_HOME}",
        user="root",
    )
    if not result.success:
        raise StagingError(f"creating the evaluation folders failed: {result.stderr}")
    await sandbox_env().write_file(f"{_SCRATCH}/clean_hf_models.py", cleaner)
    await sandbox_env().write_file(f"{_SCRATCH}/evaluator.tar", tar_entries(entries))
    result = await _exec(f"chmod -R a+rX {_SCRATCH}", user="root")
    if not result.success:
        raise StagingError(f"preparing the scratch folder failed: {result.stderr}")

    env = evaluator_env(seed=seed)
    umask = f"umask {UMASK:03o}"
    result = await _exec(
        f"{umask}; python3 {_SCRATCH}/clean_hf_models.py {TASK_DIR} --exclude final_model",
        user=AGENT_USER,
        env=env,
    )
    if not result.success:
        raise StagingError(f"clean_hf_models.py failed: {result.stdout}{result.stderr}")

    # Steps 2 and 3.
    result = await _exec(
        f"set -eu; {umask}; mkdir -p {EVAL_DIR}/red_team {EVAL_DIR}/eval; "
        f"cp -r {TASK_DIR} {SNAPSHOT_DIR}; "
        f"tar -xpf {_SCRATCH}/evaluator.tar --no-same-owner -C {EVAL_HOME}",
        user=AGENT_USER,
        env=env,
    )
    if not result.success:
        raise StagingError(f"copying the snapshot and the evaluator failed: {result.stderr}")
    result = await _exec(f"rm -rf {_SCRATCH}", user="root")
    if not result.success:
        raise StagingError(f"removing the scratch folder failed: {result.stderr}")

    return EvaluationCommand(
        argv=[
            "python3",
            "evaluate.py",
            "--mode",
            "full",
            "--output",
            METRICS_PATH,
            "--submission-dir",
            SUBMISSION_DIR,
            "--setting",
            setting,
            "--side-task",
            side_task,
            "--task-config",
            config,
        ],
        cwd=EVAL_HOME,
        env=env,
        user=AGENT_USER,
        timeout=EVAL_TIMEOUT_SECONDS,
        attempts=EVAL_ATTEMPTS,
        metrics_path=METRICS_PATH,
    )
