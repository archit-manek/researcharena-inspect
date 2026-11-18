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

`prepare_evaluation` does 1-3 and returns step 4 as an `EvaluationCommand` for the scorer to
run. Before step 1 it ends every process the agent left running: upstream's agent container,
with its own process namespace, is gone by then.

Where (D-093). On Docker the sandbox has a second service, `EVALUATOR_SERVICE`, of the same
image with the GPU, which shares only `HF_HOME` and `EVAL_DIR` with the agent's
(`sandbox.with_evaluator`), as upstream's evaluator shares only the cache overlay and the run
folder. Steps 1 and 2 run in the agent's container (upstream runs them on the host, which sees
the agent's `task/`), the snapshot landing on the shared `EVAL_DIR`; then the agent's container
is stopped, with every process in it, and step 3, step 4 and everything after it run in the
evaluator's (`evaluator_sandbox`), with the job folder at upstream's `/home/user`. On
Kubernetes, where a second pod cannot share the GPU and these folders, there is no such
service: everything runs in the agent's container, the job folder at `SHARED_EVAL_HOME`.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

import anyio
from inspect_ai.util import ExecResult, SandboxEnvironment, SandboxUnavailableError
from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.layout import (
    AGENT_USER,
    EVAL_DIR,
    EVAL_HOME,
    EVALUATOR_SERVICE,
    SHARED_EVAL_HOME,
    TASK_DIR,
    evaluator_env,
)
from researcharena_inspect.environment.staging import (
    ACCOUNT_SETUP,
    UMASK,
    Entry,
    StagingError,
    _check_side_task,
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

# Root-owned scratch folder for upstream's cleaning script and the evaluator's files; removed
# afterwards.
_SCRATCH = "/var/tmp/researcharena-evaluation"

STOP_SECONDS = 60
"""How long `stop_agent_container` waits for the agent's container to exit."""

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


def evaluator_sandbox() -> SandboxEnvironment:
    """Where upstream's evaluator runs: its own container on Docker, the agent's on Kubernetes."""
    try:
        return sandbox_env(EVALUATOR_SERVICE)
    except ValueError:  # no such service: Kubernetes (D-093)
        return sandbox_env()


def evaluator_service() -> str:
    """The service `evaluator_sandbox()` is: `EVALUATOR_SERVICE`, or "default" (the agent's)."""
    return EVALUATOR_SERVICE if evaluator_sandbox() is not sandbox_env() else "default"


async def _exec(
    box: SandboxEnvironment, cmd: str, *, user: str, env: dict[str, str] | None = None
) -> ExecResult[str]:
    return await box.exec(["bash", "-c", cmd], user=user, env=env or {})


async def stop_agent_processes(box: SandboxEnvironment | None = None) -> None:
    """End every process of the agent's user, as the end of upstream's agent container does.

    In `box`, the agent's sandbox if None. Reads `/proc` rather than calling `pkill`: the image
    need not have procps.
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
    result = await _exec(box or sandbox_env(), script, user="root")
    if not result.success:
        raise StagingError(f"stopping the agent's processes failed: {result.stderr}")


async def stop_agent_container(box: SandboxEnvironment) -> None:
    """Stop the agent's container with every process in it, as upstream's has exited (D-093).

    SIGTERM to its init (`init: true` in the Compose service), which ends its child and exits;
    the container then stops and later calls to it raise `SandboxUnavailableError`.

    Raises:
        StagingError: it was still running, or a call to it had not returned, after
            `STOP_SECONDS` (one limit for the signal and the checks).
    """
    try:
        with anyio.fail_after(STOP_SECONDS):
            await box.exec(["bash", "-c", "kill -TERM 1"], user="root")
            while True:
                await anyio.sleep(1)
                await box.exec(["true"], user="root")
    except SandboxUnavailableError:
        return
    except TimeoutError:
        raise StagingError(
            f"the agent's container was still running after {STOP_SECONDS} s"
        ) from None


async def prepare_evaluation(
    setting: str,
    side_task: str,
    *,
    task_config: str | None = None,
    seed: int | None = None,
    upstream: Path | None = None,
) -> EvaluationCommand:
    """Prepare upstream's `evaluate.py` as the pipeline does (steps 1-3).

    Call once, after the agent has finished. Leaves the agent's (cleaned) `task/` in place, its
    copy at `/mnt/eval/red_team/task_snapshot`, and the evaluator's job folder at `EVAL_HOME` in
    the evaluator's container, the agent's container stopped (Docker); or at `SHARED_EVAL_HOME`
    in the agent's container (Kubernetes). All of it belongs to the agent's user, as upstream's
    belongs to the job's account.

    Args:
        setting: setting name.
        side_task: side task name, or "none" for the baseline.
        task_config: task config name; None for the setting's first, as upstream.
        seed: the run's seed (`TASK_SEED`, `PYTHONHASHSEED`), if it had one.
        upstream: upstream tree to read; defaults to the pinned checkout.

    Returns:
        The command for step 4, to run in `evaluator_sandbox()`. Upstream's scorer looks only
        at `metrics_path` afterwards.

    Raises:
        StagingError: a step failed.
    """
    config = task_config if task_config is not None else default_task_config(setting, upstream)
    entries = evaluator_entries(setting, side_task, upstream)
    agent, evaluator = sandbox_env(), evaluator_sandbox()
    separate = evaluator is not agent
    home = EVAL_HOME if separate else SHARED_EVAL_HOME
    await stop_agent_processes(agent)

    # Step 1, with upstream's script, as the account that owns the files, in the agent's
    # container (D-251). `EVAL_DIR` stays root's until the evaluator's container takes it over:
    # only `red_team/` is the agent's.
    cleaner = (upstream_root(upstream) / "lib" / "clean_hf_models.py").read_bytes()
    result = await _exec(
        agent,
        f"set -eu; rm -rf {_SCRATCH}; install -d -m 755 {_SCRATCH} {EVAL_DIR}; "
        f"install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {SUBMISSION_DIR}",
        user="root",
    )
    if not result.success:
        raise StagingError(f"creating the evaluation folders failed: {result.stderr}")
    await agent.write_file(f"{_SCRATCH}/clean_hf_models.py", cleaner)
    result = await _exec(agent, f"chmod -R a+rX {_SCRATCH}", user="root")
    if not result.success:
        raise StagingError(f"preparing the scratch folder failed: {result.stderr}")

    env = evaluator_env(seed=seed, home=home)
    umask = f"umask {UMASK:03o}"
    result = await _exec(
        agent,
        # `-s`: not the agent's Python user site, which `HOME=/home/user` would load (D-251).
        f"{umask}; python3 -s {_SCRATCH}/clean_hf_models.py {TASK_DIR} --exclude final_model",
        user=AGENT_USER,
        env=env,
    )
    if not result.success:
        raise StagingError(f"clean_hf_models.py failed: {result.stdout}{result.stderr}")

    # Step 2. `cp -r` gives the copies the time they are made, as upstream's.
    result = await _exec(
        agent, f"set -eu; {umask}; cp -r {TASK_DIR} {SNAPSHOT_DIR}", user=AGENT_USER, env=env
    )
    if not result.success:
        raise StagingError(f"copying the snapshot failed: {result.stderr}")
    result = await _exec(agent, f"rm -rf {_SCRATCH}", user="root")
    if not result.success:
        raise StagingError(f"removing the scratch folder failed: {result.stderr}")
    if separate:  # D-093: the agent's container ends here, as upstream's has
        await stop_agent_container(agent)

    # Step 3, in the evaluator's container: the account (as staging makes it), its job folder
    # (in a fresh one, holding only the cache's mount point) and `EVAL_DIR/eval`.
    reset = "" if separate else f"rm -rf {home}; "
    result = await _exec(
        evaluator,
        f"set -eu; {ACCOUNT_SETUP}; rm -rf {_SCRATCH}; install -d -m 755 {_SCRATCH}; "
        f"{reset}install -d -m 700 -o {AGENT_USER} -g {AGENT_USER} {home}; "
        f"chown {AGENT_USER}:{AGENT_USER} {EVAL_DIR}; "
        f"install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {EVAL_DIR}/eval",
        user="root",
    )
    if not result.success:
        raise StagingError(f"creating the evaluator's folders failed: {result.stderr}")
    await evaluator.write_file(f"{_SCRATCH}/evaluator.tar", tar_entries(entries))
    result = await _exec(evaluator, f"chmod -R a+rX {_SCRATCH}", user="root")
    if not result.success:
        raise StagingError(f"preparing the scratch folder failed: {result.stderr}")
    # `tar -m` gives the files the time they are unpacked, as upstream's `cp`.
    result = await _exec(
        evaluator,
        f"{umask}; tar -xpmf {_SCRATCH}/evaluator.tar --no-same-owner -C {home}",
        user=AGENT_USER,
        env=env,
    )
    if not result.success:
        raise StagingError(f"copying the evaluator's files failed: {result.stderr}")
    result = await _exec(evaluator, f"rm -rf {_SCRATCH}", user="root")
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
        cwd=home,
        env=env,
        user=AGENT_USER,
        timeout=EVAL_TIMEOUT_SECONDS,
        attempts=EVAL_ATTEMPTS,
        metrics_path=METRICS_PATH,
    )
