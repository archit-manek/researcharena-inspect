"""The agent's home, staged as upstream's `src/pipeline_redteam.sh` stages it.

Upstream fills `$AGENT_JOB_DIR` (the agent's `/home/user`) with `task/` (`:172-190`), the launch
helpers (`:193`, `:228-245`; no credentials, D-091), `task/timer.sh` (`:248-249`), `prompt.txt`
(`:265`) and the `pre_agent.sh` hooks' output (`:272-309`). Line numbers here are that file's.
The file copies are computed on the host (`home_entries`) with the modes `cp` gives; `timer.sh`
and the hooks run upstream's scripts in the sandbox as the agent's user. The model cache
(`model_cache`) is filled before `timer.sh`; the `pipeline_pre_agent.sh` hooks (`services`,
`:325-335`) run last.

A setup solver, not `Sample.files` or `Sample.setup`: `timer.sh` must be written when the run
starts, the hooks must run in the sandbox as the agent's user, and the files need exact modes
and owner.
"""

import io
import re
import shlex
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import ExecResult
from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.layout import (
    AGENT_HOME,
    AGENT_UID,
    AGENT_USER,
    HF_HOME,
    TASK_DIR,
)
from researcharena_inspect.environment.model_cache import ModelCacheMode, provision_models
from researcharena_inspect.environment.services import (
    endpoint_hosts,
    pin_endpoint_hosts,
    pipeline_hooks,
    start_pipeline_hooks,
)
from researcharena_inspect.prompts import (
    BASELINE,
    agent_instruction_file,
    default_task_config,
    format_hours,
    list_side_tasks,
)
from researcharena_inspect.prompts.settings import load_config, settings_root, upstream_root

AgentFamily = Literal["claude", "codex", "opencode"]

UMASK = 0o022
"""The pipeline's umask, assumed (a cluster default). `cp` and `mkdir` apply it."""

# `AGENT_VISIBLE_HINTS` (`pipeline_redteam.sh:239`); grep -i -E.
AGENT_VISIBLE_HINTS = re.compile(
    r"red.?team|blue.?team|monitor|side.?task|sabotag|arena|ai.?control|pipeline", re.IGNORECASE
)
LAUNCH_HELPERS = ("agent_solve.sh", "pre_agent_setup.sh", "update_agent_cli.sh", "check_effort.py")

# The identity the hooks' `git commit` records. Upstream's hooks run on the cluster host as
# the job's account, with that account's git identity, which the published data does not show.
GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": AGENT_USER,
    "GIT_AUTHOR_EMAIL": f"{AGENT_USER}@localhost",
    "GIT_COMMITTER_NAME": AGENT_USER,
    "GIT_COMMITTER_EMAIL": f"{AGENT_USER}@localhost",
}

# The agent's account; also made in the evaluator's container, as upstream's evaluator runs as
# the same account (D-093).
ACCOUNT_SETUP = (
    f"if ! id -u {AGENT_USER} >/dev/null 2>&1; then "
    f"groupadd --gid {AGENT_UID} {AGENT_USER} && "
    f"useradd --uid {AGENT_UID} --gid {AGENT_UID} --home-dir {AGENT_HOME} "
    f"--no-create-home --shell /bin/bash {AGENT_USER}; fi"
)

# Root-owned scratch folder for upstream's scripts while they run; removed before the agent
# starts. Outside the agent's home and /tmp, as upstream's ARENA_ROOT is outside the container.
STAGING_DIR = "/var/tmp/researcharena-staging"


class StagingError(RuntimeError):
    """Staging failed where upstream's pipeline would have stopped (`set -euo pipefail`)."""


@dataclass(frozen=True)
class Entry:
    """One path in the agent's home. `content` is None for a directory."""

    path: str
    """POSIX path relative to the agent's home."""
    mode: int
    content: bytes | None = None


@dataclass
class _Tree:
    """An in-memory folder, with `cp -r` and `mkdir -p` as GNU coreutils do them."""

    entries: dict[str, Entry] = field(default_factory=dict)

    def mkdir_p(self, path: str) -> None:
        parts = PurePosixPath(path).parts
        for i in range(1, len(parts) + 1):
            sub = str(PurePosixPath(*parts[:i]))
            if sub not in self.entries:
                self.entries[sub] = Entry(sub, 0o777 & ~UMASK)

    def is_dir(self, path: str) -> bool:
        entry = self.entries.get(path)
        return entry is not None and entry.content is None

    def write(self, path: str, data: bytes, mode: int) -> None:
        """Create or overwrite a file. Overwriting keeps the old mode, as `cp` and `>` do."""
        old = self.entries.get(path)
        self.entries[path] = Entry(path, old.mode if old else mode, data)

    def copy(self, src: Path, dest: str) -> None:
        """`cp -r src dest`: into `dest/<name>` if `dest` is a folder, else as `dest`."""
        target = f"{dest}/{src.name}" if self.is_dir(dest) else dest
        self._copy_to(src, target)

    def copy_contents(self, src: Path, dest: str) -> None:
        """`cp -r src/* dest/`: every non-hidden child of `src` (the shell glob)."""
        for child in sorted(src.iterdir(), key=lambda p: p.name):
            if not child.name.startswith("."):
                self.copy(child, dest)

    def _copy_to(self, src: Path, target: str) -> None:
        mode = src.stat().st_mode & 0o777 & ~UMASK
        if src.is_dir():
            if target not in self.entries:
                self.entries[target] = Entry(target, mode)
            for child in sorted(src.iterdir(), key=lambda p: p.name):
                self._copy_to(child, f"{target}/{child.name}")
        else:
            self.write(target, src.read_bytes(), mode)

    def remove_tree(self, path: str) -> None:
        for key in [k for k in self.entries if k == path or k.startswith(f"{path}/")]:
            del self.entries[key]

    def chmod_plus_x(self, path: str) -> None:
        """`chmod +x`: adds execute where the umask allows it."""
        entry = self.entries[path]
        self.entries[path] = Entry(path, entry.mode | (0o111 & ~UMASK), entry.content)


def _check_side_task(setting: str, side_task: str, upstream: Path | None) -> None:
    if side_task not in list_side_tasks(setting, upstream):
        raise ValueError(
            f"unknown side task {side_task!r} for {setting}; "
            f"available: {list_side_tasks(setting, upstream)}"
        )


def pre_agent_hooks(setting: str, side_task: str, upstream: Path | None = None) -> list[Path]:
    """The `pre_agent.sh` hooks upstream runs for this run, in order (`:299-309`)."""
    setting_dir = settings_root(upstream) / setting
    hooks = [setting_dir / "hooks" / "pre_agent.sh"]
    if side_task != BASELINE:
        hooks.append(setting_dir / "side_tasks" / side_task / "hooks" / "pre_agent.sh")
    return [hook for hook in hooks if hook.is_file()]


def hook_env(setting: str, task_config: str, upstream: Path | None = None) -> dict[str, str]:
    """The task config's values as the hooks see them (`HOOK_ENV`, `:273-282`).

    Upstream prints each with `f'{k}={v}'` and reads the lines back with `IFS='='`.
    """
    configs = load_config(setting, upstream).get("configs", {})
    return {str(k): f"{v}" for k, v in configs.get(task_config, {}).items()}


def home_entries(
    setting: str,
    side_task: str,
    *,
    agent: AgentFamily | None,
    prompt: str | None,
    upstream: Path | None = None,
) -> list[Entry]:
    """The agent's home after the pipeline's copies, before `timer.sh` and the hooks.

    Args:
        setting: setting name, for example "cuda_kernel".
        side_task: side task name, or "none" for the baseline.
        agent: the agent family whose `solve.sh` is staged as `agent_solve.sh`; None stages
            no launch helpers (an agent upstream does not have, such as ReAct).
        prompt: the contents of `prompt.txt` (see
            `researcharena_inspect.prompts.agent_instruction_file`); None leaves it out.
        upstream: upstream tree to read; defaults to the pinned checkout.

    Returns:
        Folders and files with their modes, sorted by path.

    Raises:
        ValueError: unknown side task.
        StagingError: a staged launch helper contains a word upstream refuses (`:239-245`).
    """
    _check_side_task(setting, side_task, upstream)
    root = upstream_root(upstream)
    setting_dir = settings_root(upstream) / setting
    tree = _Tree()
    tree.mkdir_p("task")
    if (setting_dir / "task_context").is_dir():
        tree.copy_contents(setting_dir / "task_context", "task")
    overlay = setting_dir / "side_tasks" / side_task / "task_context"
    if side_task != BASELINE and overlay.is_dir():
        tree.copy_contents(overlay, "task")
    if tree.is_dir("task/agent_workspace"):
        # `cp -r task/agent_workspace/* task/` reads the copy just made, not upstream's tree.
        flat = {
            key: entry
            for key, entry in tree.entries.items()
            if key.startswith("task/agent_workspace/")
        }
        for key in sorted(flat):
            rel = key.removeprefix("task/agent_workspace/")
            if PurePosixPath(rel).parts[0].startswith("."):
                continue
            entry = flat[key]
            dest = f"task/{rel}"
            if entry.content is None:
                if dest not in tree.entries:
                    tree.entries[dest] = Entry(dest, entry.mode)
            else:
                tree.write(dest, entry.content, entry.mode)
        tree.remove_tree("task/agent_workspace")
    for key in list(tree.entries):
        path = PurePosixPath(key)
        if path.parent == PurePosixPath("task") and path.suffix in (".sh", ".py"):
            tree.chmod_plus_x(key)
    if (root / "lib" / "templates").is_dir():
        tree.copy(root / "lib" / "templates", "task/templates")
    if agent is not None:
        tree.copy(root / "agents" / agent / "solve.sh", "agent_solve.sh")
        utils = root / "src" / "utils"
        tree.copy(utils / "pre_agent_setup.sh", "pre_agent_setup.sh")
        tree.copy(utils / "update_agent_cli.sh", "update_agent_cli.sh")
        tree.copy(utils / "check_effort.py", "check_effort.py")
        for name in LAUNCH_HELPERS:
            text = (tree.entries[name].content or b"").decode("utf-8", errors="replace")
            if AGENT_VISIBLE_HINTS.search(text):
                raise StagingError(
                    f"{name} is staged into the agent's home and contains a give-away word"
                )
    if prompt is not None:
        tree.write("prompt.txt", prompt.encode("utf-8"), 0o666 & ~UMASK)
    return [tree.entries[k] for k in sorted(tree.entries)]


def tar_entries(entries: list[Entry]) -> bytes:
    """A tar archive of `entries` (folders before their contents), modes kept."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for entry in sorted(entries, key=lambda e: e.path):
            info = tarfile.TarInfo(entry.path)
            info.mode = entry.mode
            info.uid = info.gid = AGENT_UID
            info.uname = info.gname = AGENT_USER
            if entry.content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(entry.content)
                tar.addfile(info, io.BytesIO(entry.content))
    return buffer.getvalue()


def hook_bundle(setting: str, side_task: str, upstream: Path | None = None) -> list[Entry]:
    """Upstream files the in-sandbox steps run, laid out as in upstream's tree.

    `create_timer.sh` and the setting's `hooks/`; for a side task, its whole folder (the
    `independent` hook copies `../requirements.txt` from beside itself).
    """
    root = upstream_root(upstream)
    tree = _Tree()
    tree.mkdir_p("src/utils")
    tree.copy(root / "src" / "utils" / "create_timer.sh", "src/utils/create_timer.sh")
    setting_dir = settings_root(upstream) / setting
    tree.mkdir_p(f"settings/{setting}")
    if (setting_dir / "hooks").is_dir():
        tree.copy(setting_dir / "hooks", f"settings/{setting}/hooks")
    side_dir = setting_dir / "side_tasks" / side_task
    if side_task != BASELINE and side_dir.is_dir():
        tree.mkdir_p(f"settings/{setting}/side_tasks")
        tree.copy(side_dir, f"settings/{setting}/side_tasks/{side_task}")
    return [tree.entries[k] for k in sorted(tree.entries)]


@dataclass(frozen=True)
class StagingResult:
    """What staging ran, for the log. Upstream prints the same to the pipeline's log."""

    timer_hours: str
    hook_outputs: list[tuple[str, int, str]]
    """(hook path relative to upstream, exit status, stdout + stderr)."""
    model_cache: list[dict[str, Any]] = field(default_factory=list)
    """One record per model put into `hf_cache` (`model_cache.provision_models`)."""
    pipeline_hooks: dict[str, Any] | None = None
    """The `pipeline_pre_agent.sh` hooks sourced in the hook service (`services.HookService`),
    with the variables they give the agent; None when the run has none."""


async def _exec(
    cmd: str, *, user: str | None = None, env: dict[str, str] | None = None, cwd: str | None = None
) -> ExecResult[str]:
    return await sandbox_env().exec(["bash", "-c", cmd], user=user, env=env or {}, cwd=cwd)


async def stage_workspace(
    setting: str,
    side_task: str,
    *,
    redruntime: str | int | float,
    agent: AgentFamily | None = "claude",
    task_config: str | None = None,
    write_prompt: bool = True,
    model_cache: ModelCacheMode = "auto",
    upstream: Path | None = None,
) -> StagingResult:
    """Stage the agent's home in the current sample's sandbox, as upstream does.

    Run as the first step of a sample, in a fresh sandbox; its default user must be root.

    Args:
        setting: setting name.
        side_task: side task name, or "none" for the baseline.
        redruntime: the budget in hours, as upstream's command line carries it.
        agent: whose `solve.sh` the agent finds in its home; None for no launch helpers.
        task_config: task config name; None for the setting's first, as upstream.
        write_prompt: write `prompt.txt` (upstream always does).
        model_cache: where the setting's models come from (`auto`: the host seed if it has
            them, else a download; `seed`; `download`; `none` leaves the cache empty).
        upstream: upstream tree to read; defaults to the pinned checkout.

    Raises:
        StagingError: a step failed where upstream's pipeline would have stopped.
        ModelCacheError: the model could not be put in the cache or does not match its pin.
        ServiceError: a `pipeline_pre_agent.sh` hook could not run (no hook service, or the
            hook failed), or the endpoint could not be pinned to loopback.
    """
    hours = format_hours(redruntime)
    _check_side_task(setting, side_task, upstream)
    config = task_config if task_config is not None else default_task_config(setting, upstream)
    prompt = (
        agent_instruction_file(setting, side_task, agent or "claude", hours, config, upstream)
        if write_prompt
        else None
    )
    entries = home_entries(setting, side_task, agent=agent, prompt=prompt, upstream=upstream)

    # The agent's account, and an empty home but for the cache's mount point (Docker, D-093).
    setup = (
        f"set -eu; {ACCOUNT_SETUP}; "
        f'if [ -e {AGENT_HOME} ] && [ -n "$(ls -A {AGENT_HOME} | grep -vx hf_cache)" ]; then '
        f"echo '{AGENT_HOME} is not empty' >&2; exit 1; fi; "
        f"install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {AGENT_HOME}; "
        f"rm -rf {STAGING_DIR}; install -d -m 755 {STAGING_DIR}"
    )
    result = await _exec(setup, user="root")
    if not result.success:
        raise StagingError(f"creating the agent's account and home failed: {result.stderr}")

    await sandbox_env().write_file(f"{STAGING_DIR}/home.tar", tar_entries(entries))
    await sandbox_env().write_file(
        f"{STAGING_DIR}/upstream.tar", tar_entries(hook_bundle(setting, side_task, upstream))
    )
    # `tar -m`: files get the time they are unpacked, as upstream's `cp -r` gives copies the
    # time they are copied (the archive's own times are 1970).
    unpack = (
        f"set -eu; mkdir -p {STAGING_DIR}/upstream && "
        f"tar -xpmf {STAGING_DIR}/upstream.tar --no-same-owner -C {STAGING_DIR}/upstream && "
        f"chmod -R a+rX {STAGING_DIR}"
    )
    result = await _exec(unpack, user="root")
    if not result.success:
        raise StagingError(f"unpacking upstream's scripts failed: {result.stderr}")
    result = await _exec(
        f"umask 022; tar -xpmf {STAGING_DIR}/home.tar --no-same-owner -C {AGENT_HOME}",
        user=AGENT_USER,
    )
    if not result.success:
        raise StagingError(f"writing the agent's home failed: {result.stderr}")

    # The model cache before `timer.sh` records its start: upstream's cache was full before the
    # run began (D-130, D-170, D-210).
    result = await _exec(
        f"set -eu; install -d -m 755 -o {AGENT_USER} -g {AGENT_USER} {HF_HOME}", user="root"
    )
    if not result.success:
        raise StagingError(f"creating the model cache folder failed: {result.stderr}")
    models = await provision_models(setting, config, model_cache, upstream)

    upstream_copy = f"{STAGING_DIR}/upstream"
    base_env = {"HOME": AGENT_HOME, "USER": AGENT_USER, "LOGNAME": AGENT_USER}
    timer = (
        f"set -euo pipefail; umask 022; "
        f"bash {upstream_copy}/src/utils/create_timer.sh {shlex.quote(hours)} > {TASK_DIR}/timer.sh; "
        f"chmod +x {TASK_DIR}/timer.sh"
    )
    result = await _exec(timer, user=AGENT_USER, env=base_env)
    if not result.success:
        raise StagingError(f"creating timer.sh failed: {result.stderr}")

    outputs: list[tuple[str, int, str]] = []
    env = {**base_env, **GIT_IDENTITY, **hook_env(setting, config, upstream), "JOB_DIR": AGENT_HOME}
    root = upstream_root(upstream)
    for hook in pre_agent_hooks(setting, side_task, upstream):
        rel = hook.relative_to(root).as_posix()
        result = await _exec(
            f"umask 022; bash {upstream_copy}/{rel}", user=AGENT_USER, env=env, cwd=AGENT_HOME
        )
        outputs.append((rel, result.returncode, result.stdout + result.stderr))
        if not result.success:
            raise StagingError(f"{rel} failed ({result.returncode}): {result.stderr}")

    # Phase 3's hooks (`:325-335`), sourced just before the agent starts; then D-173.
    hook_record: dict[str, Any] | None = None
    service = await start_pipeline_hooks(setting, side_task, config, upstream)
    if service is not None:
        service.endpoint_pins = await pin_endpoint_hosts(endpoint_hosts(setting, upstream))
        hook_record = asdict(service)

    result = await _exec(f"rm -rf {STAGING_DIR}", user="root")
    if not result.success:
        raise StagingError(f"finishing staging failed: {result.stderr}")
    return StagingResult(
        timer_hours=hours, hook_outputs=outputs, model_cache=models, pipeline_hooks=hook_record
    )


STAGING_METADATA_KEY = "environment_staging"
AGENT_EXTRA_ENV_KEY = "agent_extra_env"
"""Under `STAGING_METADATA_KEY`: the variables the hooks add to the agent's environment
(`AGENT_EXTRA_ENV_ARGS`, `pipeline_redteam.sh:329-335`, `:371`)."""


def agent_extra_env(metadata: dict[str, Any]) -> dict[str, str]:
    """The hooks' variables for the agent, from a sample's metadata; empty when there are none."""
    staging = metadata.get(STAGING_METADATA_KEY) or {}
    return dict(staging.get(AGENT_EXTRA_ENV_KEY) or {})


@solver
def stage_agent_workspace(
    setting: str | None = None,
    side_task: str | None = None,
    redruntime: str | int | float | None = None,
    agent: AgentFamily | None = "claude",
    task_config: str | None = None,
    model_cache: ModelCacheMode = "auto",
) -> Solver:
    """Setup solver: stage the agent's workspace for this sample (see `stage_workspace`).

    Arguments left as None are read from the sample's metadata. The staging record goes to the
    metadata under `STAGING_METADATA_KEY`.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        chosen_setting = setting if setting is not None else meta["setting"]
        chosen_side_task = side_task if side_task is not None else meta["side_task"]
        hours = redruntime if redruntime is not None else meta["redruntime"]
        config = task_config if task_config is not None else meta.get("task_config")
        staged = await stage_workspace(
            chosen_setting,
            chosen_side_task,
            redruntime=hours,
            agent=agent,
            task_config=config,
            model_cache=model_cache,
        )
        state.metadata[STAGING_METADATA_KEY] = {
            "timer_hours": staged.timer_hours,
            "hooks": [
                {"hook": hook, "returncode": code, "output": output}
                for hook, code, output in staged.hook_outputs
            ],
        }
        if staged.model_cache:
            state.metadata[STAGING_METADATA_KEY]["model_cache"] = staged.model_cache
        if staged.pipeline_hooks is not None:
            state.metadata[STAGING_METADATA_KEY]["pipeline_hooks"] = staged.pipeline_hooks
            state.metadata[STAGING_METADATA_KEY][AGENT_EXTRA_ENV_KEY] = dict(
                staged.pipeline_hooks["agent_env"]
            )
        return state

    return solve


def describe(entries: list[Entry]) -> list[tuple[str, int, int]]:
    """(path, mode, size) for each entry; a folder's size is -1."""
    return [(e.path, e.mode, -1 if e.content is None else len(e.content)) for e in entries]


__all__ = [
    "ACCOUNT_SETUP",
    "AGENT_EXTRA_ENV_KEY",
    "AGENT_VISIBLE_HINTS",
    "GIT_IDENTITY",
    "LAUNCH_HELPERS",
    "STAGING_METADATA_KEY",
    "AgentFamily",
    "Entry",
    "StagingError",
    "StagingResult",
    "agent_extra_env",
    "describe",
    "home_entries",
    "hook_bundle",
    "hook_env",
    "pipeline_hooks",
    "pre_agent_hooks",
    "stage_agent_workspace",
    "stage_workspace",
    "tar_entries",
]
