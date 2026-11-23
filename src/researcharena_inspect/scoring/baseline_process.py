"""`cuda_baseline_process`: CUDA's baseline measured in a process of its own (D-160, opt-in).

Upstream's `evaluate.py` times plain PyTorch and then the agent's kernels in one process
(`bench.py:189-263`), unless `workspace/baseline_tps.txt` exists. Some kernels fault only after
that baseline pass (D-037). With the option, after `prepare_evaluation` and before the first
attempt, the scorer:

1. records and removes the snapshot's `workspace/baseline_tps.txt` (not the agent's `task/`);
2. records which read-back files (D-001) are present and leaves them: if one gives a result,
   `evaluate.py` returns it without benchmarking, as upstream does;
3. runs `cuda_baseline.py` as an attempt runs, logging to `/mnt/eval/eval/cuda_baseline.log`;
4. uses its baseline if it is finite and above 0; otherwise removes any cached baseline, so
   `evaluate.py` measures in-process as upstream does, and records `failed`;
5. before each attempt, writes the measured text back if the cached file changed or is gone
   (upstream's later attempts would read what attempt 1 cached).
"""

import hashlib
import json
import math
import shlex
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from researcharena_inspect.environment.evaluation import (
    EVAL_TIMEOUT_SECONDS,
    SNAPSHOT_DIR,
    EvaluationCommand,
    evaluator_sandbox,
)
from researcharena_inspect.environment.layout import EVAL_DIR
from researcharena_inspect.prompts.settings import settings_root
from researcharena_inspect.scoring.cuda_baseline import CACHE_NAME, RESULT_PREFIX

OPTION_SETTINGS: tuple[str, ...] = ("cuda_kernel",)
"""Settings whose scorer has the in-process baseline pass the option takes out."""

METADATA_KEY = "cuda_baseline_process"

WORK_DIR = "/var/tmp/researcharena-cuda-baseline"
SCRIPT = f"{WORK_DIR}/cuda_baseline.py"
LOG_PATH = f"{EVAL_DIR}/eval/cuda_baseline.log"
DEFAULT_CACHE = f"{SNAPSHOT_DIR}/workspace/{CACHE_NAME}"

# Not only on PASS: the first of these that is a JSON object is the main score with no benchmark,
# its speedup on PASS and 0.0 otherwise (`evaluate.py:32-53`).
READ_BACK_FILES = (
    "workspace/final_results.json",
    "final_results.json",
    "workspace/last_bench.json",
)
"""What `evaluate.py:33-37` returns without measuring when it gives `compile_status` PASS."""

CHECKED_FILES = ("bench.py", "config.py")
"""The workspace files the baseline process imports; whether they are upstream's is recorded."""

EXEC_GRACE_SECONDS = 120
TEXT_LIMIT = 4096
LOG_LIMIT_BYTES = 200_000

Status = Literal["measured", "failed", "not_applicable"]


@dataclass
class BaselineProcess:
    """What the option did for one sample; kept in the score's metadata."""

    enabled: bool = True
    status: Status = "failed"
    reason: str | None = None
    baseline_tps: float | None = None
    """The measured baseline, as written to the cache file (`.4f`)."""
    cache_path: str | None = None
    cache_text: str | None = None
    agent_baseline_file: str | None = None
    """Text of the snapshot's `workspace/baseline_tps.txt` before step 1; None if absent."""
    removed_by_process: str | None = None
    """Text the baseline process removed at the config's cache path (another path only if the
    workspace's `config.py` moves `workspace_dir`)."""
    read_back_files: list[str] = field(default_factory=list)
    workspace_code_is_upstreams: dict[str, bool] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    user: str | None = None
    cwd: str | None = None
    returncode: int | None = None
    exec_timed_out: bool = False
    seconds: float = 0.0
    result: dict[str, Any] | None = None
    log: str = ""
    restored_before_attempts: list[int] = field(default_factory=list)
    evaluation_baseline_lines: list[list[str]] = field(default_factory=list)
    """Per attempt, the `BASELINE_TPS=` lines of its log."""

    @property
    def applied(self) -> bool:
        return self.status == "measured"

    def summary(self) -> dict[str, Any]:
        return asdict(self)

    def explanation(self) -> str:
        if self.status == "measured":
            return f"Baseline measured in its own process (D-160): {self.baseline_tps} tokens/s. "
        if self.status == "failed":
            return (
                f"Baseline process failed ({self.reason}); evaluate.py measured its own "
                "baseline in-process, as upstream (D-160 not applied). "
            )
        return ""


def not_applicable(setting: str) -> BaselineProcess:
    return BaselineProcess(
        status="not_applicable",
        reason=f"the option applies to {list(OPTION_SETTINGS)}, not {setting}",
    )


def _keep(raw: str) -> str:
    if len(raw) <= LOG_LIMIT_BYTES:
        return raw
    half = LOG_LIMIT_BYTES // 2
    return raw[:half] + f"\n[... {len(raw) - 2 * half} characters not kept ...]\n" + raw[-half:]


def result_line(log: str) -> dict[str, Any] | None:
    """The last `CUDA_BASELINE_RESULT=` line of the log, parsed."""
    for line in reversed(log.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def baseline_lines(log: str) -> list[str]:
    """`bench.py`'s `BASELINE_TPS=` lines: "(cached)" when it read the file (`:194`)."""
    return [line.strip() for line in log.splitlines() if line.strip().startswith("BASELINE_TPS=")]


def valid_tps(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def upstream_sha256(name: str, upstream: Path | None = None) -> str:
    path = settings_root(upstream) / "cuda_kernel" / "task_context" / name
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _as_user(command: EvaluationCommand, script: str, *args: str) -> Any:
    return await evaluator_sandbox().exec(
        ["bash", "-c", script, "baseline", *args],
        user=command.user,
        env=command.env,
        cwd=command.cwd,
    )


async def read_text(command: EvaluationCommand, path: str) -> str | None:
    """A file's first 4 KiB; None if absent or unreadable.

    Read as the evaluator's user because the path is the workspace's choice (its `config.py`).
    """
    result = await _as_user(command, f'[ -f "$1" ] || exit 3; head -c {TEXT_LIMIT} -- "$1"', path)
    return result.stdout if result.success else None


async def remove(command: EvaluationCommand, *paths: str) -> None:
    result = await _as_user(command, 'rm -rf -- "$@"', *paths)
    if not result.success:
        raise RuntimeError(f"removing {paths} failed: {result.stderr}")


async def write_text(command: EvaluationCommand, path: str, text: str) -> None:
    result = await _as_user(
        command,
        'mkdir -p -- "$(dirname -- "$2")" && rm -rf -- "$2" && printf "%s" "$1" > "$2"',
        text,
        path,
    )
    if not result.success:
        raise RuntimeError(f"writing {path} failed: {result.stderr}")


async def _present(command: EvaluationCommand, names: tuple[str, ...]) -> list[str]:
    result = await _as_user(
        command,
        'd=$1; shift; for f in "$@"; do [ -e "$d/$f" ] && echo "$f"; done; true',
        SNAPSHOT_DIR,
        *names,
    )
    return [line for line in result.stdout.splitlines() if line in names]


async def _code_is_upstreams(command: EvaluationCommand) -> dict[str, bool]:
    result = await _as_user(
        command,
        'cd -- "$1" && shift && for f in "$@"; do [ -f "$f" ] && sha256sum -- "$f"; done; true',
        SNAPSHOT_DIR,
        *CHECKED_FILES,
    )
    found = {
        parts[1].lstrip("*"): parts[0]
        for line in result.stdout.splitlines()
        if len(parts := line.split()) == 2
    }
    return {name: found.get(name) == upstream_sha256(name) for name in CHECKED_FILES}


def process_command(timeout: int = EVAL_TIMEOUT_SECONDS) -> list[str]:
    """The baseline process as the scorer runs it: like an attempt, under `timeout`."""
    argv = ["python3", SCRIPT, "--task-dir", SNAPSHOT_DIR]
    return [
        "bash",
        "-c",
        (
            f"timeout --signal=TERM --kill-after=30s {timeout}s "
            f"{' '.join(shlex.quote(a) for a in argv)} > {shlex.quote(LOG_PATH)} 2>&1"
        ),
    ]


async def measure_baseline(command: EvaluationCommand) -> BaselineProcess:
    """Steps 1 to 4 of the module docstring, on a prepared sandbox."""
    box = evaluator_sandbox()
    record = BaselineProcess(user=command.user, cwd=command.cwd)
    # Step 1.
    record.agent_baseline_file = await read_text(command, DEFAULT_CACHE)
    await remove(command, DEFAULT_CACHE)
    # Step 2.
    record.read_back_files = await _present(command, READ_BACK_FILES)
    record.workspace_code_is_upstreams = await _code_is_upstreams(command)

    # Step 3.
    script = (Path(__file__).parent / "cuda_baseline.py").read_bytes()
    setup = await box.exec(
        ["bash", "-c", f"set -e; rm -rf {WORK_DIR}; install -d -m 755 {WORK_DIR}"], user="root"
    )
    if not setup.success:
        raise RuntimeError(f"creating {WORK_DIR} failed: {setup.stderr}")
    await box.write_file(SCRIPT, script)
    chmod = await box.exec(["chmod", "644", SCRIPT], user="root")
    if not chmod.success:
        raise RuntimeError(f"chmod {SCRIPT} failed: {chmod.stderr}")
    record.command = process_command(command.timeout)
    start = time.monotonic()
    try:
        result = await box.exec(
            record.command,
            cwd=command.cwd,
            env=command.env,
            user=command.user,
            timeout=command.timeout + 30 + EXEC_GRACE_SECONDS,
        )
        record.returncode = result.returncode
    except TimeoutError:
        record.exec_timed_out = True
    record.seconds = round(time.monotonic() - start, 1)
    try:
        raw = await box.read_file(LOG_PATH, text=False)
    except FileNotFoundError:
        raw = b""
    log = raw.decode("utf-8", errors="replace")
    record.log = _keep(log)
    await box.exec(["rm", "-rf", WORK_DIR], user="root")

    # Step 4.
    parsed = result_line(log)
    record.result = parsed
    reason = _check(record, parsed)
    if parsed is not None:
        record.cache_path = parsed.get("cache_path") or None
        record.removed_by_process = parsed.get("removed_before")
    if reason is None:
        assert parsed is not None
        assert record.cache_path is not None
        on_disk = await read_text(command, record.cache_path)
        if on_disk is None or on_disk != parsed.get("cache_text"):
            reason = "the cached baseline on disk is not what the process reported"
        else:
            record.cache_text = on_disk
            try:
                record.baseline_tps = float(on_disk.strip())
            except ValueError:
                reason = f"the cached baseline is not a number: {on_disk!r}"
    if reason is None and not valid_tps(record.baseline_tps):
        reason = f"the measured baseline is not a positive number: {record.baseline_tps}"
    if reason is not None:
        record.status, record.reason = "failed", reason
        record.baseline_tps = None
        record.cache_text = None
        # Upstream's path: no cached baseline, so evaluate.py measures its own.
        await remove(command, *dict.fromkeys([DEFAULT_CACHE, record.cache_path or DEFAULT_CACHE]))
        return record
    record.status = "measured"
    return record


def _check(record: BaselineProcess, parsed: dict[str, Any] | None) -> str | None:
    if record.exec_timed_out:
        return "the baseline process did not return"
    if record.returncode != 0:
        detail = (parsed or {}).get("error")
        return f"exit status {record.returncode}" + (f": {detail}" if detail else "")
    if parsed is None:
        return "no result line in the baseline process's output"
    if parsed.get("status") != "measured" or not parsed.get("cache_path"):
        return f"the baseline process reported {parsed.get('status')!r}"
    if not valid_tps(parsed.get("baseline_tps")):
        return f"the measured baseline is not a positive number: {parsed.get('baseline_tps')}"
    return None


async def keep_baseline(record: BaselineProcess, command: EvaluationCommand, attempt: int) -> None:
    """Step 5: before attempt `attempt`, the cached file holds the measured text."""
    if not record.applied or record.cache_path is None or record.cache_text is None:
        return
    if await read_text(command, record.cache_path) == record.cache_text:
        return
    await write_text(command, record.cache_path, record.cache_text)
    record.restored_before_attempts.append(attempt)


__all__ = [
    "DEFAULT_CACHE",
    "METADATA_KEY",
    "OPTION_SETTINGS",
    "READ_BACK_FILES",
    "BaselineProcess",
    "baseline_lines",
    "keep_baseline",
    "measure_baseline",
    "not_applicable",
    "process_command",
    "result_line",
]
