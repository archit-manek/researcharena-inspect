"""Runs upstream's own prompt code from the pinned checkout, for the parity tests.

Upstream builds its prompts inside bash scripts that also start containers and agents.
The oracle runs, unchanged:

- the blocks of `src/pipeline_redteam.sh` and `src/pipeline_monitor.sh` that pick the
  task config, build the prompts and stage the files, cut out by their marker lines;
- each family's whole `agents/<family>/solve.sh` and `monitors/<family>/monitor.sh`, with
  stub `claude`, `codex` and `opencode` programs on PATH that record their arguments and
  stdin. So the tests see exactly what each CLI would have received.

Other stubs: `timeout` (just runs the command; macOS has none), `date +%s` (a fixed clock,
so the resume prompt is deterministic), and the CLI installer and effort checker that the
scripts expect in the home folder. None of these touch prompt text.

Bash before 4.4 treats an empty array as unbound under `set -u` (macOS ships 3.2). There the
blocks run with `set -eo pipefail`, which can't change the output of a run that succeeds,
and the agent and monitor efforts are set to "high" so the scripts' effort arrays aren't
empty. With a newer bash (CI) the blocks run under upstream's own `set -euo pipefail` and
the efforts come from upstream's `effort.json`.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from researcharena_inspect.prompts import MonitorAccess

# The fixed clock for `date +%s`.
FAKE_EPOCH = 1_800_000_000

_CLI_STUB = r"""#!/usr/bin/env bash
# Records one call: argv (NUL-separated), stdin and working directory.
log="${STUB_LOG:?}"
count_file="$log/.count"
n=$(( $(cat "$count_file" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$count_file"
d="$log/call$n"
mkdir -p "$d"
printf '%s\0' "$@" > "$d/argv"
cat > "$d/stdin"
pwd > "$d/cwd"
# The first call succeeds; later ones report a timeout, which ends solve.sh's resume loop.
if [ "$n" -ge "${STUB_TIMEOUT_FROM_CALL:-2}" ]; then exit 124; fi
exit 0
"""

_TIMEOUT_STUB = r"""#!/usr/bin/env bash
# timeout [options] DURATION COMMAND...: run COMMAND without a limit.
while [ $# -gt 0 ]; do
    case "$1" in
        -*) shift ;;
        *) break ;;
    esac
done
shift
exec "$@"
"""

_DATE_STUB = r"""#!/usr/bin/env bash
if [ "${1:-}" = "+%s" ]; then echo "${FAKE_EPOCH:?}"; else exec /bin/date "$@"; fi
"""


@cache
def bash_allows_empty_arrays_under_nounset() -> bool:
    """Whether this bash expands an empty array under `set -u` (bash 4.4 and later)."""
    probe = subprocess.run(
        ["bash", "-c", 'set -u; a=(); echo "${a[@]}"'], capture_output=True, check=False
    )
    return probe.returncode == 0


def shell_options() -> str:
    """The `set` line the oracle runs upstream's blocks under (see the module docstring)."""
    if bash_allows_empty_arrays_under_nounset():
        return "set -euo pipefail"
    return "set -eo pipefail"


def script_block(script: Path, start: str, end: str, *, include_end: bool) -> str:
    """The lines of `script` from the one starting with `start` to the next starting with `end`.

    Each marker must start exactly one line from the start onwards, so a changed upstream
    script fails loudly instead of running the wrong lines.
    """
    lines = script.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith(start)]
    if len(starts) != 1:
        raise ValueError(f"{script.name}: {len(starts)} lines start with {start!r}")
    first = starts[0]
    ends = [i for i in range(first + 1, len(lines)) if lines[i].startswith(end)]
    if not ends:
        raise ValueError(f"{script.name}: no line starting with {end!r} after {start!r}")
    last = ends[0] + (1 if include_end else 0)
    return "".join(lines[first:last])


def shell_function(script: Path, name: str) -> str:
    """The definition of the bash function `name` in `script` (up to its closing brace)."""
    return script_block(script, f"{name}() {{", "}", include_end=True)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@dataclass(frozen=True)
class Stubs:
    bin_dir: Path
    home_files: dict[str, str]

    def path(self) -> str:
        return f"{self.bin_dir}{os.pathsep}{os.environ['PATH']}"

    def install_home(self, home: Path) -> None:
        """Put the stub CLI installer and effort checker where the scripts look for them."""
        for name, text in self.home_files.items():
            _write_executable(home / name, text)


def make_stubs(folder: Path) -> Stubs:
    bin_dir = folder / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for cli in ("claude", "codex", "opencode"):
        _write_executable(bin_dir / cli, _CLI_STUB)
    _write_executable(bin_dir / "timeout", _TIMEOUT_STUB)
    _write_executable(bin_dir / "date", _DATE_STUB)
    home_files = {
        "update_agent_cli.sh": "#!/usr/bin/env bash\nexit 0\n",
        "check_effort.py": "import sys\nsys.exit(0)\n",
    }
    return Stubs(bin_dir=bin_dir, home_files=home_files)


def base_env(home: Path, stubs: Stubs) -> dict[str, str]:
    """A clean environment: no credentials, a throwaway home, the stubs first on PATH."""
    env = {"HOME": str(home), "PATH": stubs.path(), "FAKE_EPOCH": str(FAKE_EPOCH)}
    for name in ("LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


@dataclass(frozen=True)
class CliCall:
    argv: list[str]
    stdin: bytes
    cwd: str


def read_calls(log: Path) -> list[CliCall]:
    calls = []
    n = 1
    while (log / f"call{n}").is_dir():
        d = log / f"call{n}"
        argv = (d / "argv").read_bytes().split(b"\0")[:-1]
        calls.append(
            CliCall(
                argv=[a.decode("utf-8") for a in argv],
                stdin=(d / "stdin").read_bytes(),
                cwd=(d / "cwd").read_text(encoding="utf-8").strip(),
            )
        )
        n += 1
    return calls


def _run_bash(script: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"upstream block failed ({result.returncode})\n--- stdout\n{result.stdout}"
            f"\n--- stderr\n{result.stderr}"
        )
    return result


# ----------------------------------------------------------------------------- agent


@dataclass(frozen=True)
class AgentRun:
    """What upstream produced for one red-team run."""

    task_config: str
    resume_agent: str
    instruction_md: bytes
    prompt_txt: bytes
    calls: list[CliCall]


_AGENT_DRIVER = r"""
{options}
SETTING_DIR="$ARENA_ROOT/settings/$SETTING"
# --- pipeline_redteam.sh: task config, effort, resume_agent
{config_block}
# --- pipeline_redteam.sh: PHASE 1
{prompt_block}
printf '%s' "$TASK_CONFIG" > "$OUT/task_config"
printf '%s' "$RESUME_AGENT" > "$OUT/resume_agent"
EFFORT="$RED_EFFORT_LEVEL"
[[ -n "${{FORCE_EFFORT:-}}" ]] && EFFORT="$FORCE_EFFORT"
mkdir -p "$AGENT_JOB_DIR/task"
cd "$AGENT_JOB_DIR/task"
set +e
HOME="$AGENT_HOME" PROMPT_FILE="$AGENT_JOB_DIR/prompt.txt" AGENT_CONFIG="$RED_MODEL" \
    AGENT_EFFORT="$EFFORT" TIME_BUDGET_HOURS="$REDRUNTIME" AGENT_RESUME="$RESUME_AGENT" \
    CLAUDE_OAUTH_TOKEN=stub OPENCODE_MIN_ROUND_SECONDS=0 \
    bash "$ARENA_ROOT/agents/$RED_PROVIDER/solve.sh" > "$OUT/solve_out" 2> "$OUT/solve_err"
echo "$?" > "$OUT/solve_rc"
"""

# A model id per family as upstream's runs name them (opencode's has a provider prefix).
RED_MODELS = {"claude": "claude-opus-4-6", "codex": "gpt-5.4", "opencode": "opencode/glm-5.1"}


def run_upstream_agent(
    upstream: Path,
    work: Path,
    stubs: Stubs,
    *,
    setting: str,
    side_task: str,
    task_config: str,
    redruntime: str,
    family: str,
) -> AgentRun:
    """Build and deliver the red-team prompt with upstream's code; record what the CLI got.

    `task_config` "" means "not given", as `run.sh` passes it by default.
    """
    work.mkdir(parents=True)
    out, home, log = work / "out", work / "home", work / "cli_log"
    for d in (out, home, log):
        d.mkdir()
    stubs.install_home(home)
    script_path = upstream / "src" / "pipeline_redteam.sh"
    driver = _AGENT_DRIVER.format(
        options=shell_options(),
        config_block=script_block(
            script_path,
            "# Derive task config from config.json if not specified",
            "# ======================= PHASE 0: SETUP",
            include_end=False,
        ),
        prompt_block=script_block(
            script_path,
            "# ======================= PHASE 1: GENERATE PROMPT",
            "# ======================= PHASE 2: PRE-AGENT HOOKS",
            include_end=False,
        ),
    )
    env = base_env(work / "driver_home", stubs)
    (work / "driver_home").mkdir()
    env |= {
        "ARENA_ROOT": str(upstream),
        "SETTING": setting,
        "SIDE_TASK": side_task,
        "TASK_CONFIG": task_config,
        "REDRUNTIME": redruntime,
        "RED_PROVIDER": family,
        "RED_MODEL": RED_MODELS[family],
        "EVAL_DIR": str(work / "eval_dir"),
        "AGENT_JOB_DIR": str(work / "agent_home"),
        "AGENT_HOME": str(home),
        "OUT": str(out),
        "STUB_LOG": str(log),
    }
    if not bash_allows_empty_arrays_under_nounset():
        env["FORCE_EFFORT"] = "high"
    (work / "eval_dir").mkdir()
    (work / "agent_home").mkdir()
    _run_bash(driver, env, work)
    return AgentRun(
        task_config=(out / "task_config").read_text(encoding="utf-8"),
        resume_agent=(out / "resume_agent").read_text(encoding="utf-8"),
        instruction_md=(work / "eval_dir" / "instruction.md").read_bytes(),
        prompt_txt=(work / "agent_home" / "prompt.txt").read_bytes(),
        calls=read_calls(log),
    )


def run_upstream_resume_prompt(upstream: Path, family: str, remaining_seconds: int) -> bytes:
    """`make_resume_prompt` from `agents/<family>/solve.sh`, with `remaining_seconds` left."""
    solve = upstream / "agents" / family / "solve.sh"
    script = "\n".join(
        [
            "set -uo pipefail",
            "date() { echo 1000000; }",
            shell_function(solve, "time_left_seconds"),
            shell_function(solve, "make_resume_prompt"),
            f"DEADLINE_TS=$((1000000 + {remaining_seconds}))",
            "make_resume_prompt",
        ]
    )
    env = {"PATH": os.environ["PATH"]}
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, check=True)
    return result.stdout


# --------------------------------------------------------------------------- monitor


@dataclass(frozen=True)
class MonitorRun:
    """What upstream produced for one monitor run."""

    prompt_txt: bytes
    monitor_prompt_md: bytes
    main_task_instruction_md: bytes
    staged_instruction: bytes
    staged_transcript: bytes
    run_dir_name: str
    calls: list[CliCall]


_MONITOR_DRIVER = r"""
{options}
trap 'rm -rf "${{BLUE_JOB_DIR:-/nonexistent}}"' EXIT
# --- pipeline_monitor.sh: from reading metadata.json to writing prompt.txt
{monitor_block}
cp "$BLUE_RUN_DIR/monitor_prompt.md" "$OUT/monitor_prompt.md"
cp "$BLUE_JOB_DIR/prompt.txt" "$OUT/prompt.txt"
cp "$BLUE_JOB_DIR/main_task_instruction.md" "$OUT/main_task_instruction.md"
cp "$BLUE_JOB_DIR/task/task_instruction.md" "$OUT/staged_instruction"
cp "$BLUE_JOB_DIR/task/agent_transcript.txt" "$OUT/staged_transcript"
basename "$BLUE_RUN_DIR" > "$OUT/run_dir_name"
# The monitor container: HOME is the job dir, the working directory its task/ folder.
for f in "$STUB_HOME"/*; do cp "$f" "$BLUE_JOB_DIR/"; done
EFFORT="$BLUE_EFFORT_LEVEL"
[[ -n "${{FORCE_EFFORT:-}}" ]] && EFFORT="$FORCE_EFFORT"
cd "$BLUE_JOB_DIR/task"
HOME="$BLUE_JOB_DIR" AGENT_CONFIG="$BLUE_MODEL" BLUE_EFFORT_LEVEL="$EFFORT" \
    bash "$BLUE_JOB_DIR/monitor_solve.sh" > "$OUT/monitor_out" 2> "$OUT/monitor_err"
"""

BLUE_MODELS = {"claude": "claude-sonnet-4-6", "codex": "gpt-5.4", "opencode": "opencode/glm-5.1"}


def make_eval_dir(
    eval_dir: Path,
    *,
    setting: str,
    side_task: str,
    task_config: str | None,
    transcripts: dict[str, str],
    snapshot: Path,
) -> None:
    """A red-team run folder as `pipeline_redteam.sh` leaves it, for the monitor to read."""
    (eval_dir / "red_team").mkdir(parents=True)
    metadata: dict[str, object] = {"setting": setting, "side_task": side_task}
    if task_config is not None:
        metadata["task_config"] = task_config
    metadata["red_provider"] = "claude"
    (eval_dir / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    for name, text in transcripts.items():
        (eval_dir / "red_team" / name).write_bytes(text.encode("utf-8"))
    shutil.copytree(snapshot, eval_dir / "red_team" / "task_snapshot", symlinks=True)


def run_upstream_monitor(
    upstream: Path,
    work: Path,
    stubs: Stubs,
    *,
    eval_dir: Path,
    access: MonitorAccess,
    family: str,
) -> MonitorRun:
    """Build and deliver the monitor prompt with upstream's code; record what the CLI got."""
    work.mkdir(parents=True)
    out, stub_home, log = work / "out", work / "stub_home", work / "cli_log"
    for d in (out, stub_home, log):
        d.mkdir()
    stubs.install_home(stub_home)
    driver = _MONITOR_DRIVER.format(
        options=shell_options(),
        monitor_block=script_block(
            upstream / "src" / "pipeline_monitor.sh",
            "# Read metadata",
            'echo "$MONITOR_PROMPT" > "$BLUE_JOB_DIR/prompt.txt"',
            include_end=True,
        ),
    )
    env = base_env(work / "driver_home", stubs)
    (work / "driver_home").mkdir()
    env |= {
        "ARENA_ROOT": str(upstream),
        "EVAL_DIR": str(eval_dir),
        "BLUERUNTIME": "0.5",
        "WITH_ARTIFACT": "true" if access.with_artifact else "false",
        "WITH_COT": "true" if access.with_cot else "false",
        "BLUE_PROVIDER": family,
        "BLUE_MODEL": BLUE_MODELS[family],
        "BLUE_CLI_VERSION_VAR": "",
        "BLUE_CLI_VERSION": "",
        "OUT": str(out),
        "STUB_HOME": str(stub_home),
        "STUB_LOG": str(log),
        "STUB_TIMEOUT_FROM_CALL": "99",
    }
    if not bash_allows_empty_arrays_under_nounset():
        env["FORCE_EFFORT"] = "high"
    _run_bash(driver, env, work)
    return MonitorRun(
        prompt_txt=(out / "prompt.txt").read_bytes(),
        monitor_prompt_md=(out / "monitor_prompt.md").read_bytes(),
        main_task_instruction_md=(out / "main_task_instruction.md").read_bytes(),
        staged_instruction=(out / "staged_instruction").read_bytes(),
        staged_transcript=(out / "staged_transcript").read_bytes(),
        run_dir_name=(out / "run_dir_name").read_text(encoding="utf-8").strip(),
        calls=read_calls(log),
    )
