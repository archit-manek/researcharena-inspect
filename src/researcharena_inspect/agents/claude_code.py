"""The red-team agent as upstream runs it: Claude Code (`agents/claude/solve.sh:97-100`).

Inspect SWE resumes with `--resume <session id>` where upstream uses `--continue`; both continue
the run's one session. The model calls go through Inspect's bridge. Two patches to Inspect SWE
live here: `capture_cli_output` keeps the CLI's stdout (upstream's `solve_out.json`), which
Inspect SWE discards, and `prompt_on_stdin` sends a new session's prompt on stdin, as upstream
does. Deviations: D-076 (login), D-077 (effort flags), D-078 (`--continue`), D-079 (CLI build),
D-080 (Inspect SWE's settings).
"""

import importlib
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from inspect_ai.agent import Agent
from inspect_ai.util import ExecRemoteStreamingOptions, sandbox

from researcharena_inspect.environment.layout import TASK_DIR

CLI_INSTALL_PATH = "/usr/local/bin/claude"
"""Where the installed CLI is linked, so `claude` is on the agent's PATH (upstream puts
`$HOME/.agent_cli/bin` first, `solve.sh:37`) and Inspect SWE finds it without reinstalling."""

# solve.sh:26 exports BASH_MAX_TIMEOUT_MS=36000000, over pipeline_redteam.sh:368's 3600000;
# pipeline_redteam.sh:369 passes BASH_DEFAULT_TIMEOUT_MS.
BASH_TIMEOUT_ENV = {"BASH_MAX_TIMEOUT_MS": "36000000", "BASH_DEFAULT_TIMEOUT_MS": "3600000"}


AUTO_MEMORY_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0"}
"""Keep Claude Code's auto-memory on, as it was upstream. Inspect SWE turns it off by default
(`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`); an explicit "0" turns it on. In 3 of the 45 published
Claude CUDA runs the agent read or wrote `~/.claude/projects/-home-user-task/memory/`."""


def claude_code_env(model: str, effort: str, redruntime: str, resume: bool) -> dict[str, str]:
    """The job settings upstream passes to the agent process (`pipeline_redteam.sh:362-365`).

    The agent sees them too. Container-wide variables belong to the sandbox.
    """
    # D-076: no OAuth token and no ANTHROPIC_API_KEY="" (solve.sh:11-24).
    return {
        **BASH_TIMEOUT_ENV,
        **AUTO_MEMORY_ENV,
        "AGENT_CONFIG": model,
        "AGENT_EFFORT": effort,
        "TIME_BUDGET_HOURS": redruntime,
        "AGENT_RESUME": "true" if resume else "false",
    }


def red_team_claude_code(
    *,
    effort: str,
    env: dict[str, str],
    small_model: str | None = None,
    user: str | None = None,
) -> Agent:
    """Claude Code configured as upstream's red-team agent.

    Make one per sample: the session id is drawn once per agent, and each later call
    resumes that session.

    Args:
        effort: the resolved effort (`default` sets none), set on the served model.
        env: variables for the CLI process (`claude_code_env`).
        small_model: model for Claude Code's background calls; `None` sends them to the
            task's model.
        user: sandbox user to run as; `None` for the sandbox's default.
    """
    from inspect_swe import claude_code

    # D-077: the effort is set on the served model, not with `--effort` and
    # `--thinking-display` (solve.sh:99). D-078: `--resume` for `--continue`; the first prompt
    # goes on stdin only inside `prompt_on_stdin`, which the solver opens (D-257). D-105: see
    # `solver.AgentRounds`. D-080: Inspect SWE's own environment is kept, including
    # CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: clearing it adds no tool through the bridge, and
    # forcing tool search on hides most tools from the model (see monitor/agents.py).
    return claude_code(
        cwd=TASK_DIR,
        effort=None if effort == "default" else effort,  # type: ignore[arg-type]
        # Installed before the budget starts (install_claude_code).
        version="sandbox",
        env=env,
        haiku_model=small_model,
        user=user,
        # Upstream does not retry a refused or crashed CLI run inside a round; the
        # continuation loop decides what happens next (solve.sh:112-133).
        retry_refusals=None,
        retry_uncaught_errors=None,
    )


@dataclass(frozen=True)
class InstalledCli:
    path: str
    version: str
    """`claude --version` output, as upstream records it in `cli_version.txt`."""
    requested: str


async def install_claude_code(version: str, user: str | None = None) -> InstalledCli:
    """Install the CLI before the agent's clock starts, as `solve.sh:35-38` does.

    Args:
        version: `latest`, `stable`, a version such as `2.1.162`, or `sandbox`/`auto` to use a
            `claude` already in the image. Upstream installs `latest` unless
            `CLAUDE_CLI_VERSION` is set (`src/utils/update_agent_cli.sh:38-43`).
        user: sandbox user to install for.
    """
    from inspect_swe._claude_code.agentbinary import claude_code_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed

    # D-079: Anthropic's native build, not the npm package. D-083: a failed install fails the
    # sample; upstream scores the untouched workspace.
    sbox = sandbox()
    path = await ensure_agent_binary_installed(claude_code_binary_source(), version, user, sbox)
    if path != CLI_INSTALL_PATH:
        link = await sbox.exec(["ln", "-sf", path, CLI_INSTALL_PATH], user="root")
        if not link.success:
            raise RuntimeError(f"could not link {path} to {CLI_INSTALL_PATH}: {link.stderr}")
    result = await sbox.exec([CLI_INSTALL_PATH, "--version"], user=user)
    if not result.success:
        raise RuntimeError(f"`claude --version` failed: {result.stderr}")
    return InstalledCli(path=path, version=result.stdout.strip(), requested=version)


# --- Capturing the CLI's output -------------------------------------------------------------

_stdout_sink: ContextVar[list[str] | None] = ContextVar("researcharena_cli_stdout", default=None)
_stderr_sink: ContextVar[list[str] | None] = ContextVar("researcharena_cli_stderr", default=None)
_exit_sink: ContextVar[list[int] | None] = ContextVar("researcharena_cli_exits", default=None)

_TEE_ATTRIBUTE = "_researcharena_tee"
_SWE_MODULE = "inspect_swe._claude_code.claude_code"


def _install_tee() -> None:
    """Wrap Inspect SWE's `claude_code_event_stream` (0.2.x), once.

    Every event passes on unchanged; inside a capture, each stdout line, stderr chunk and exit
    code is also kept. Outside one (the monitor, other tasks) nothing changes.
    """
    module = importlib.import_module(_SWE_MODULE)
    original = module.claude_code_event_stream
    if getattr(original, _TEE_ATTRIBUTE, False):
        return
    stream = importlib.import_module("inspect_swe._claude_code._events.stream")
    line_events = (stream.JsonlEvent, stream.JsonlParseError)
    stderr_event = stream.StderrEvent
    exit_event = stream.ExitEvent

    async def tee(proc: Any) -> AsyncIterator[Any]:
        out, err, exits = _stdout_sink.get(), _stderr_sink.get(), _exit_sink.get()
        async for event in original(proc):
            if out is not None and isinstance(event, line_events):
                out.append(event.line)
            elif err is not None and isinstance(event, stderr_event):
                err.append(event.data)
            elif exits is not None and isinstance(event, exit_event):
                exits.append(event.code)
            yield event

    setattr(tee, _TEE_ATTRIBUTE, True)
    setattr(module, "claude_code_event_stream", tee)  # noqa: B010


@contextmanager
def capture_cli_output(
    stdout: list[str], stderr: list[str], exit_codes: list[int] | None = None
) -> Iterator[None]:
    """Append the CLI's stdout lines, stderr chunks and exit codes to the given lists while open.

    Stdout lines are as Inspect SWE splits them (stripped, blank lines dropped); joined with
    newlines they are upstream's `solve_out.json` without its non-JSON preamble. `exit_codes`
    gets one entry per CLI process that exited, including the refusal's 1, which Inspect SWE
    does not raise for (D-084).
    """
    _install_tee()
    out_token = _stdout_sink.set(stdout)
    err_token = _stderr_sink.set(stderr)
    exit_token = _exit_sink.set(exit_codes)
    try:
        yield
    finally:
        _stdout_sink.reset(out_token)
        _stderr_sink.reset(err_token)
        _exit_sink.reset(exit_token)


# --- The first prompt on stdin (D-078, D-257) -----------------------------------------------

_SWE_SHELL = ["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
"""How Inspect SWE (0.2.x) starts the CLI: `bash -c 'exec 0</dev/null; "$@"' bash <cli> ...`."""

_STDIN_SHELL = ["bash", "-c", 'set +o pipefail; printf \'%s\' "$(cat)" | "$@"', "bash"]
"""A shell that pipes its stdin, the prompt, to the CLI, as upstream's `printf '%s' "$PROMPT" |
claude ...` (`solve.sh:104`) does; the line differs from upstream's in reading the prompt from
stdin rather than from a variable. `$(cat)` reads all of stdin whether or not the CLI does, so
the sandbox's write of the prompt completes and a CLI that exits at once still has its stderr
and exit code reported; like upstream's `$(cat prompt.txt)` it drops trailing newlines (the
prompt has none). No variable holds the prompt, so no inherited `export` or `allexport` can put
it in an environment, and `printf` is a builtin, so it is in no process's arguments either.
`set +o pipefail` keeps the pipeline's status the CLI's even if the shell inherits `pipefail`
(`SHELLOPTS`): otherwise a CLI that exits 0 before reading all of a large prompt would be
reported as the writer's SIGPIPE, 141."""

_prompt_on_stdin: ContextVar[bool] = ContextVar("researcharena_prompt_on_stdin", default=False)
_STDIN_ATTRIBUTE = "_researcharena_stdin"


def stdin_launch(cmd: list[str]) -> tuple[list[str], str] | None:
    """Inspect SWE's CLI command for a new session, with the prompt moved off the command line.

    Returns `_STDIN_SHELL <cli> --session-id <id> <flags>` and the prompt, for the caller to
    send on stdin. A resumed session (`--resume`) gets None and keeps its prompt as an argument,
    as upstream's `--continue "<prompt>"` does (`solve.sh:121`).

    Raises:
        RuntimeError: a command of any other shape, so that a change in Inspect SWE cannot put
            the prompt back on the command line unnoticed.
    """
    # The separator is found by position: a flag's value can itself be `--` (a system message
    # passed as `--append-system-prompt --`, say).
    shape_ok = len(cmd) >= 8 and cmd[:4] == _SWE_SHELL and cmd[-2] == "--"
    if shape_ok and cmd[5] == "--resume":
        return None
    if not shape_ok or cmd[5] != "--session-id":
        raise RuntimeError(f"unexpected Claude Code launch from Inspect SWE: {cmd[:6]!r}")
    return [*_STDIN_SHELL, *cmd[4:-2]], cmd[-1]


class _StdinSandbox:
    """The sandbox inside `prompt_on_stdin`: a new session's launch goes through `stdin_launch`.

    The CLI's launch is the only `exec_remote` call `claude_code()` makes.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def exec_remote(self, cmd: list[str], options: Any = None, *, stream: bool = True) -> Any:
        launch = stdin_launch(cmd)
        if launch is not None:
            cmd, prompt = launch
            options = replace(options or ExecRemoteStreamingOptions(), input=prompt)
        return await self._inner.exec_remote(cmd, options, stream=stream)


@contextmanager
def prompt_on_stdin() -> Iterator[None]:
    """While open, Claude Code runs started by Inspect SWE get a new session's prompt on stdin.

    No process in the sandbox then has the prompt on its command line (D-257). The launching
    shell stays as the CLI's parent instead of `exec`-ing it. Patches `sandbox_env` in Inspect
    SWE's module, which it looks up at each launch; outside this context it is unchanged.
    """
    module = importlib.import_module(_SWE_MODULE)
    original = module.sandbox_env
    if not getattr(original, _STDIN_ATTRIBUTE, False):

        def sandbox_env(name: str | None = None) -> Any:
            inner = original(name)
            return _StdinSandbox(inner) if _prompt_on_stdin.get() else inner

        setattr(sandbox_env, _STDIN_ATTRIBUTE, True)
        setattr(module, "sandbox_env", sandbox_env)  # noqa: B010
    token = _prompt_on_stdin.set(True)
    try:
        yield
    finally:
        _prompt_on_stdin.reset(token)
