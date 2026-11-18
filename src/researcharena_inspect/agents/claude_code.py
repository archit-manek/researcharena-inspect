"""The red-team agent as upstream runs it: Claude Code, through Inspect SWE.

Upstream runs each round as (`agents/claude/solve.sh:97-100` at the pinned commit)

    timeout --signal=TERM --kill-after=30s "${remaining}s" \
      claude --print --verbose --output-format stream-json --model "$AGENT_CONFIG" \
             --effort "$EFFORT" --thinking-display summarized \
             --dangerously-skip-permissions [--continue "<resume prompt>"]

with the first prompt on stdin, in `/home/user/task`, after installing the CLI. Inspect SWE's
`claude_code()` runs `claude --session-id <id> --dangerously-skip-permissions --model <model>
--print --output-format stream-json --verbose -- <prompt>` and, when called again on the same
state with a new user message, `claude --resume <id> ... -- <message>`: the same session
continued, as `--continue` does with the only session in the folder (upstream's runs keep one
session id across all rounds too). The model calls go through Inspect's bridge to the task's
model. What cannot be passed as upstream passes it is marked below with its deviation id:
the login (D-076), the effort and thinking flags (D-077), stdin and `--continue`
(D-078), the CLI build (D-079) and Inspect SWE's own settings (D-080).

Inspect SWE keeps none of the CLI's stdout, which upstream saves as `red_team/solve_out.json`
and turns into the monitor's transcripts. `capture_cli_output` copies each stdout line and
stderr chunk of the CLI as Inspect SWE reads them, and its exit code, for the rounds run
inside it.
"""

import importlib
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from inspect_ai.agent import Agent
from inspect_ai.util import sandbox

from researcharena_inspect.environment.layout import TASK_DIR

CLI_INSTALL_PATH = "/usr/local/bin/claude"
"""Where the installed CLI is linked, so `claude` is on the agent's PATH (upstream puts
`$HOME/.agent_cli/bin` first, `solve.sh:37`) and Inspect SWE finds it without reinstalling."""

# The bash tool limits of upstream's agent: solve.sh:26 exports BASH_MAX_TIMEOUT_MS=36000000
# (over the pipeline's 3600000); pipeline_redteam.sh:369 passes BASH_DEFAULT_TIMEOUT_MS.
BASH_TIMEOUT_ENV = {"BASH_MAX_TIMEOUT_MS": "36000000", "BASH_DEFAULT_TIMEOUT_MS": "3600000"}


AUTO_MEMORY_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0"}
"""Keep Claude Code's auto-memory on, as it was upstream. Inspect SWE turns it off by default
(`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`); an explicit "0" turns it on. In 3 of the 45 published
Claude CUDA runs the agent read or wrote `~/.claude/projects/-home-user-task/memory/`."""


def claude_code_env(model: str, effort: str, redruntime: str, resume: bool) -> dict[str, str]:
    """The variables upstream sets for the agent process, which its tools can also read.

    `pipeline_redteam.sh:362-365` passes the job's settings into the container and `solve.sh`
    reads them; the agent sees them too (`env` in its shell). Credentials (`CLAUDE_OAUTH_TOKEN`,
    `HF_TOKEN`, `ANTHROPIC_API_KEY=""`) are not passed: the port authenticates through the
    bridge. Container-wide variables (`PATH`, `HF_HOME`, `CUDA_VISIBLE_DEVICES`) belong to the
    sandbox.
    """
    # D-076: no OAuth token and no ANTHROPIC_API_KEY="" (solve.sh:11-24); the bridge
    # authenticates with Inspect's API key.
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
        effort: the resolved effort (`default` sets none). Applied to the served model as
            `reasoning_effort`, which Inspect's Anthropic provider sends as adaptive thinking
            with `output_config.effort` and summarized thinking (`--effort`,
            `--thinking-display summarized` upstream).
        env: variables for the CLI process (`claude_code_env`).
        small_model: model for Claude Code's background and Haiku-class calls (web page
            summaries, some subagents). The task passes Haiku 4.5, which upstream's CLI used
            by default (`pins.CLAUDE_CODE_SMALL_MODEL`); `None` sends them to the task's
            model, which is Inspect SWE's default.
        user: sandbox user to run as; `None` for the sandbox's default.
    """
    from inspect_swe import claude_code

    # D-077: `--effort` and `--thinking-display summarized` (solve.sh:99) are not CLI flags
    # here; the effort is set on the served model and the bridge drops the CLI's own
    # generation settings. D-078: the prompt goes in as an argument, not on stdin, and later
    # rounds use `--resume <session id>` for `--continue`. D-105: Inspect SWE resumes only once
    # the conversation has an assistant message; after a round with none, the next round gets
    # a new agent and so a new session (`solver.AgentRounds`), since the same `--session-id`
    # again would be refused ("Session ID ... is already in use").
    # D-080: Inspect SWE adds its own environment (IS_SANDBOX,
    # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC,
    # CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS, MCP settings, the bridge's URL and model names).
    # The experimental-betas switch is kept: clearing it changes no tool the model is offered
    # through the bridge, and forcing tool search on hides most tools from the model (see
    # monitor/agents.py; measured with a mock model). Haiku-class calls go to `small_model`.
    return claude_code(
        cwd=TASK_DIR,
        effort=None if effort == "default" else effort,  # type: ignore[arg-type]
        # Installed before the budget starts (install_claude_code); "sandbox" reuses it.
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
            `claude` already in the image. Upstream installs the latest npm release unless
            `CLAUDE_CLI_VERSION` is set (`src/utils/update_agent_cli.sh:38-43`).
        user: sandbox user to install for.
    """
    from inspect_swe._claude_code.agentbinary import claude_code_binary_source
    from inspect_swe._util.agentbinary import ensure_agent_binary_installed

    # D-079: Anthropic's native build of the same version, not the npm package; linked into
    # /usr/local/bin rather than installed in ~/.agent_cli. A failed install fails the sample
    # (D-083), where upstream's solve.sh exits 1 and the untouched workspace is scored.
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
    """Wrap the function through which Inspect SWE reads the CLI's output, once.

    `inspect_swe._claude_code.claude_code.claude_code_event_stream` (inspect_swe 0.2.x) turns
    the CLI's stdout into JSON events. The wrapper passes every event on unchanged and, when a
    capture is open in the current context, also keeps each stdout line, stderr chunk and exit
    code. Outside a capture (the monitor, other tasks) it changes nothing.
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

    Stdout lines are kept as Inspect SWE splits them: one JSON event per line, stripped, blank
    lines dropped. Joined with newlines they are the `stream-json` that upstream's
    `solve_out.json` holds after its non-JSON preamble. `exit_codes` gets one entry per CLI
    process that exited (none for one cancelled before it exited), including the exit that
    Inspect SWE does not raise for: 1 after a refusal (see `solver.AgentRounds`, D-084).
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
