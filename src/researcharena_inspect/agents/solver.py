"""The red-team agent phase as a solver: Phase 3 of `src/pipeline_redteam.sh` and `solve.sh`.

In order: `RED_START` (`pipeline_redteam.sh:350`) and the `RED_TIMEOUT` limit (`:344`, `:357`);
`source pre_agent_setup.sh` (`:378`); for Claude Code, the CLI install (`solve.sh:35-38`); the
agent's clock (`solve.sh:59-65`) and the rounds (`agents.continuation`); the process sweep
(`agents.processes`) and `RED_END` (`:403-406`). Scoring follows however the agent ended, except
that a model agent with no model output at all ends with an error (D-108).
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import cache

import anyio
from inspect_ai.agent import Agent, AgentState
from inspect_ai.event import Event, ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, apply_limits, sandbox, span, time_limit

from researcharena_inspect.agents.budget import ROUND_KILL_AFTER_S, Budget, red_timeout_seconds
from researcharena_inspect.agents.claude_code import (
    capture_cli_output,
    claude_code_env,
    install_claude_code,
    prompt_on_stdin,
    red_team_claude_code,
)
from researcharena_inspect.agents.continuation import (
    RC_BUDGET,
    RC_FAILED,
    RC_KILLED,
    RC_OK,
    ContinuationResult,
    Phase,
    run_with_continuation,
)
from researcharena_inspect.agents.processes import (
    kill_new_processes,
    list_processes,
    signal_cli,
)
from researcharena_inspect.agents.react import red_team_react
from researcharena_inspect.agents.record import AgentRun
from researcharena_inspect.agents.replay import (
    ReplayMode,
    replay_input,
    replay_source,
    replay_workspace,
)
from researcharena_inspect.environment.layout import (
    AGENT_USER,
    PRE_AGENT_SETUP,
    TASK_DIR,
    agent_env,
)
from researcharena_inspect.environment.services import replay_hook_evidence
from researcharena_inspect.environment.staging import agent_extra_env
from researcharena_inspect.pins import CLAUDE_CODE_SMALL_MODEL, UPSTREAM_COMMIT
from researcharena_inspect.upstream import upstream_dir

AGENTS = ("claude_code", "react", "null", "replay")
MODEL_AGENTS = ("claude_code", "react")
"""The agents that call the task's model; `null` and `replay` call none."""

_PATH_MARKER = "__researcharena_agent_path__="

CLEANUP_TIMEOUT_S = 60
"""Bound on stopping the agent's processes after the run, even when the sample is cancelled."""


class NoModelOutputError(RuntimeError):
    """A model agent got no response from any model call; the sample is not scored (D-108)."""


def _returned(event: Event) -> bool:
    """A model call that returned output to its caller: no error, no longer pending."""
    return isinstance(event, ModelEvent) and event.error is None and not event.pending


def model_output_seen() -> bool:
    """Whether any model call of the sample so far returned output (D-108). A refusal counts.

    Read from the transcript's `ModelEvent`s, Claude Code's bridge calls included. If the
    history cannot be read back (Inspect's bounded transcript), the answer is yes, so that
    doubt never fails a sample.
    """
    try:
        return transcript().history.find_last(_returned) is not None
    except Exception:  # TranscriptHistoryUnavailableError, or the store's own error
        return True


def _last_model_call() -> str:
    try:
        last = transcript().history.find_last(lambda e: isinstance(e, ModelEvent))
    except Exception:
        return "its model calls could not be read back"
    if not isinstance(last, ModelEvent):
        return "it made no model call"
    if last.error is None:
        return "its last model call did not finish"
    return f"its last model call failed: {' '.join(last.error.split())[:500]}"


def no_model_output_message(run: AgentRun) -> str:
    last_line = run.log[-1] if run.log else None
    return (
        f"No model output: the {run.agent} agent ({run.model}) got no response from any model "
        f"call, so the workspace is not scored (D-108); {_last_model_call()}. "
        f"Rounds: {len(run.rounds)}, exit code: {run.exit_code}, last log line: {last_line!r}. "
        "The run's record is in the sample's store (AgentRun) and metadata. "
        "fail_on_no_model_output=False scores the workspace as upstream does."
    )


@cache
def resolve_agent_effort(setting: str, model: str, provider: str = "claude") -> str:
    """The red-team effort from upstream's `effort.json`, by `src/resolve_effort.py`.

    As `pipeline_redteam.sh:77-78` calls it. Both agents use the `claude` entry, so ReAct runs
    at Claude Code's effort. `default` means none.
    """
    script = upstream_dir(UPSTREAM_COMMIT) / "src" / "resolve_effort.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            *("--setting", setting, "--team", "red_team"),
            *("--provider", provider, "--model", model),
        ],
        capture_output=True,
        check=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class PreAgentSetup:
    returncode: int
    output: str
    """What the script printed (upstream's `solve_out.json` starts with its stdout)."""
    path: str | None
    """`PATH` in the agent's shell afterwards; None if the script failed."""


async def source_pre_agent_setup(env: dict[str, str]) -> PreAgentSetup:
    """`source /home/user/pre_agent_setup.sh` in the agent's shell, as the agent's user.

    The script changes `PATH`, which the agent's process inherits upstream, so it is read back.
    """
    script = f'source {PRE_AGENT_SETUP} && printf "\\n%s%s\\n" "{_PATH_MARKER}" "$PATH"'
    result = await sandbox().exec(["bash", "-c", script], user=AGENT_USER, env=env, cwd=TASK_DIR)
    out = result.stdout
    path = None
    if result.success and _PATH_MARKER in out:
        out, _, tail = out.rpartition(f"\n{_PATH_MARKER}")
        path = tail.strip("\n")
    return PreAgentSetup(result.returncode, out + result.stderr, path)


@dataclass
class CliStop:
    """How a Claude Code round ends at its deadline: `timeout --signal=TERM --kill-after=30s`."""

    keep: list[int]
    """PIDs running before the agent started (`list_processes`); the CLI is among the others."""
    exit_codes: list[int]
    """The CLI's exit codes, one per process that exited, as `capture_cli_output` keeps them."""
    kill_after: float = ROUND_KILL_AFTER_S
    """Seconds from SIGTERM to SIGKILL (`--kill-after`)."""


@dataclass
class DeadlineStop:
    """What happened when a Claude Code round reached its deadline."""

    phase: Phase
    term_at_s: float = 0.0
    """Seconds from the round's launch to the SIGTERM: its time limit."""
    terminated: list[str] = field(default_factory=list)
    """Processes sent SIGTERM at the deadline ("<pid> <command line>"): the CLI and its group."""
    killed_after_grace: bool = False
    """Whether the round was still running when the grace ran out (then SIGKILL, cancel)."""
    killed: list[str] = field(default_factory=list)
    """Processes sent SIGKILL when the grace ran out."""
    ended_after_term_s: float | None = None
    """Seconds from the SIGTERM to the end of the round, however it ended."""
    cli_exit_code: int | None = None
    """The CLI's own exit code, if it exited before it was killed (0 or 143 measured)."""
    agent_error: str | None = None
    """What Inspect SWE raised for that exit (it raises for any code but 0 and a refusal's 1)."""
    signal_errors: list[str] = field(default_factory=list)
    rc: int = RC_BUDGET


class AgentRounds:
    """Runs one round of an agent on a shared conversation, for the continuation loop.

    Each round runs as upstream's `timeout "${remaining}s"` (`solve.sh:97`) and returns the code
    `timeout` would: the CLI's exit code, 124 at the deadline, or 137 if killed after the grace.

    Args:
        agent: the agent; it changes `state` in place, so its work is kept however it stops.
        state: the conversation, shared by the rounds.
        name: the agent's name, for spans and log lines.
        log: receives error lines.
        cli: for Claude Code, how to stop its CLI and read its exit code; None for ReAct.
        new_agent: for Claude Code, makes a new agent for D-105; None never replaces it.
    """

    def __init__(
        self,
        agent: Agent,
        state: AgentState,
        name: str,
        log: list[str],
        cli: CliStop | None = None,
        new_agent: Callable[[], Agent] | None = None,
    ) -> None:
        self.agent = agent
        self.state = state
        self.name = name
        self.log = log
        self.cli = cli
        self.new_agent = new_agent
        self.deadline_stops: list[DeadlineStop] = []
        self.new_sessions: list[int] = []
        """Indexes (0 = the first round) of the rounds that ran in a new CLI session."""
        self._launched = 0

    async def __call__(self, phase: Phase, prompt: str, remaining: int) -> int:
        if phase == "resume":
            # D-105: with no assistant message yet, Inspect SWE would reuse `--session-id`,
            # which the CLI refuses, so the round gets a new agent and session.
            if self.new_agent is not None and not any(
                isinstance(m, ChatMessageAssistant) for m in self.state.messages
            ):
                self.agent = self.new_agent()
                self.new_sessions.append(self._launched)
            self.state.messages.append(ChatMessageUser(content=prompt))
        self._launched += 1
        if self.cli is None:
            return await self._react_round(phase, remaining)
        return await self._cli_round(phase, remaining, self.cli)

    async def _react_round(self, phase: Phase, remaining: int) -> int:
        # D-081: no CLI to send SIGTERM; the round is cancelled at the deadline.
        with apply_limits([time_limit(remaining)], catch_errors=True) as scope:
            try:
                async with span(name=f"{self.name} {phase}", type="agent"):
                    self.state = await self.agent(self.state)
            except LimitExceededError:
                raise  # a limit of the sample, not of this round
            except Exception as ex:
                # The loop decides what follows (solve.sh:105-133).
                self.log.append(f"[agent] {phase} error: {ex!r}")
                return RC_FAILED
        return RC_BUDGET if scope.limit_error is not None else RC_OK

    async def _cli_round(self, phase: Phase, remaining: int, cli: CliStop) -> int:
        # D-081: `timeout --signal=TERM --kill-after=30s` (solve.sh:97) through sandbox execs,
        # so each signal lands slightly late; then the round is cancelled.
        launched = len(cli.exit_codes)
        stop: DeadlineStop | None = None
        error: Exception | None = None
        start = anyio.current_time()
        term_time = start
        with anyio.CancelScope() as round_scope:

            async def at_deadline() -> None:
                nonlocal stop, term_time
                await anyio.sleep(remaining)
                term_time = anyio.current_time()
                stop = DeadlineStop(phase, term_at_s=round(term_time - start, 3))
                try:
                    stop.terminated = await signal_cli("TERM", cli.keep)
                except Exception as ex:
                    stop.signal_errors.append(f"TERM: {ex!r}")
                await anyio.sleep_until(term_time + cli.kill_after)
                stop.killed_after_grace = True
                try:
                    stop.killed = await signal_cli("KILL", cli.keep)
                except Exception as ex:
                    stop.signal_errors.append(f"KILL: {ex!r}")
                round_scope.cancel()

            async with anyio.create_task_group() as tg:
                tg.start_soon(at_deadline)
                try:
                    async with span(name=f"{self.name} {phase}", type="agent"):
                        self.state = await self.agent(self.state)
                except Exception as ex:
                    error = ex  # raised outside the task group, which would wrap it
                tg.cancel_scope.cancel()
        ended = anyio.current_time()
        if isinstance(error, LimitExceededError):
            raise error  # a limit of the sample, not of this round
        codes = cli.exit_codes[launched:]
        exit_code = codes[-1] if codes else None
        if stop is not None:
            # `timeout` returns 124 once its limit has fired, whatever the CLI exits with.
            stop.cli_exit_code = exit_code
            stop.agent_error = None if error is None else repr(error)
            stop.ended_after_term_s = round(ended - term_time, 3)
            stop.rc = RC_KILLED if stop.killed_after_grace else RC_BUDGET
            self.deadline_stops.append(stop)
            return stop.rc
        if error is not None:
            # The loop decides what follows (solve.sh:105-133).
            self.log.append(f"[agent] {phase} error: {error!r}")
            return exit_code or RC_FAILED
        # D-084: Inspect SWE does not raise for a refusal's exit 1, so the code is read from
        # the CLI itself, as upstream's AGENT_RC is.
        return exit_code if exit_code is not None else RC_OK


@solver
def red_team_agent(
    agent: str = "claude_code",
    effort: str = "auto",
    claude_code_version: str = "latest",
    small_model: str | None = CLAUDE_CODE_SMALL_MODEL,
    react_compaction: bool = True,
    replay_mode: ReplayMode = "fresh",
    fail_on_no_model_output: bool = True,
) -> Solver:
    """Run the red-team agent on the sample's prompt within upstream's time budget.

    Reads the sample's metadata (`task.red_team_sample`) and keeps the run as an `AgentRun`.

    Args:
        agent: `claude_code` (Inspect SWE), `react`, `null` (does nothing; no model call) or
            `replay` (puts the sample's `replay` workspace in place; no model call).
        effort: the agent's reasoning effort; `auto` reads upstream's `effort.json`
            (`xhigh`), `default` sets none.
        claude_code_version: Claude Code version to install (`latest`, `stable`, `2.1.162`,
            ..., or `sandbox` for one already in the image).
        small_model: model for Claude Code's background calls; `None` uses the task's model.
        react_compaction: let ReAct summarise its conversation near the context limit.
        replay_mode: for `replay`: `fresh` or `as_published` (see `agents.replay`).
        fail_on_no_model_output: for `claude_code` and `react`: raise `NoModelOutputError`
            if no model call returned output (D-108); `False` scores as upstream does.
    """
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata
        model = get_model()
        resolved = resolve_agent_effort(meta["setting"], model.name) if effort == "auto" else effort
        redruntime = meta["redruntime"]
        resume = bool(meta["resume_agent"])
        prompt = state.input_text

        run = state.store_as(AgentRun)
        run.agent = agent
        run.model = model.name
        run.effort = resolved
        run.redruntime = redruntime
        run.resume = resume
        run.red_timeout_seconds = red_timeout_seconds(redruntime)
        stdout: list[str] = []
        stderr: list[str] = []
        cli_exit_codes: list[int] = []
        log: list[str] = []
        agent_state = AgentState(messages=list(state.messages))
        rounds: AgentRounds | None = None
        result = ContinuationResult()

        keep = await list_processes()
        # D-085: upstream's RED_START (pipeline_redteam.sh:350) also covers container start.
        run.red_start = time.time()
        # The container's `--env` values (pipeline_redteam.sh:359-369); credentials are not
        # passed (D-076, D-091).
        env = agent_env(
            redruntime=redruntime,
            resume_agent=resume,
            agent_config=model.name,
            agent_effort=resolved,
        )
        # `AGENT_EXTRA_ENV_ARGS` from the `pipeline_pre_agent.sh` hooks (:371).
        env.update(agent_extra_env(meta))
        no_model_exit: int | None = None

        def model_agent() -> Agent:
            if agent == "claude_code":
                return red_team_claude_code(
                    effort=resolved,
                    env=env | claude_code_env(model.name, resolved, redruntime, resume),
                    small_model=small_model,
                    user=AGENT_USER,
                )
            return red_team_react(
                model,
                resolved,
                compaction=react_compaction,
                user=AGENT_USER,
                env=env,
                cwd=TASK_DIR,
            )

        try:
            with apply_limits([time_limit(run.red_timeout_seconds)], catch_errors=True) as outer:
                setup = await source_pre_agent_setup(env)
                run.pre_agent_setup_rc = setup.returncode
                run.pre_agent_setup_output = setup.output
                env["PATH"] = setup.path or env["PATH"]
                if setup.path is None:
                    # `source pre_agent_setup.sh && bash agent_solve.sh`: no agent at all.
                    log.append(
                        f"[agent] pre_agent_setup.sh exited {setup.returncode}; "
                        "the agent did not start"
                    )
                elif agent == "null":
                    done = await sandbox().exec(["true"], user=AGENT_USER, env=env, cwd=TASK_DIR)
                    no_model_exit = done.returncode
                elif agent == "replay":
                    source = replay_source(meta["replay"])
                    given = await anyio.to_thread.run_sync(replay_input, source)
                    replayed = await replay_workspace(given.files, replay_mode, source.setting)
                    run.replay = {
                        "source": source.source,
                        **replayed,
                        "provenance": given.provenance,
                    }
                    if given.evidence is not None:  # D-172: a stored run's capture log
                        run.replay["evidence"] = await replay_hook_evidence(
                            given.evidence, has_hooks=bool(agent_extra_env(meta))
                        )
                    no_model_exit = 0
                else:
                    if agent == "claude_code":
                        cli = await install_claude_code(claude_code_version, user=AGENT_USER)
                        run.cli_path, run.cli_version, run.cli_requested = (
                            cli.path,
                            cli.version,
                            cli.requested,
                        )
                    cli_stop = (
                        CliStop(keep, cli_exit_codes, kill_after=ROUND_KILL_AFTER_S)
                        if agent == "claude_code"
                        else None
                    )
                    rounds = AgentRounds(
                        model_agent(),
                        agent_state,
                        agent,
                        log,
                        cli=cli_stop,
                        new_agent=model_agent if agent == "claude_code" else None,  # D-105
                    )
                    budget = Budget(redruntime)
                    run.budget_seconds = budget.total_seconds
                    run.budget_start = budget.start
                    # D-078, D-257: Claude Code's first prompt on stdin, as upstream's.
                    with capture_cli_output(stdout, stderr, cli_exit_codes), prompt_on_stdin():
                        await run_with_continuation(
                            rounds,
                            budget,
                            prompt,
                            resume,
                            log.append,
                            cli="claude" if agent == "claude_code" else agent,
                            result=result,
                        )
            if outer.limit_error is not None:
                run.outer_timeout = True
                log.append(f"[agent] stopped by RED_TIMEOUT ({run.red_timeout_seconds}s)")
        finally:
            with anyio.CancelScope(shield=True), anyio.move_on_after(CLEANUP_TIMEOUT_S):
                try:
                    run.killed = await kill_new_processes(keep)
                except Exception as ex:
                    log.append(f"[agent] could not stop the agent's processes: {ex!r}")
            run.red_end = time.time()
            run.rounds = result.to_dicts()
            run.exit_code = result.exit_code if result.rounds else no_model_exit
            run.stdout, run.stderr, run.log = stdout, stderr, log
            run.cli_exit_codes = cli_exit_codes
            if rounds is not None:
                agent_state = rounds.state
                run.deadline_stops = [asdict(s) for s in rounds.deadline_stops]
                run.new_sessions = list(rounds.new_sessions)
            state.messages = agent_state.messages
            if agent_state.output:
                state.output = agent_state.output
            state.metadata.update(
                agent=agent,
                agent_model=model.name,
                agent_effort=resolved,
                cli_version=run.cli_version,
                agent_rounds=len(run.rounds),
                agent_exit_code=run.exit_code,
                agent_time_taken_s=run.time_taken_s,
            )
        # D-108. Checked after the record is written, so the errored sample keeps it.
        if fail_on_no_model_output and agent in MODEL_AGENTS and not model_output_seen():
            raise NoModelOutputError(no_model_output_message(run))
        return state

    return solve
