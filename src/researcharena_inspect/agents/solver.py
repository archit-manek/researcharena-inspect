"""The red-team agent phase as a solver: Phase 3 of `src/pipeline_redteam.sh` and `solve.sh`.

Order of events, with upstream's lines at the pinned commit:

1. `RED_START` (`pipeline_redteam.sh:350`); everything below runs under `RED_TIMEOUT`, the
   budget plus five minutes (`:344`, `:357`).
2. Claude Code only: install the CLI and record its version (`solve.sh:35-38`).
3. Start the agent's clock (`solve.sh:59-65`) and run the rounds: the first prompt, then
   "Continue where you left off..." until the budget is spent (`agents.continuation`).
4. Stop everything the agent started (`agents.processes`), as the container teardown and
   `free_gpu_memory` do (`:415`), and record `RED_END` (`:403-406`).

The sample then goes on to scoring with the workspace as the agent left it, whether the agent
finished, ran out of time or failed, as upstream's pipeline goes on to Phase 4.

Every agent runs as the environment's agent user (`environment.AGENT_USER`) with upstream's
container variables (`environment.agent_env`), in the staged workspace, after its shell has
sourced `pre_agent_setup.sh` (`:378`: `source /home/user/pre_agent_setup.sh && bash
/home/user/agent_solve.sh`). Two agents call no model: `null` does nothing and exits at once, so
the staged workspace is scored as it was staged; `replay` puts a given workspace (a published
run's `task_snapshot/` or a local one) in place of the agent's work (`agents.replay`).
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cache

import anyio
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, apply_limits, sandbox, span, time_limit

from researcharena_inspect.agents.budget import Budget, red_timeout_seconds
from researcharena_inspect.agents.claude_code import (
    capture_cli_output,
    claude_code_env,
    install_claude_code,
    red_team_claude_code,
)
from researcharena_inspect.agents.continuation import (
    RC_BUDGET,
    RC_FAILED,
    RC_OK,
    ContinuationResult,
    Phase,
    run_with_continuation,
)
from researcharena_inspect.agents.processes import kill_new_processes, list_processes
from researcharena_inspect.agents.react import red_team_react
from researcharena_inspect.agents.record import AgentRun
from researcharena_inspect.agents.replay import (
    ReplayMode,
    replay_source,
    replay_workspace,
    snapshot_files,
)
from researcharena_inspect.environment.layout import (
    AGENT_USER,
    PRE_AGENT_SETUP,
    TASK_DIR,
    agent_env,
)
from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.upstream import upstream_dir

AGENTS = ("claude_code", "react", "null", "replay")
MODEL_AGENTS = ("claude_code", "react")
"""The agents that call the task's model; `null` and `replay` call none."""

_PATH_MARKER = "__researcharena_agent_path__="

CLEANUP_TIMEOUT_S = 60
"""Bound on stopping the agent's processes after the run, even when the sample is cancelled."""


@cache
def resolve_agent_effort(setting: str, model: str, provider: str = "claude") -> str:
    """The red-team effort from upstream's `effort.json`, by upstream's `src/resolve_effort.py`.

    As `pipeline_redteam.sh:77-78` calls it. Both agents use the `claude` entry (`xhigh` at the
    pinned commit), so ReAct runs at the same effort as Claude Code. `default` means none.
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

    The script prepends the workspace to `PATH`, checks CUDA and runs the workspace's own
    `hook_pre_agent.sh` if there is one (`src/utils/pre_agent_setup.sh`). The `PATH` it leaves
    is what the agent's process inherits, so it is read back and given to the agent.
    """
    script = f'source {PRE_AGENT_SETUP} && printf "\\n%s%s\\n" "{_PATH_MARKER}" "$PATH"'
    result = await sandbox().exec(["bash", "-c", script], user=AGENT_USER, env=env, cwd=TASK_DIR)
    out = result.stdout
    path = None
    if result.success and _PATH_MARKER in out:
        out, _, tail = out.rpartition(f"\n{_PATH_MARKER}")
        path = tail.strip("\n")
    return PreAgentSetup(result.returncode, out + result.stderr, path)


class AgentRounds:
    """Runs one round of an agent on a shared conversation, for the continuation loop.

    A resume round appends the continuation prompt as a user message; Claude Code then runs
    `claude --resume <session> -- <prompt>` and ReAct carries on from the conversation.
    """

    def __init__(self, agent: Agent, state: AgentState, name: str, log: list[str]) -> None:
        self.agent = agent
        self.state = state
        self.name = name
        self.log = log

    async def __call__(self, phase: Phase, prompt: str, remaining: int) -> int:
        if phase == "resume":
            self.state.messages.append(ChatMessageUser(content=prompt))
        # `timeout "${remaining}s"` (solve.sh:97). The agent changes `self.state` in place, so
        # its work is kept when the limit stops it or it raises. D-081: at the limit Inspect
        # cancels the round and kills the CLI's process; upstream sends SIGTERM and SIGKILLs
        # 30 s later.
        with apply_limits([time_limit(remaining)], catch_errors=True) as scope:
            try:
                async with span(name=f"{self.name} {phase}", type="agent"):
                    self.state = await self.agent(self.state)
            except LimitExceededError:
                raise  # a limit of the sample, not of this round
            except Exception as ex:
                # The CLI exited non-zero or the model call failed: the round ends with a
                # non-zero code, and the loop decides (solve.sh:105-133). D-084: Inspect SWE
                # treats the CLI's exit after a refusal as success, so such a round counts as 0.
                self.log.append(f"[agent] {phase} error: {ex!r}")
                return RC_FAILED
        return RC_BUDGET if scope.limit_error is not None else RC_OK


@solver
def red_team_agent(
    agent: str = "claude_code",
    effort: str = "auto",
    claude_code_version: str = "latest",
    small_model: str | None = None,
    react_compaction: bool = True,
    replay_mode: ReplayMode = "fresh",
) -> Solver:
    """Run the red-team agent on the sample's prompt within upstream's time budget.

    Reads `setting`, `redruntime` and `resume_agent` from the sample's metadata (see
    `task.red_team_sample`), and `replay` for the replay agent, and keeps the run in the store
    as an `AgentRun`.

    Args:
        agent: `claude_code` (Inspect SWE), `react`, `null` (does nothing; no model call) or
            `replay` (puts the sample's `replay` workspace in place; no model call).
        effort: the agent's reasoning effort; `auto` reads upstream's `effort.json`
            (`xhigh`), `default` sets none.
        claude_code_version: Claude Code version to install (`latest`, `stable`, `2.1.162`,
            ..., or `sandbox` for one already in the image). Upstream installs the latest.
        small_model: model for Claude Code's Haiku-class background calls; `None` uses the
            task's model.
        react_compaction: let ReAct summarise its conversation near the context limit.
        replay_mode: for `replay`: `fresh` (without the scorer's cached results) or
            `as_published` (see `agents.replay`).
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
        log: list[str] = []
        agent_state = AgentState(messages=list(state.messages))
        rounds: AgentRounds | None = None
        result = ContinuationResult()

        keep = await list_processes()
        # D-085: upstream's RED_START (pipeline_redteam.sh:350) comes before the container
        # starts and runs pre_agent_setup.sh (:357-378), so RED_TIMEOUT and time_taken.txt
        # include both. Here the sandbox is up and the setup solver has run before this point.
        run.red_start = time.time()
        # The container's `--env` values (pipeline_redteam.sh:359-369); credentials are not
        # passed (D-076, D-091).
        env = agent_env(
            redruntime=redruntime,
            resume_agent=resume,
            agent_config=model.name,
            agent_effort=resolved,
        )
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
                    files = await anyio.to_thread.run_sync(snapshot_files, source)
                    replayed = await replay_workspace(files, replay_mode)
                    run.replay = {"source": source.source, **replayed}
                    no_model_exit = 0
                else:
                    if agent == "claude_code":
                        cli = await install_claude_code(claude_code_version, user=AGENT_USER)
                        run.cli_path, run.cli_version, run.cli_requested = (
                            cli.path,
                            cli.version,
                            cli.requested,
                        )
                    rounds = AgentRounds(model_agent(), agent_state, agent, log)
                    budget = Budget(redruntime)
                    run.budget_seconds = budget.total_seconds
                    run.budget_start = budget.start
                    with capture_cli_output(stdout, stderr):
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
            if rounds is not None:
                agent_state = rounds.state
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
        return state

    return solve
