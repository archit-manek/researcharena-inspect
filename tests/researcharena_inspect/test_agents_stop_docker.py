"""The stop at a Claude Code round's deadline, on real processes in Docker (no model).

`timeout --signal=TERM --kill-after=30s` (`solve.sh:97`) sends SIGTERM to the CLI and its
process group at the deadline and SIGKILL after the grace. Here a stand-in CLI (bash named
`/usr/local/bin/claude` with Inspect SWE's flags, started through `exec_remote` as Inspect SWE
starts the real one) runs past a 3-second deadline with a 2-second grace, once exiting on
SIGTERM and once ignoring it. It has a child in its process group (which `timeout` reaches)
and one in its own session (as Claude Code's Bash tool is, which `timeout` does not reach).
Decoys check that nothing else is signalled. `test_agents_docker.py` runs the real CLI.
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import ExecCompleted, ExecRemoteStreamingOptions, sandbox

from researcharena_inspect.agents import RC_BUDGET, RC_KILLED
from researcharena_inspect.agents.processes import kill_new_processes, list_processes
from researcharena_inspect.agents.solver import AgentRounds, CliStop
from researcharena_inspect.task import red_team_sandbox

pytestmark = [pytest.mark.docker, pytest.mark.slow]

REMAINING = 3
GRACE = 2.0
STOP_TOLERANCE_S = 6
"""How long after the SIGKILL the round may take to end: sandbox exec round trips."""
CLI = "/usr/local/bin/claude"
FAKE_CLI = "/tmp/fake_cli.sh"
FLAGS = "--print --output-format stream-json --verbose -- 'do the task'"

FAKE_CLI_SCRIPT = r"""
[ "${IGNORE_TERM:-0}" = 1 ] && trap '' TERM
sleep 1000 &
setsid sleep 1001 &
wait
"""

# Started before the agent, so in `keep`: never signalled, though it looks like the CLI.
OLD_DECOY = f"exec -a {CLI} bash -c 'sleep 997; true' {FLAGS}"
# New, but `claude` without `--print`: not the round's CLI.
NEW_DECOY = f"exec -a {CLI} bash -c 'sleep 996; true' --version"


def _background(command: str) -> list[str]:
    return ["bash", "-c", f"setsid bash -c {_quote(command)} > /dev/null 2>&1 &"]


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def fake_cli_agent(codes: list[int], ignore_term: bool) -> Agent:
    """Runs the stand-in CLI as Inspect SWE runs Claude Code, raising for a non-zero exit."""

    async def run(state: AgentState) -> AgentState:
        proc = await sandbox().exec_remote(
            cmd=["bash", "-c", f"exec -a {CLI} bash {FAKE_CLI} {FLAGS}"],
            options=ExecRemoteStreamingOptions(env={"IGNORE_TERM": "1" if ignore_term else "0"}),
            stream=True,
        )
        async for event in proc:
            if isinstance(event, ExecCompleted):
                codes.append(event.exit_code)
                if event.exit_code != 0:
                    raise RuntimeError(f"Error executing claude code agent {event.exit_code}: ")
        return state

    return cast(Agent, run)


async def _commands() -> list[str]:
    result = await sandbox().exec(
        ["bash", "-c", "for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline; echo; done"]
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@solver
def stop_scenarios():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await sandbox().write_file(FAKE_CLI, FAKE_CLI_SCRIPT)
        await sandbox().exec(_background(OLD_DECOY))
        await anyio.sleep(0.5)
        keep = await list_processes()
        await sandbox().exec(_background(NEW_DECOY))
        results: dict[str, Any] = {}
        for name, ignore in (("exits_on_term", False), ("ignores_term", True)):
            codes: list[int] = []
            rounds = AgentRounds(
                fake_cli_agent(codes, ignore),
                AgentState(messages=[]),
                "claude_code",
                [],
                cli=CliStop(keep=keep, exit_codes=codes, kill_after=GRACE),
            )
            start = time.monotonic()
            rc = await rounds("exec", "do the task", REMAINING)
            results[name] = {
                "rc": rc,
                "took": time.monotonic() - start,
                "stops": [asdict(s) for s in rounds.deadline_stops],
                "codes": codes,
                "after": await _commands(),
            }
        results["killed_at_end"] = await kill_new_processes(keep)
        results["after_end"] = await _commands()
        state.metadata["scenarios"] = results
        return state

    return solve


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    task = Task(
        dataset=[Sample(input="x")],
        solver=stop_scenarios(),
        sandbox=red_team_sandbox("docker", None),
    )
    log_dir: Path = tmp_path_factory.mktemp("stop")
    [log] = eval(task, model="mockllm/model", display="none", log_dir=str(log_dir))
    assert log.status == "success", log.error
    assert log.samples is not None
    return log.samples[0].metadata["scenarios"]


def _signalled(lines: list[str], text: str) -> bool:
    return any(text in line for line in lines)


@pytest.mark.parametrize("name", ["exits_on_term", "ignores_term"])
def test_sigterm_reaches_the_cli_and_its_group_only(scenarios: dict[str, Any], name: str) -> None:
    [stop] = scenarios[name]["stops"]
    termed = stop["terminated"]
    assert _signalled(termed, f"{CLI} {FAKE_CLI} --print")  # the CLI
    assert _signalled(termed, "sleep 1000")  # its child in its process group
    assert not _signalled(termed, "sleep 1001")  # own session, like the Bash tool's shell
    assert not _signalled(termed, "sleep 997")  # was running before the agent
    assert not _signalled(termed, "sleep 996")  # `claude` without `--print`
    assert not _signalled(termed, "inspect-sandbox-tools")
    assert not _signalled(termed, "exec -a")  # the shell Inspect started the CLI with
    assert len(termed) == 2


def test_cli_that_exits_on_sigterm_ends_the_round_with_124(scenarios: dict[str, Any]) -> None:
    result = scenarios["exits_on_term"]
    [stop] = result["stops"]
    assert result["rc"] == RC_BUDGET
    assert result["codes"] == [143]  # bash killed by SIGTERM
    assert stop["cli_exit_code"] == 143
    assert stop["term_at_s"] == pytest.approx(REMAINING, abs=0.1)
    assert not stop["killed_after_grace"]
    assert stop["killed"] == []
    assert stop["ended_after_term_s"] < GRACE
    assert REMAINING <= result["took"] < REMAINING + GRACE
    assert not _signalled(result["after"], f"{FAKE_CLI} --print")
    assert not _signalled(result["after"], "sleep 1000")
    assert _signalled(result["after"], "sleep 1001")  # left for the end of the agent phase


def test_cli_that_ignores_sigterm_is_killed_after_the_grace(scenarios: dict[str, Any]) -> None:
    result = scenarios["ignores_term"]
    [stop] = result["stops"]
    assert result["rc"] == RC_KILLED
    assert result["codes"] in ([], [137])  # cancelled, or its SIGKILL seen first
    assert stop["term_at_s"] == pytest.approx(REMAINING, abs=0.1)
    assert stop["killed_after_grace"]
    assert _signalled(stop["killed"], f"{CLI} {FAKE_CLI} --print")
    assert _signalled(stop["killed"], "sleep 1000")
    # SIGKILL goes out GRACE s after SIGTERM; the round then ends once the SIGKILL's exec and
    # Inspect's kill of the cancelled process return (about 2.5 s with Docker Desktop on a Mac).
    assert GRACE <= stop["ended_after_term_s"] <= GRACE + STOP_TOLERANCE_S
    assert result["took"] >= REMAINING + GRACE
    assert not _signalled(result["after"], f"{FAKE_CLI} --print")
    assert not _signalled(result["after"], "sleep 1000")


def test_what_is_left_is_killed_after_the_last_round(scenarios: dict[str, Any]) -> None:
    killed = scenarios["killed_at_end"]
    assert _signalled(killed, "sleep 1001")
    assert _signalled(killed, "sleep 996")
    assert not _signalled(killed, "sleep 997")  # predates the agent
    assert not _signalled(scenarios["after_end"], "sleep 1001")
