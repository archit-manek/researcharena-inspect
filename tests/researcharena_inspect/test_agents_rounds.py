"""One round of the agent as `solve.sh:97` runs it: its exit code and its stop at the deadline.

No Docker and no model: the agents are scripted, the CLI's exit codes are put where
`capture_cli_output` would put them, and `signal_cli` is replaced by a recorder. The Docker
tests (`test_agents_docker.py`) run the same paths with the real CLI.
"""

import importlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.util import ExecCompleted, ExecStderr, ExecStdout

from researcharena_inspect.agents import (
    RC_BUDGET,
    RC_FAILED,
    RC_KILLED,
    RC_OK,
    Budget,
    run_with_continuation,
)
from researcharena_inspect.agents.claude_code import capture_cli_output
from researcharena_inspect.agents.solver import AgentRounds, CliStop

from .solve_sh_oracle import START_EPOCH, run_solve_sh

REFUSAL_RC = 1
"""What Claude Code 2.1.162 exits with after a refusal (`test_agents_docker.py` checks it)."""


async def _fake_proc(chunks: list[object]) -> AsyncIterator[object]:
    for chunk in chunks:
        yield chunk


async def test_cli_exit_codes_are_captured_with_the_output() -> None:
    module = importlib.import_module("inspect_swe._claude_code.claude_code")
    refusal = [
        ExecStdout(data='{"type":"result","is_error":true,"stop_reason":"refusal"}\n'),
        ExecCompleted(exit_code=REFUSAL_RC),
    ]
    failure = [ExecStderr(data="boom\n"), ExecCompleted(exit_code=2)]
    stdout: list[str] = []
    stderr: list[str] = []
    codes: list[int] = []
    with capture_cli_output(stdout, stderr, codes):
        for chunks in (refusal, failure):
            events = [e async for e in module.claude_code_event_stream(_fake_proc(chunks))]
            assert type(events[-1]).__name__ == "ExitEvent"  # passed on unchanged
    assert codes == [REFUSAL_RC, 2]
    assert stderr == ["boom\n"]
    assert len(stdout) == 1
    # Without a list for them, exit codes are not kept; the output still is.
    with capture_cli_output(stdout, stderr):
        [e async for e in module.claude_code_event_stream(_fake_proc(refusal))]
    assert codes == [REFUSAL_RC, 2]
    assert len(stdout) == 2


def _agent(fn: Any) -> Agent:
    return cast(Agent, fn)


def _cli_agent(codes: list[int], exit_code: int | None, raises: bool = False) -> Agent:
    """A Claude Code round whose CLI exits with `exit_code`, as Inspect SWE handles it."""

    async def run(state: AgentState) -> AgentState:
        if exit_code is not None:
            codes.append(exit_code)  # what the tee records
        if raises:
            raise RuntimeError(f"Error executing claude code agent {exit_code}: boom")
        return state

    return _agent(run)


def _rounds(agent: Agent, codes: list[int] | None, log: list[str]) -> AgentRounds:
    cli = None if codes is None else CliStop(keep=[], exit_codes=codes)
    return AgentRounds(agent, AgentState(messages=[]), "claude_code", log, cli=cli)


@pytest.mark.parametrize(
    ("exit_code", "raises", "expected", "logged"),
    [
        (0, False, RC_OK, False),
        # Inspect SWE returns normally after a refusal (exit 1, no stderr, content_filter):
        # the round's code is the CLI's 1 (D-084), not 0.
        (REFUSAL_RC, False, REFUSAL_RC, False),
        # Any other non-zero exit raises; the CLI's own code is kept.
        (2, True, 2, True),
        (1, True, 1, True),
        # A failure with no exit code from the CLI (a bridge error, say).
        (None, True, RC_FAILED, True),
        (None, False, RC_OK, False),
    ],
)
async def test_claude_code_round_returns_the_clis_exit_code(
    exit_code: int | None, raises: bool, expected: int, logged: bool
) -> None:
    codes: list[int] = [0, 0]  # earlier rounds' codes are not this round's
    log: list[str] = []
    rounds = _rounds(_cli_agent(codes, exit_code, raises), codes, log)
    assert await rounds("exec", "p", 60) == expected
    assert bool(log) == logged
    assert rounds.deadline_stops == []


async def test_react_round_codes_are_unchanged() -> None:
    async def fails(state: AgentState) -> AgentState:
        raise RuntimeError("model error")

    async def works(state: AgentState) -> AgentState:
        return state

    log: list[str] = []
    assert await _rounds(_agent(fails), None, log)("exec", "p", 60) == RC_FAILED
    assert log == ["[agent] exec error: RuntimeError('model error')"]
    assert await _rounds(_agent(works), None, [])("exec", "p", 60) == RC_OK


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_refusals_stop_the_loop_as_upstreams_solve_sh(tmp_path: Path) -> None:
    # Round 0 is refused (resumed anyway), round 1 ends normally, round 2 is refused: upstream
    # stops there with "resume failed rc=1". Same rounds, codes and log lines as solve.sh.
    script = [(100, REFUSAL_RC), (200, 0), (300, REFUSAL_RC), (400, 0)]
    upstream = run_solve_sh(tmp_path, script, hours="1", resume=True, prompt="p")

    clock = FakeClock(START_EPOCH)
    codes: list[int] = []
    calls = 0

    async def cli(state: AgentState) -> AgentState:
        nonlocal calls
        duration, rc = script[calls]
        calls += 1
        clock.now += duration
        codes.append(rc)  # Inspect SWE returns normally for a refusal's exit 1
        return state

    log: list[str] = []
    rounds = _rounds(_agent(cli), codes, log)
    result = await run_with_continuation(rounds, Budget("1", clock=clock), "p", True, log.append)

    assert [r.rc for r in result.rounds] == [REFUSAL_RC, 0, REFUSAL_RC]
    assert len(upstream.calls) == calls == 3
    assert log == upstream.stderr
    assert "[agent] resume failed rc=1; stopping resume loop" in log
    assert result.exit_code == upstream.exit_code == REFUSAL_RC


# --- The stop at the deadline -----------------------------------------------------------------

REMAINING = 1
GRACE = 0.5


class FakeSignals:
    """Stands in for `signal_cli`: records when each signal is sent and lets the CLI react."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, float]] = []
        self.on_term: anyio.Event | None = None
        self.start = 0.0

    def begin(self) -> anyio.Event:
        """Start the clock (inside the test's event loop); the event is set on SIGTERM."""
        self.start = anyio.current_time()
        self.on_term = anyio.Event()
        return self.on_term

    async def __call__(self, signal: str, keep: list[int], sandbox_name: str | None = None):
        self.sent.append((signal, anyio.current_time() - self.start))
        if signal == "TERM" and self.on_term is not None:
            self.on_term.set()
        return [f"42 /usr/local/bin/claude --print ({signal})"]


@pytest.fixture
def signals(monkeypatch: pytest.MonkeyPatch) -> FakeSignals:
    fake = FakeSignals()
    monkeypatch.setattr("researcharena_inspect.agents.solver.signal_cli", fake)
    return fake


async def test_cli_that_exits_on_sigterm_ends_the_round_with_124(signals: FakeSignals) -> None:
    codes: list[int] = []
    term = signals.begin()

    async def cli(state: AgentState) -> AgentState:
        await term.wait()
        await anyio.sleep(0.1)  # the CLI's shutdown
        codes.append(143)
        raise RuntimeError("Error executing claude code agent 143: ")

    log: list[str] = []
    rounds = AgentRounds(
        _agent(cli),
        AgentState(messages=[]),
        "claude_code",
        log,
        cli=CliStop(keep=[1], exit_codes=codes, kill_after=GRACE),
    )
    start = anyio.current_time()
    assert await rounds("resume", "p", REMAINING) == RC_BUDGET
    took = anyio.current_time() - start
    # SIGTERM at the deadline, not before; no SIGKILL; no error line for the CLI's 143.
    assert [s for s, _ in signals.sent] == ["TERM"]
    assert signals.sent[0][1] == pytest.approx(REMAINING, abs=0.2)
    assert took < REMAINING + GRACE
    assert log == []
    [stop] = rounds.deadline_stops
    assert stop.cli_exit_code == 143
    assert stop.terminated
    assert stop.term_at_s == pytest.approx(REMAINING, abs=0.2)
    assert not stop.killed_after_grace
    assert not stop.killed
    assert stop.ended_after_term_s is not None
    assert stop.ended_after_term_s < GRACE
    assert stop.rc == RC_BUDGET


async def test_cli_that_ignores_sigterm_is_killed_after_the_grace(signals: FakeSignals) -> None:
    signals.begin()

    async def cli(state: AgentState) -> AgentState:
        await anyio.sleep(3600)  # ignores SIGTERM; only the cancellation stops it
        return state

    rounds = AgentRounds(
        _agent(cli),
        AgentState(messages=[]),
        "claude_code",
        [],
        cli=CliStop(keep=[1], exit_codes=[], kill_after=GRACE),
    )
    start = anyio.current_time()
    assert await rounds("exec", "p", REMAINING) == RC_KILLED
    took = anyio.current_time() - start
    assert [s for s, _ in signals.sent] == ["TERM", "KILL"]
    assert signals.sent[0][1] == pytest.approx(REMAINING, abs=0.2)
    assert signals.sent[1][1] == pytest.approx(REMAINING + GRACE, abs=0.2)
    assert took == pytest.approx(REMAINING + GRACE, abs=0.3)
    [stop] = rounds.deadline_stops
    assert stop.killed_after_grace
    assert stop.killed
    assert stop.ended_after_term_s == pytest.approx(GRACE, abs=0.3)
    assert stop.rc == RC_KILLED


async def test_round_that_ends_before_the_deadline_is_not_signalled(
    signals: FakeSignals,
) -> None:
    signals.begin()
    codes: list[int] = []
    rounds = _rounds(_cli_agent(codes, 0), codes, [])
    assert await rounds("exec", "p", REMAINING) == RC_OK
    await anyio.sleep(REMAINING + 0.2)
    assert signals.sent == []
    assert rounds.deadline_stops == []


async def test_signal_failure_still_stops_the_round(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(signal: str, keep: list[int], sandbox_name: str | None = None):
        raise RuntimeError("sandbox gone")

    monkeypatch.setattr("researcharena_inspect.agents.solver.signal_cli", broken)

    async def cli(state: AgentState) -> AgentState:
        await anyio.sleep(3600)
        return state

    rounds = AgentRounds(
        _agent(cli),
        AgentState(messages=[]),
        "claude_code",
        [],
        cli=CliStop(keep=[], exit_codes=[], kill_after=GRACE),
    )
    assert await rounds("exec", "p", REMAINING) == RC_KILLED
    [stop] = rounds.deadline_stops
    assert [e.split(":")[0] for e in stop.signal_errors] == ["TERM", "KILL"]


# --- D-105: a resume round before any assistant message -------------------------------------


def _session_agent(name: str, seen: list[tuple[str, list[str]]], answer: bool) -> Agent:
    """A Claude Code stand-in: one per CLI session; records which session ran on what."""

    async def run(state: AgentState) -> AgentState:
        seen.append((name, [m.role for m in state.messages]))
        if answer:
            state.messages.append(ChatMessageAssistant(content=f"answer from {name}"))
        return state

    return _agent(run)


async def test_resume_before_any_assistant_message_starts_a_new_session() -> None:
    # Round 0's model calls all fail (no assistant message); Inspect SWE would start round 1
    # with the same --session-id, which the CLI refuses. The round gets a new agent instead,
    # and later rounds resume that agent's session.
    seen: list[tuple[str, list[str]]] = []
    made = iter(["session 2", "session 3"])

    def new_agent() -> Agent:
        return _session_agent(next(made), seen, answer=True)

    state = AgentState(messages=[ChatMessageUser(content="first prompt")])
    codes: list[int] = []
    rounds = AgentRounds(
        _session_agent("session 1", seen, answer=False),
        state,
        "claude_code",
        [],
        cli=CliStop(keep=[], exit_codes=codes),
        new_agent=new_agent,
    )
    assert await rounds("exec", "first prompt", 60) == RC_OK
    assert await rounds("resume", "continue 1", 60) == RC_OK
    assert await rounds("resume", "continue 2", 60) == RC_OK
    assert seen == [
        ("session 1", ["user"]),
        # The new session starts with every user message so far: the first prompt, then
        # the continuation prompt (Inspect SWE joins them into the CLI's first prompt).
        ("session 2", ["user", "user"]),
        ("session 2", ["user", "user", "assistant", "user"]),
    ]
    assert rounds.new_sessions == [1]


async def test_resume_after_an_assistant_message_keeps_the_session() -> None:
    seen: list[tuple[str, list[str]]] = []

    def new_agent() -> Agent:
        raise AssertionError("no new session once the model has answered")

    rounds = AgentRounds(
        _session_agent("session 1", seen, answer=True),
        AgentState(messages=[ChatMessageUser(content="first prompt")]),
        "claude_code",
        [],
        cli=CliStop(keep=[], exit_codes=[]),
        new_agent=new_agent,
    )
    assert await rounds("exec", "first prompt", 60) == RC_OK
    assert await rounds("resume", "continue 1", 60) == RC_OK
    assert [name for name, _ in seen] == ["session 1", "session 1"]
    assert rounds.new_sessions == []


async def test_each_resume_without_an_answer_gets_its_own_session() -> None:
    # A new session that fails too leaves no assistant message, so the next round needs
    # another one: reusing it would hit the same refusal.
    seen: list[tuple[str, list[str]]] = []
    count = 0

    def new_agent() -> Agent:
        nonlocal count
        count += 1
        return _session_agent(f"session {count + 1}", seen, answer=False)

    rounds = AgentRounds(
        _session_agent("session 1", seen, answer=False),
        AgentState(messages=[ChatMessageUser(content="first prompt")]),
        "claude_code",
        [],
        cli=CliStop(keep=[], exit_codes=[]),
        new_agent=new_agent,
    )
    await rounds("exec", "first prompt", 60)
    await rounds("resume", "continue 1", 60)
    await rounds("resume", "continue 2", 60)
    assert [name for name, _ in seen] == ["session 1", "session 2", "session 3"]
    assert rounds.new_sessions == [1, 2]


async def test_react_rounds_never_get_a_new_agent() -> None:
    seen: list[tuple[str, list[str]]] = []
    rounds = AgentRounds(
        _session_agent("react", seen, answer=False),
        AgentState(messages=[ChatMessageUser(content="first prompt")]),
        "react",
        [],
    )
    await rounds("exec", "first prompt", 60)
    await rounds("resume", "continue 1", 60)
    assert [name for name, _ in seen] == ["react", "react"]
    assert rounds.new_sessions == []
