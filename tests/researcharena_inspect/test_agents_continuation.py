"""The budget and the continuation rule, against upstream's own `solve.sh` (no Docker, no model).

Each scenario scripts an agent's rounds as (seconds it works, exit code). Upstream's script runs
them on a simulated clock (`solve_sh_oracle`); the port's loop runs them on a fake clock. The
two must make the same rounds with byte-identical prompts, the same time limits, the same exit
code and the same log lines.
"""

import subprocess
from pathlib import Path

import pytest

from researcharena_inspect.agents import (
    RC_BUDGET,
    Budget,
    ContinuationResult,
    red_timeout_seconds,
    run_with_continuation,
)
from researcharena_inspect.agents.continuation import Phase
from researcharena_inspect.prompts import resume_prompt
from researcharena_inspect.upstream import upstream_dir

from .prompt_oracle import script_block
from .solve_sh_oracle import START_EPOCH, run_solve_sh


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def run_port_loop(
    rounds: list[tuple[int, int]], hours: str, resume: bool, prompt: str
) -> tuple[ContinuationResult, list[str], list[tuple[Phase, str, int]], int]:
    """The port's loop with the same scripted agent as the stub `claude` of the oracle."""
    clock = FakeClock(START_EPOCH)
    calls: list[tuple[Phase, str, int]] = []

    async def run_round(phase: Phase, text: str, remaining: int) -> int:
        duration, rc = rounds[min(len(calls), len(rounds) - 1)]
        calls.append((phase, text, remaining))
        if duration >= remaining:
            clock.now += remaining
            return RC_BUDGET
        clock.now += duration
        return rc

    log: list[str] = []
    budget = Budget(hours, clock=clock)
    result = await run_with_continuation(run_round, budget, prompt, resume, log.append)
    return result, log, calls, int(clock.now)


SCENARIOS = {
    # The agent stops early three times, then is still working when the budget runs out.
    "finishes_early_three_times": ([(600, 0), (1200, 0), (900, 0), (5000, 0)], "1", True),
    # It stops early again and again with little time left (the "0 hours 0 minutes" prompts).
    "stops_early_near_the_end": ([(3540, 0), (20, 0), (30, 0), (30, 0)], "1", True),
    # The first run uses the whole budget: no resume.
    "first_run_uses_the_budget": ([(5000, 0)], "1", True),
    # The first run ends exactly at the deadline: timeout stops it (code 124).
    "first_run_ends_at_the_deadline": ([(3600, 0)], "1", True),
    # The first run fails (code 1): upstream still resumes it.
    "failed_first_run_is_resumed": ([(100, 1), (200, 0), (4000, 0)], "1", True),
    # A resume round fails (code 2): the loop stops there.
    "failed_resume_stops": ([(100, 0), (200, 2), (100, 0)], "1", True),
    # resume_agent is false (the post-training settings): one run only.
    "no_resume": ([(100, 0), (100, 0)], "1", False),
    # A ten-minute budget, typed as upstream's smoke runs type it.
    "ten_minutes": ([(61, 0), (125, 0), (1000, 0)], "0.1667", True),
    # A budget given as a decimal number of hours.
    "half_hour_with_many_rounds": ([(300, 0)], "0.5", True),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_loop_matches_solve_sh(name: str, tmp_path: Path) -> None:
    rounds, hours, resume = SCENARIOS[name]
    prompt = "# Task\n\nDo the task.\nIt has `backticks`, $dollars and \\backslashes."
    upstream = run_solve_sh(tmp_path, rounds, hours=hours, resume=resume, prompt=prompt)
    result, log, calls, end = await run_port_loop(rounds, hours, resume, prompt)

    assert len(calls) == len(upstream.calls), name
    for (phase, text, remaining), call in zip(calls, upstream.calls, strict=True):
        assert text.encode("utf-8") == call.prompt.encode("utf-8")
        assert remaining == call.limit
        assert ("--continue" in call.argv) == (phase == "resume")
    assert result.exit_code == upstream.exit_code
    assert log == upstream.stderr
    assert end == upstream.end_clock
    assert [r.rc for r in result.rounds] == [
        124 if d >= c.limit else rc
        for (d, rc), c in zip(
            [rounds[min(i, len(rounds) - 1)] for i in range(len(calls))],
            upstream.calls,
            strict=True,
        )
    ]


async def test_resume_prompts_carry_the_time_left(tmp_path: Path) -> None:
    rounds = [(600, 0), (1200, 0), (900, 0), (5000, 0)]
    result, _, calls, _ = await run_port_loop(rounds, "1", True, "p")
    assert [remaining for _, _, remaining in calls] == [3600, 3000, 1800, 900]
    assert [text for phase, text, _ in calls if phase == "resume"] == [
        resume_prompt(3000),
        resume_prompt(1800),
        resume_prompt(900),
    ]
    assert "You have 0 hours 50 minutes of wall-clock time remaining." in calls[1][1]
    assert [r.number for r in result.rounds] == [0, 1, 2, 3]
    assert result.exit_code == RC_BUDGET


async def test_many_short_rounds_are_resumed_until_the_deadline() -> None:
    # No cap on rounds: an agent that stops after every 300 s of a half hour is resumed 5 times;
    # the sixth round has exactly 300 s left, so the budget stops it (124).
    result, _, calls, end = await run_port_loop([(300, 0)], "0.5", True, "p")
    assert len(calls) == 6
    assert end == START_EPOCH + 1800
    assert [r.rc for r in result.rounds] == [0, 0, 0, 0, 0, RC_BUDGET]


def test_budget_truncates_like_date() -> None:
    clock = FakeClock(1000.9)
    budget = Budget("0.1667", clock=clock)
    assert (budget.start, budget.total_seconds, budget.deadline) == (1000, 600, 1600)
    clock.now = 1599.99
    assert budget.remaining() == 1
    clock.now = 1600.5
    assert budget.remaining() == 0
    clock.now = 5000
    assert budget.remaining() == 0
    assert budget.hours == "0.1667"


@pytest.mark.parametrize("hours", ["1", "5", "0.5", "0.1667", "0.007", "2.25"])
def test_red_timeout_matches_the_pipeline(hours: str) -> None:
    script = upstream_dir() / "src" / "pipeline_redteam.sh"
    line = script_block(script, "RED_TIMEOUT=$(", "echo", include_end=False)
    out = subprocess.run(
        ["bash", "-c", f'REDRUNTIME="{hours}"\n{line}echo "$RED_TIMEOUT"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert red_timeout_seconds(hours) == int(out.stdout.strip())
