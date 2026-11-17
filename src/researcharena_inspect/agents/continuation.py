"""Upstream's re-prompting rule: run the agent, then resume it until the budget is spent.

A line-for-line port of the control flow in `agents/claude/solve.sh:87-138` at the pinned
commit (Codex and OpenCode share it: `agents/codex/solve.sh:83-146`,
`agents/opencode/solve.sh:151-216`, the latter with an extra fast-exit check not ported here):

1. Round 0 ("exec") gets the prompt. A round is skipped with code 124 if no time is left;
   otherwise it runs with the time left as its limit and returns the CLI's exit code, 124 if
   the limit stopped it.
2. If `AGENT_RESUME` is false, stop. Otherwise, if round 0 did not end with 124 and time is
   left, loop: stop if no time is left; run a "resume" round with `resume_prompt(time left)`;
   stop if it returned 124 (budget used) or anything other than 0 (failure); else repeat.
   A failed round 0 (code 1, say) is still resumed; a failed resume round is not.
3. There is no cap on the number of rounds.

The agent behind a round is supplied by the caller (`RoundRunner`), so the same loop drives
Claude Code and ReAct. Each round's log line is the one `solve.sh` writes to stderr.
"""

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from researcharena_inspect.agents.budget import Budget
from researcharena_inspect.prompts import resume_prompt

Phase = Literal["exec", "resume"]

RC_OK = 0
RC_BUDGET = 124
"""GNU `timeout`'s exit code when it stopped the command, and the skip code of
`run_claude_with_remaining_budget` (`solve.sh:92-95`)."""
RC_FAILED = 1
"""What the port records for a failed round whose real exit code it does not have (the model
call or the agent raised; the CLI's own non-zero exit code is recorded when there is one)."""
RC_KILLED = 137
"""What `timeout --kill-after` leaves when it had to SIGKILL the command (`solve.sh:97`): it
signals its own process group, so `timeout` itself dies of SIGKILL and bash reports 128 + 9.
Measured with GNU coreutils 8.32 (Ubuntu 22.04, the base of upstream's image)."""

RoundRunner = Callable[[Phase, str, int], Awaitable[int]]
"""Runs one round: `(phase, prompt, remaining_seconds) -> exit code` (0, 124 or other)."""

Log = Callable[[str], None]


@dataclass
class Round:
    """One launch of the agent."""

    phase: Phase
    number: int
    """0 for the first run, then the resume round number (`RESUME_ROUND`)."""
    prompt: str
    remaining_at_start: int
    """Seconds left when the round launched: its time limit."""
    started_at: float
    """Seconds since the budget started."""
    rc: int
    ended_at: float


@dataclass
class ContinuationResult:
    rounds: list[Round] = field(default_factory=list)
    exit_code: int = RC_OK
    """`AGENT_RC` at the end, which `solve.sh` exits with (`:138`)."""

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.rounds]


async def run_with_continuation(
    run_round: RoundRunner,
    budget: Budget,
    prompt: str,
    resume: bool,
    log: Log,
    cli: str = "claude",
    result: ContinuationResult | None = None,
) -> ContinuationResult:
    """Run the agent and resume it as `solve.sh` does.

    Args:
        run_round: runs one round of the agent; see `RoundRunner`.
        budget: the agent's deadline, already started.
        prompt: the first prompt (round 0).
        resume: `resume_agent` from the setting's `config.json` (`AGENT_RESUME`).
        log: receives `solve.sh`'s stderr lines.
        cli: the name used in the log lines (`solve.sh` says `claude`).
        result: filled in as rounds finish, so a caller that is cancelled keeps the rounds
            run so far; a new one if omitted.
    """
    result = result if result is not None else ContinuationResult()

    async def launch(phase: Phase, number: int, text: str) -> int:
        remaining = budget.remaining()
        if remaining <= 0:
            log(f"[agent] {phase}: no budget left, skipping")
            return RC_BUDGET
        log(f"[agent] {phase}: launching with remaining budget {remaining}s")
        started = budget.elapsed()
        rc = await run_round(phase, text, remaining)
        result.rounds.append(
            Round(phase, number, text, remaining, started, rc, ended_at=budget.elapsed())
        )
        result.exit_code = rc
        return rc

    log(f"[agent] TIMEOUT_HOURS={budget.hours} total_timeout_seconds={budget.total_seconds}")
    log(f"[agent] phase 1/2: initial {cli} exec run")
    rc = await launch("exec", 0, prompt)
    remaining_after_exec = budget.remaining()
    log(f"[agent] exec finished rc={rc} remaining_seconds={remaining_after_exec}")

    if not resume:
        log("[agent] AGENT_RESUME=false: not resuming")
    elif rc != RC_BUDGET and remaining_after_exec > 0:
        number = 1
        while True:
            if budget.remaining() <= 0:
                log(f"[agent] no time left before resume round {number}; stopping")
                break
            log(f"[agent] phase 2/2 round {number}: {cli} --continue")
            rc = await launch("resume", number, resume_prompt(budget.remaining()))
            log(f"[agent] resume round {number} rc={rc} remaining={budget.remaining()}s")
            if rc == RC_BUDGET:
                log("[agent] resume used remaining budget")
                break
            if rc != RC_OK:
                log(f"[agent] resume failed rc={rc}; stopping resume loop")
                break
            number += 1
    else:
        log("[agent] skipping resume (exec used full budget or no time left)")

    result.exit_code = rc
    return result
