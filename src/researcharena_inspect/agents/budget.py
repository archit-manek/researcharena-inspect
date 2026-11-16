"""The red-team agent's time budget, as upstream computes and enforces it.

Two clocks at the pinned commit:

- `src/pipeline_redteam.sh:344` sets `RED_TIMEOUT = ceil(h * 3600) + 300` and runs the whole
  agent container under `timeout --signal=TERM --kill-after=60s` (`:357`): container start,
  `pre_agent_setup.sh`, the CLI install and every agent round.
- `agents/claude/solve.sh:59-65` starts the agent's own clock after the CLI install
  (`update_agent_cli.sh`, `:38`): `DEADLINE_TS = START_TS + max(1, int(h * 3600))`, with
  `START_TS=$(date +%s)`. Each round runs under `timeout ... "${remaining}s"` (`:97`), where
  `remaining = DEADLINE_TS - $(date +%s)`, floored at 0 (`:68-72`).

`Budget` is the second clock; `red_timeout_seconds` the first. Both count wall-clock time:
model latency, tool runs, rate-limit waits and the CLI's start-up between rounds all use it up.
"""

import math
import time
from collections.abc import Callable

from researcharena_inspect.prompts import budget_seconds, format_hours

Clock = Callable[[], float]
"""Seconds since the epoch, like `time.time`; tests pass a fake one."""

RED_TIMEOUT_GRACE_S = 300
"""The "+ 5min grace" of `pipeline_redteam.sh:344`."""

ROUND_KILL_AFTER_S = 30
"""`timeout --kill-after=30s` on each round (`solve.sh:97`): a CLI that ignores SIGTERM is
killed this long after the deadline. Inspect cancels the round at the deadline instead."""


def red_timeout_seconds(redruntime: str | int | float) -> int:
    """`RED_TIMEOUT`: the hard limit on the whole agent phase (`pipeline_redteam.sh:344`)."""
    # python3 -c "import math; print(int(math.ceil($REDRUNTIME * 3600)) + 300)"
    return math.ceil(float(format_hours(redruntime)) * 3600) + RED_TIMEOUT_GRACE_S


class Budget:
    """The agent's deadline, started when the agent is about to run (`solve.sh:59-72`).

    Upstream reads the clock with `date +%s`, so the start and every reading are whole
    seconds; this class truncates the same way.
    """

    def __init__(self, redruntime: str | int | float, clock: Clock = time.time) -> None:
        self._clock = clock
        self.hours = format_hours(redruntime)
        """The budget in hours as the command line gave it (`TIME_BUDGET_HOURS`)."""
        self.total_seconds = budget_seconds(redruntime)
        self.start = int(clock())
        self.deadline = self.start + self.total_seconds

    def remaining(self) -> int:
        """Whole seconds left, never negative (`time_left_seconds`)."""
        return max(0, self.deadline - int(self._clock()))

    def elapsed(self) -> float:
        """Seconds since the start, unrounded."""
        return self._clock() - self.start
