"""The red-team agent's two clocks, as upstream computes them. Both count wall-clock time.

- `red_timeout_seconds`: `RED_TIMEOUT` (`src/pipeline_redteam.sh:344`), the outer limit on the
  whole agent container (`:357`), including setup and the CLI install.
- `Budget`: the agent's own deadline (`agents/claude/solve.sh:59-72`), started after the CLI
  install (`:38`); each round gets the time left (`:97`).
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
killed this long after the deadline. The port's Claude Code rounds do the same
(`solver.AgentRounds`); its ReAct rounds are cancelled at the deadline."""


def red_timeout_seconds(redruntime: str | int | float) -> int:
    """`RED_TIMEOUT`: the hard limit on the whole agent phase (`pipeline_redteam.sh:344`)."""
    return math.ceil(float(format_hours(redruntime)) * 3600) + RED_TIMEOUT_GRACE_S


class Budget:
    """The agent's deadline (`solve.sh:59-72`), in whole seconds as upstream's `date +%s`."""

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
