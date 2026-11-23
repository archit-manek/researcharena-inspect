"""The red-team agent: Claude Code (as upstream runs it) or ReAct, on upstream's time budget.

`solver.red_team_agent` is the entry point; `record` turns a logged run into a run folder.
"""

from researcharena_inspect.agents.budget import Budget, red_timeout_seconds
from researcharena_inspect.agents.continuation import (
    RC_BUDGET,
    RC_FAILED,
    RC_KILLED,
    RC_OK,
    ContinuationResult,
    Round,
    run_with_continuation,
)
from researcharena_inspect.agents.record import AgentRun, write_run_dir, write_run_dirs
from researcharena_inspect.agents.solver import AGENTS, red_team_agent, resolve_agent_effort

__all__ = [
    "AGENTS",
    "RC_BUDGET",
    "RC_FAILED",
    "RC_KILLED",
    "RC_OK",
    "AgentRun",
    "Budget",
    "ContinuationResult",
    "Round",
    "red_team_agent",
    "red_timeout_seconds",
    "resolve_agent_effort",
    "run_with_continuation",
    "write_run_dir",
    "write_run_dirs",
]
