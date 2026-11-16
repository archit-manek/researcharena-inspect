"""The red-team agent: Claude Code (as upstream runs it) or ReAct, on upstream's time budget.

- `budget`: the agent's deadline and the hard limit on the agent phase.
- `continuation`: upstream's rule for re-prompting an agent that stops early.
- `claude_code`: Claude Code through Inspect SWE, with the CLI's output kept.
- `react`: Inspect's ReAct agent with a shell and a file editor.
- `processes`: stopping what the agent left running when its time is up.
- `solver`: the agent phase as a solver (`red_team_agent`).
- `record`: what a run leaves in the log, and the run folder it becomes for the monitor.
"""

from researcharena_inspect.agents.budget import Budget, red_timeout_seconds
from researcharena_inspect.agents.continuation import (
    RC_BUDGET,
    RC_FAILED,
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
