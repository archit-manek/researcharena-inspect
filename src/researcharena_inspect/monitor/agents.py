"""The monitor agents: Claude Code through Inspect SWE, or Inspect's ReAct agent.

Upstream runs `claude --print` with every tool in `/home/user/task` (`monitors/claude/monitor.sh`).
Inspect SWE passes the same flags, but the prompt as an argument rather than on stdin (D-078,
left as it is for the monitor), and sets the effort on the served model rather than by `--effort`.

The ReAct monitor has no upstream counterpart: a shell and a file editor, no system prompt
beyond the monitor prompt, and it stops when it stops calling tools, as `claude --print` does.
"""

from copy import copy

from inspect_ai.agent import Agent, react
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.tool import bash, text_editor

from researcharena_inspect.monitor.staging import SANDBOX_TASK_DIR
from researcharena_inspect.pins import CLAUDE_CODE_SMALL_MODEL

MONITOR_AGENTS = ("claude_code", "react")

# monitor.sh:22 exports BASH_MAX_TIMEOUT_MS; pipeline_monitor.sh:298 passes
# BASH_DEFAULT_TIMEOUT_MS into the container.
# D-089: of Inspect SWE's own CLI switches, auto-memory is turned back on; experimental betas
# and non-essential traffic stay off (clearing betas changes no tool the bridge offers).
CLAUDE_CODE_ENV = {
    "BASH_MAX_TIMEOUT_MS": "36000000",
    "BASH_DEFAULT_TIMEOUT_MS": "3600000",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0",
}

# Claude Code's default Bash timeout in the upstream monitor (BASH_DEFAULT_TIMEOUT_MS).
REACT_BASH_TIMEOUT_S = 3600


def monitor_agent(
    name: str,
    effort: str,
    model: Model,
    claude_code_version: str = "latest",
    small_model: str | None = CLAUDE_CODE_SMALL_MODEL,
) -> Agent:
    """Build the monitor agent, `claude_code` or `react`; effort `default` sets none.

    `small_model` takes Claude Code's own background calls; `None` sends them to `model`.
    """
    reasoning_effort = None if effort == "default" else effort
    if name == "claude_code":
        # Imported here: inspect_swe is only needed for this agent.
        from inspect_swe import claude_code

        # The served model defaults to the task's model, which is `model`.
        return claude_code(
            cwd=SANDBOX_TASK_DIR,
            effort=reasoning_effort,  # type: ignore[arg-type]
            version=claude_code_version,
            env=CLAUDE_CODE_ENV,
            haiku_model=small_model,
        )
    if name == "react":
        if reasoning_effort is not None:
            model = copy(model)
            model.config = model.config.merge(
                GenerateConfig(reasoning_effort=reasoning_effort)  # type: ignore[arg-type]
            )
        return react(
            prompt=None,
            tools=[bash(timeout=REACT_BASH_TIMEOUT_S), text_editor()],
            model=model,
            submit=False,
        )
    raise ValueError(f"monitor_agent must be one of {MONITOR_AGENTS}, got {name!r}")
