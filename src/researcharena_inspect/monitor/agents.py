"""The monitor agents: Claude Code through Inspect SWE, or Inspect's ReAct agent.

Upstream runs the monitor CLI as (`monitors/claude/monitor.sh` at the pinned commit)

    claude --print --verbose --model "$AGENT_CONFIG" --effort "$EFFORT" \
        --output-format stream-json --dangerously-skip-permissions < prompt.txt

in `/home/user/task`, with every tool, `BASH_MAX_TIMEOUT_MS=36000000` and
`BASH_DEFAULT_TIMEOUT_MS=3600000`, after installing the latest Claude Code release. Inspect SWE's
`claude_code()` runs the same flags (`--print --output-format stream-json --verbose
--dangerously-skip-permissions --model`), passes the prompt as an argument instead of on stdin,
and routes the model calls through Inspect's bridge, so the effort is set on the served model
(`GenerateConfig.reasoning_effort`) rather than by `--effort`.

The ReAct monitor has no upstream counterpart. It gets a shell and a file editor, the
equivalent of Claude Code's Bash, Read, Write and Edit tools, no system prompt beyond the
monitor prompt, and stops when it stops calling tools, as `claude --print` does.
"""

from copy import copy

from inspect_ai.agent import Agent, react
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.tool import bash, text_editor

from researcharena_inspect.monitor.staging import SANDBOX_TASK_DIR
from researcharena_inspect.pins import CLAUDE_CODE_SMALL_MODEL

MONITOR_AGENTS = ("claude_code", "react")

# monitor.sh exports BASH_MAX_TIMEOUT_MS=36000000; pipeline_monitor.sh passes
# BASH_DEFAULT_TIMEOUT_MS=3600000 into the container.
#
# D-089: Inspect SWE sets three variables of its own for the CLI that upstream's monitor did
# not have. What the port does with each:
# - CLAUDE_CODE_DISABLE_AUTO_MEMORY=1: turned back on here ("0"), as for the red-team agent, so
#   the system prompt carries the auto-memory section in CLI versions that have one.
# - CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1: kept. With it cleared the CLI (2.1.162) offers
#   the model the same 25 tools through Inspect's bridge, and with tool search forced on as
#   well (ENABLE_TOOL_SEARCH=true) the bridge passes on only 9 of the CLI's 26 tools, because
#   it does not carry deferred tools. Measured with a mock model, 17 Nov 2026.
# - CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1: kept (telemetry, updates, feature flags).
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
    """Build the monitor agent.

    Args:
        name: `claude_code` or `react`.
        effort: reasoning effort for the monitor model; `default` leaves the model's own.
        model: the monitor model, the task's model.
        claude_code_version: Claude Code version for Inspect SWE (`latest`, `stable`, `auto`,
            `sandbox` or a version such as `2.1.162`). Upstream installs the latest release.
        small_model: model for Claude Code's own background calls: Haiku 4.5 as in the
            published runs; `None` sends them to the monitor model (Inspect SWE's default).
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
