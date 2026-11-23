"""The comparison agent: Inspect's ReAct loop with a shell and a file editor.

Upstream has no ReAct agent. This one gets what Claude Code gets, as far as Inspect's tools
allow: the same prompts and loop, the same effort, `bash` and `text_editor` for Bash, Read,
Write and Edit (no web tools, no subagents), the agent's user, environment and workspace.
`bash`'s description adds what Claude Code's model is told: where commands start, absolute
paths for the editor, and the one-hour limit. Unlike Claude Code's Bash tool, each command
starts a new shell and takes no `timeout`. `submit=False` ends a round when the model answers without a tool
call, as `claude --print` exits.
"""

from copy import copy

from inspect_ai.agent import Agent, react
from inspect_ai.model import CompactionSummary, GenerateConfig, Model
from inspect_ai.tool import Tool, text_editor, tool, tool_with
from inspect_ai.util import sandbox

from researcharena_inspect.environment.layout import TASK_DIR

REACT_BASH_TIMEOUT_S = 3600
"""Claude Code's default Bash timeout in upstream's agent (`BASH_DEFAULT_TIMEOUT_MS`)."""


def react_bash_description(cwd: str) -> str:
    """Inspect's description of `bash`, then where commands start, editor paths and the limit."""
    return (
        "Use this function to execute bash commands. "
        f"Each command runs in a new shell that starts in {cwd}, the working directory, so "
        "relative paths start there. "
        f"Use absolute paths for workspace files in the file editor, such as {cwd}/file.py; "
        "it resolves relative paths from /. "
        "`command` is the only argument, and each command is stopped after at most "
        f"{REACT_BASH_TIMEOUT_S} seconds ({REACT_BASH_TIMEOUT_S // 60} minutes)."
    )


REACT_TOOLS = ("bash", "text_editor")


@tool(name="bash")
def agent_bash(
    timeout: int | None = None,
    user: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> Tool:
    """Inspect's `bash()` with the agent's user, environment and working directory.

    The same command line (`bash --login -c`), description and output as Inspect's tool;
    `red_team_react` gives it `react_bash_description(cwd)` instead.
    """

    async def execute(command: str) -> str:
        """Use this function to execute bash commands.

        Args:
          command: The bash command to execute.

        Returns:
          The output of the command.
        """
        result = await sandbox().exec(
            cmd=["bash", "--login", "-c", command],
            cwd=cwd,
            timeout=timeout,
            user=user,
            env=env or {},
        )
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


def red_team_react(
    model: Model,
    effort: str,
    compaction: bool = True,
    *,
    user: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str = TASK_DIR,
) -> Agent:
    """ReAct configured as the comparison red-team agent.

    Args:
        model: the task's model.
        effort: the resolved effort (`default` sets none), set as `reasoning_effort`.
        compaction: summarise the conversation at 90% of the context window, as Claude Code
            compacts its own.
        user: sandbox user the tools run as.
        env: the agent's environment for `bash` (`environment.agent_env`).
        cwd: where `bash` starts, which its description names.
    """
    if effort != "default":
        model = copy(model)
        model.config = model.config.merge(
            GenerateConfig(reasoning_effort=effort)  # type: ignore[arg-type]
        )
    return react(
        prompt=None,
        tools=[
            tool_with(
                agent_bash(timeout=REACT_BASH_TIMEOUT_S, user=user, env=env, cwd=cwd),
                description=react_bash_description(cwd),
            ),
            text_editor(user=user),
        ],
        model=model,
        submit=False,
        compaction=CompactionSummary(threshold=0.9, memory=False) if compaction else None,
    )
