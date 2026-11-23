"""The comparison agent: Inspect's ReAct loop with a shell and a file editor.

Upstream has no ReAct agent; this one is given what the Claude Code agent is given, as far as
Inspect's standard tools allow:

- the same first prompt (the `claude` family's, whose suffix names "Read, Write, Bash") and the
  same continuation prompts, through the same loop (`agents.continuation`);
- `bash` for Claude Code's Bash tool, with its default timeout (`BASH_DEFAULT_TIMEOUT_MS`,
  one hour), and `text_editor` for Read, Write and Edit; no web tools (they need a search
  provider) and no subagents;
- both run as the agent's user; `bash` also gets the agent's environment and starts in the
  workspace, as Claude Code's shell does (Inspect's own `bash()` takes a user but no
  environment or working directory);
- three sentences in `bash`'s description (`react_bash_description`), a tool every model
  reads: where commands start, absolute paths for the editor (it resolves relative ones from
  `/`), and the per-command limit. Claude Code's model is told the starting directory, absolute
  paths for Read and Edit, and the 60-minute default; unlike this tool, its shell keeps its
  working directory between calls and its Bash tool takes an optional `timeout` up to the
  maximum upstream sets (`BASH_MAX_TIMEOUT_MS`);
- the same served-model effort, and no system prompt beyond the task prompt;
- `submit=False`: a round ends when the model answers without calling a tool, which is when
  `claude --print` exits;
- summary compaction near the context limit, as Claude Code compacts its own conversation.
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
        compaction: summarise the conversation at 90% of the context window.
        user: sandbox user the tools run as (the environment's agent user).
        env: the agent's environment for `bash` (`environment.agent_env`).
        cwd: where `bash` starts (the workspace), which its description names.
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
