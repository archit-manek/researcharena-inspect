"""The comparison agent: Inspect's ReAct loop with a shell and a file editor.

Upstream has no ReAct agent; this one is given what the Claude Code agent is given, as far as
Inspect's standard tools allow:

- the same first prompt (the `claude` family's, whose suffix names "Read, Write, Bash") and the
  same continuation prompts, through the same loop (`agents.continuation`);
- `bash` for Claude Code's Bash tool, with its default timeout (`BASH_DEFAULT_TIMEOUT_MS`,
  one hour), and `text_editor` for Read, Write and Edit; no web tools (they need a search
  provider) and no subagents;
- the same served-model effort, and no system prompt beyond the task prompt;
- `submit=False`: a round ends when the model answers without calling a tool, which is when
  `claude --print` exits;
- summary compaction near the context limit, as Claude Code compacts its own conversation.
"""

from copy import copy

from inspect_ai.agent import Agent, react
from inspect_ai.model import CompactionSummary, GenerateConfig, Model
from inspect_ai.tool import bash, text_editor

REACT_BASH_TIMEOUT_S = 3600
"""Claude Code's default Bash timeout in upstream's agent (`BASH_DEFAULT_TIMEOUT_MS`)."""

REACT_TOOLS = ("bash", "text_editor")


def red_team_react(model: Model, effort: str, compaction: bool = True) -> Agent:
    """ReAct configured as the comparison red-team agent.

    Args:
        model: the task's model.
        effort: the resolved effort (`default` sets none), set as `reasoning_effort`.
        compaction: summarise the conversation at 90% of the context window.
    """
    if effort != "default":
        model = copy(model)
        model.config = model.config.merge(
            GenerateConfig(reasoning_effort=effort)  # type: ignore[arg-type]
        )
    return react(
        prompt=None,
        tools=[bash(timeout=REACT_BASH_TIMEOUT_S), text_editor()],
        model=model,
        submit=False,
        compaction=CompactionSummary(threshold=0.9, memory=False) if compaction else None,
    )
