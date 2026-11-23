"""Monitor transcripts for agents that leave no CLI trace, such as Inspect's ReAct agent.

Inspect messages become Claude Code `stream-json` events, which upstream's Claude parser
(`agents/claude/human_readable_trace.py`) renders; only the conversion is new.

- One event per content block and per tool call, as Claude Code streams them.
- User and system messages are left out, as Claude Code's stream does not echo prompts. This
  also keeps the red-team prompt, which names the side task, from the monitor.
- Reasoning becomes a `thinking` block (the summary if redacted), or an empty
  `redacted_thinking` block; the no-CoT pass drops both and keeps visible text (D-040).

Not reproduced: Claude Code's own tool names, its closing `Result` event, and the other
families' formats (D-051). Reasoning returned only encrypted gives empty thinking blocks.
"""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentReasoning,
    ContentText,
)

from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.transcripts.cli import Transcripts, run_trace_parser, trace_parser


def _reasoning_block(content: ContentReasoning) -> dict[str, Any]:
    if content.redacted and not content.summary:
        return {"type": "redacted_thinking"}
    text = content.summary if content.redacted else content.reasoning
    return {"type": "thinking", "thinking": text or ""}


def _content_block(content: Any) -> dict[str, Any]:
    if isinstance(content, ContentReasoning):
        return _reasoning_block(content)
    if isinstance(content, ContentText):
        return {"type": "text", "text": content.text}
    # Images, documents, server-side tool use: upstream's parser prints them as raw JSON.
    return content.model_dump(exclude_none=True)


def _assistant_event(block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "assistant", "message": {"role": "assistant", "content": [block]}}


def messages_to_stream_events(
    messages: Sequence[ChatMessage],
    *,
    session_id: str | None = None,
    model: str | None = None,
    tools: Sequence[str] | None = None,
    cwd: str | None = None,
) -> list[dict[str, Any]]:
    """Convert Inspect messages to the Claude Code `stream-json` events upstream's parser reads."""
    if model is None:
        model = next(
            (m.model for m in messages if isinstance(m, ChatMessageAssistant) and m.model), None
        )
    init: dict[str, Any] = {"type": "system", "subtype": "init"}
    if session_id is not None:
        init["session_id"] = session_id
    if model is not None:
        init["model"] = model
    if tools:
        init["tools"] = list(tools)
    if cwd is not None:
        init["cwd"] = cwd
    events: list[dict[str, Any]] = [init]

    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            content = (
                [ContentText(text=message.content)]
                if isinstance(message.content, str)
                else message.content
            )
            for item in content:
                if isinstance(item, ContentText) and not item.text:
                    continue
                events.append(_assistant_event(_content_block(item)))
            for call in message.tool_calls or []:
                events.append(
                    _assistant_event(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function,
                            "input": call.arguments,
                        }
                    )
                )
        elif isinstance(message, ChatMessageTool):
            text = message.text
            if not text and message.error is not None:
                text = message.error.message
            block = {"type": "tool_result", "tool_use_id": message.tool_call_id, "content": text}
            events.append({"type": "user", "message": {"role": "user", "content": [block]}})
    return events


def render_messages(
    messages: Sequence[ChatMessage],
    *,
    session_id: str | None = None,
    model: str | None = None,
    tools: Sequence[str] | None = None,
    cwd: str | None = None,
    commit: str = UPSTREAM_COMMIT,
) -> Transcripts:
    """Render an Inspect sample's messages as the two monitor transcripts, in upstream's format.

    Args:
        messages: the agent's messages (for example `TaskState.messages`).
        session_id: shown in the transcript's "Session start" line, if given.
        model: shown as "Model:"; defaults to the model of the first assistant message.
        tools: tool names shown as "Tools:", if given.
        cwd: shown as "Working dir:", if given.
        commit: the upstream commit whose Claude parser renders the events.
    """
    events = messages_to_stream_events(
        messages, session_id=session_id, model=model, tools=tools, cwd=cwd
    )
    parser = trace_parser("claude", commit)
    with tempfile.TemporaryDirectory(prefix="researcharena_trace_") as tmp:
        raw = Path(tmp) / "solve_out.json"
        raw.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
            encoding="utf-8",
        )
        texts = []
        for no_thinking, name in ((False, "solve_parsed.txt"), (True, "solve_parsed_no_cot.txt")):
            out = Path(tmp) / name
            if not run_trace_parser(parser, raw, out, no_thinking=no_thinking):
                raise RuntimeError(f"upstream's Claude trace parser failed on {raw}")
            texts.append(out.read_text(encoding="utf-8"))
    return Transcripts(with_cot=texts[0], without_cot=texts[1])
