"""Transcripts: upstream's parsers on CLI output, and the renderer for Inspect messages."""

import json
from pathlib import Path

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall, ToolCallError

from researcharena_inspect.published import fetch_run_file
from researcharena_inspect.transcripts import (
    agent_family,
    messages_to_stream_events,
    parse_cli_output,
    render_messages,
)

# Runs whose published transcripts are byte-identical to the port's.
PARITY_RUNS = [
    (
        "cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740",
        "claude",
    ),
    ("cuda_kernel/cuda_kernel__baseline__codex__gpt-5.4__20260726_221542_c17417074", "codex"),
    (
        "inference_optimization/inference_optimization__baseline__opencode__opencode_gemini-3.1-pro__20260604_013703_c17311118",
        "opencode",
    ),
]


@pytest.mark.parametrize(
    ("provider", "family"),
    [
        ("claude", "claude"),
        ("claude_norepeat", "claude"),
        ("codex_norepeat_high", "codex"),
        ("codex_high", "codex"),
        ("opencode_norepeat", "opencode"),
    ],
)
def test_agent_family(provider: str, family: str) -> None:
    assert agent_family(provider) == family


def test_unknown_agent_family() -> None:
    with pytest.raises(ValueError, match="unknown agent family"):
        agent_family("gemini")


def write_jsonl(path: Path, events: list[dict], compact: bool = False) -> Path:
    separators = (",", ":") if compact else None
    path.write_text(
        "".join(json.dumps(e, separators=separators) + "\n" for e in events), encoding="utf-8"
    )
    return path


CLAUDE_EVENTS = [
    {
        "type": "system",
        "subtype": "init",
        "session_id": "s1",
        "model": "m",
        "cwd": "/home/user/task",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "secret plan"}],
        },
    },
    {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "I will look."}]},
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt"}],
        },
    },
]


@pytest.mark.dataset_download  # runs upstream's parser from the pinned checkout
def test_parse_cli_output_claude(tmp_path: Path) -> None:
    raw = write_jsonl(tmp_path / "solve_out.json", CLAUDE_EVENTS)
    out = parse_cli_output(raw, "claude_norepeat")
    assert out.with_cot.startswith(
        "Session start — s1\n  Model: m\n  Working dir: /home/user/task\n"
    )
    assert "  [thinking block]\n    secret plan\n" in out.with_cot
    assert "secret plan" not in out.without_cot
    # D-040: the agent's own prose survives the no-CoT pass.
    assert "  I will look.\n" in out.without_cot
    assert "  Tool call — Bash (t1)\n    $ ls\n" in out.without_cot
    assert "  Tool result — Bash (t1)\n    a.txt\n" in out.without_cot
    assert out.for_access(with_cot=False) == out.without_cot


@pytest.mark.dataset_download
def test_parse_cli_output_falls_back_like_the_pipeline(tmp_path: Path) -> None:
    # A message whose content is a string makes upstream's Claude parser crash on both passes:
    # the CoT transcript becomes the raw output, and the no-CoT one a copy of it (D-018).
    raw = write_jsonl(
        tmp_path / "solve_out.json", [{"type": "assistant", "message": {"content": "x"}}]
    )
    out = parse_cli_output(raw, "claude")
    assert out.with_cot == raw.read_text(encoding="utf-8")
    assert out.without_cot == out.with_cot


@pytest.mark.dataset_download
def test_parse_cli_output_codex_and_opencode(tmp_path: Path) -> None:
    codex_events = [
        {"type": "thread.started", "thread_id": "th"},
        {"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "think"}},
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "hello"}},
    ]
    # Upstream's Codex parser recognises its format only by the compact prefix
    # '{"type":"thread.started"'; anything else is copied verbatim, reasoning included.
    spaced = write_jsonl(tmp_path / "spaced.json", codex_events)
    assert parse_cli_output(spaced, "codex").without_cot == spaced.read_text(encoding="utf-8")
    codex = write_jsonl(tmp_path / "codex.json", codex_events, compact=True)
    out = parse_cli_output(codex, "codex")
    assert "=== Event 1 | type: thread.started ===" in out.with_cot
    assert '"text": "think"' in out.with_cot
    assert "think" not in out.without_cot
    assert '"text": "hello"' in out.without_cot

    opencode = write_jsonl(
        tmp_path / "opencode.json",
        [
            {"type": "reasoning", "part": {"type": "reasoning", "text": "hmm"}},
            {"type": "text", "part": {"type": "text", "text": "answer"}},
            {
                "type": "tool_use",
                "part": {
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "x" * 2500,
                    },
                },
            },
        ],
    )
    out = parse_cli_output(opencode, "opencode")
    assert "Reasoning:\n    hmm" in out.with_cot
    assert "hmm" not in out.without_cot
    # D-051: OpenCode transcripts cut tool output at 2,000 characters.
    assert "x" * 2000 + "\n    ... [truncated]" in out.without_cot
    assert "x" * 2001 not in out.without_cot


MESSAGES = [
    ChatMessageSystem(content="system prompt"),
    ChatMessageUser(content="red-team prompt naming the side task"),
    ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="private reasoning"),
            ContentText(text="Let me inspect the kernels."),
        ],
        tool_calls=[
            ToolCall(id="c1", function="bash", arguments={"command": "ls kernels"}),
            ToolCall(
                id="c2", function="text_editor", arguments={"command": "view", "path": "a.cu"}
            ),
        ],
        model="claude-sonnet-4-6",
    ),
    ChatMessageTool(content="rmsnorm.cu", tool_call_id="c1", function="bash"),
    ChatMessageTool(
        content="",
        tool_call_id="c2",
        function="text_editor",
        error=ToolCallError(type="file_not_found", message="no such file"),
    ),
    ChatMessageUser(content="continue"),
    ChatMessageAssistant(
        content=[ContentReasoning(reasoning="", summary="summary only", redacted=True)],
        model="claude-sonnet-4-6",
    ),
    ChatMessageAssistant(
        content=[ContentReasoning(reasoning="opaque", redacted=True), ContentText(text="Done.")]
    ),
]


def test_messages_to_stream_events_follows_claude_code() -> None:
    events = messages_to_stream_events(MESSAGES, session_id="sid")
    assert events[0] == {
        "type": "system",
        "subtype": "init",
        "session_id": "sid",
        "model": "claude-sonnet-4-6",
    }
    blocks = [e["message"]["content"][0] for e in events[1:]]
    assert [b["type"] for b in blocks] == [
        "thinking",
        "text",
        "tool_use",
        "tool_use",
        "tool_result",
        "tool_result",
        "thinking",
        "redacted_thinking",
        "text",
    ]
    assert blocks[5]["content"] == "no such file"
    assert blocks[6]["thinking"] == "summary only"
    assert all("side task" not in json.dumps(e) for e in events)


@pytest.mark.dataset_download
def test_render_messages_with_upstreams_parser() -> None:
    out = render_messages(MESSAGES, session_id="sid", cwd="/home/user/task")
    assert out.with_cot.startswith(
        "Session start — sid\n  Model: claude-sonnet-4-6\n  Working dir: /home/user/task\n\n"
        "Assistant — turn 1\n  [thinking block]\n    private reasoning\n\n"
        "Assistant — turn 2\n  Let me inspect the kernels.\n\n"
        "Assistant — turn 3\n  Tool call — bash (c1)\n    $ ls kernels\n"
    )
    assert "  Tool result — text_editor (c2)\n    no such file\n" in out.with_cot
    assert "  [redacted_thinking block]\n" in out.with_cot
    assert "private reasoning" not in out.without_cot
    assert "summary only" not in out.without_cot
    assert "[redacted_thinking block]" not in out.without_cot
    assert "  Let me inspect the kernels.\n" in out.without_cot  # D-040
    assert "side task" not in out.with_cot
    assert "continue" not in out.with_cot


@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
@pytest.mark.parametrize(("run_id", "family"), PARITY_RUNS)
def test_parse_cli_output_reproduces_published_transcripts(run_id: str, family: str) -> None:
    raw = fetch_run_file(run_id, "red_team/solve_out_sanitized.json")
    out = parse_cli_output(raw, family)
    cot = fetch_run_file(run_id, "red_team/solve_parsed_sanitized.txt").read_text(encoding="utf-8")
    no_cot = fetch_run_file(run_id, "red_team/solve_parsed_no_cot_sanitized.txt").read_text(
        encoding="utf-8"
    )
    assert out.with_cot == cot
    assert out.without_cot == no_cot
