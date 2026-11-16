"""A logged red-team sample becomes a run folder the monitor reads (no Docker, no model)."""

import json
from pathlib import Path

import pytest
from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store

from researcharena_inspect.agents import AgentRun, write_run_dir
from researcharena_inspect.monitor import bundle_to_sample, local_bundle, researcharena_monitor
from researcharena_inspect.prompts import resume_prompt
from researcharena_inspect.task import red_team_sample
from researcharena_inspect.transcripts import parse_cli_output

STDOUT = [
    {
        "type": "system",
        "subtype": "init",
        "cwd": "/home/user/task",
        "session_id": "0b4f",
        "tools": ["Bash", "Read", "Write"],
        "model": "claude-sonnet-4-6",
    },
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "Plan: read the kernels first."}],
        },
    },
    {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "Looking."}]},
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
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "kernels"}],
        },
    },
    {"type": "result", "subtype": "success", "result": "Done.", "num_turns": 2},
]


def _sample(agent: str, messages: list, run_fields: dict) -> EvalSample:
    template = red_team_sample("cuda_kernel", "independent", "1")
    assert template.metadata is not None
    assert isinstance(template.input, str)
    store = Store()
    run = AgentRun(store=store)
    run.agent = agent
    run.model = "claude-sonnet-4-6"
    run.effort = "xhigh"
    run.redruntime = "1"
    run.resume = True
    run.red_start = 1_800_000_000.25
    run.red_end = 1_800_003_655.9
    for key, value in run_fields.items():
        setattr(run, key, value)
    return EvalSample(
        id=str(template.id),
        epoch=1,
        input=template.input,
        target="",
        messages=[ChatMessageUser(content=template.input), *messages],
        metadata=template.metadata,
        store=dict(store.items()),
    )


def claude_sample() -> EvalSample:
    lines = [json.dumps(e, separators=(",", ":")) for e in STDOUT]
    return _sample(
        "claude_code",
        [],
        {
            "stdout": lines,
            "stderr": ["note\n"],
            "log": ["[agent] TIMEOUT_HOURS=1 total_timeout_seconds=3600"],
            "cli_version": "2.1.162 (Claude Code)",
            "cli_path": "/var/tmp/x/claude-2.1.162-linux-x64",
            "cli_requested": "2.1.162",
        },
    )


def react_sample() -> EvalSample:
    messages = [
        ChatMessageAssistant(
            content=[
                ContentReasoning(reasoning="Plan: read the kernels first."),
                ContentText(text="Looking."),
            ],
            tool_calls=[ToolCall(id="t1", function="bash", arguments={"command": "ls"})],
            model="claude-sonnet-4-6",
        ),
        ChatMessageTool(content="kernels", tool_call_id="t1", function="bash"),
        ChatMessageAssistant(content="Done.", model="claude-sonnet-4-6"),
        ChatMessageUser(content=resume_prompt(1200)),
        ChatMessageAssistant(content="Still done.", model="claude-sonnet-4-6"),
    ]
    return _sample("react", messages, {"log": ["[agent] TIMEOUT_HOURS=1"]})


def test_claude_code_run_folder(tmp_path: Path) -> None:
    sample = claude_sample()
    run_dir = write_run_dir(sample, tmp_path)
    assert run_dir.name == (
        "cuda_kernel__independent__claude__claude-sonnet-4-6__20270115_080000_e1"
    )
    red = run_dir / "red_team"
    raw = (red / "solve_out.json").read_text("utf-8")
    assert raw == "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in STDOUT)
    parsed = parse_cli_output(red / "solve_out.json", "claude")
    assert (red / "solve_parsed.txt").read_text("utf-8") == parsed.with_cot
    assert (red / "solve_parsed_no_cot.txt").read_text("utf-8") == parsed.without_cot
    assert "Plan: read the kernels first." in parsed.with_cot
    assert "Plan: read the kernels first." not in parsed.without_cot
    assert "Session start — 0b4f" in parsed.with_cot
    assert "$ ls" in parsed.without_cot
    assert "Result — success" in parsed.without_cot
    assert (red / "time_taken.txt").read_text("utf-8") == "3655\n"
    assert "version: 2.1.162 (Claude Code)\n" in (red / "cli_version.txt").read_text("utf-8")
    assert (red / "solve_err.txt").read_text("utf-8").startswith("[agent] TIMEOUT_HOURS=1")
    assert sample.metadata is not None
    assert (run_dir / "instruction.md").read_text("utf-8") == sample.metadata["instruction_file"]

    meta = json.loads((run_dir / "metadata.json").read_text("utf-8"))
    assert list(meta)[:13] == [
        "setting",
        "side_task",
        "redruntime_hours",
        "red_provider",
        "red_model",
        "red_effort_level",
        "resume_agent",
        "cluster_id",
        "experiment",
        "task_config",
        "seed",
        "baseline",
        "timestamp",
    ]
    assert meta["redruntime_hours"] == 1
    assert meta["red_provider"] == "claude"
    assert meta["task_config"] == "tinyllama"
    assert meta["baseline"] is False


def test_react_run_folder_hides_prompts_from_the_transcript(tmp_path: Path) -> None:
    sample = react_sample()
    run_dir = write_run_dir(sample, tmp_path)
    red = run_dir / "red_team"
    with_cot = (red / "solve_parsed.txt").read_text("utf-8")
    without_cot = (red / "solve_parsed_no_cot.txt").read_text("utf-8")
    assert "Plan: read the kernels first." in with_cot
    assert "Plan: read the kernels first." not in without_cot
    assert "Tool call — bash (t1)" in without_cot
    # Like Claude Code's stream, neither the task prompt (which names the side task) nor the
    # continuation prompts appear.
    for text in (with_cot, without_cot):
        assert "CUDA Kernel Optimization Task" not in text
        assert "Continue where you left off" not in text
    events = [json.loads(line) for line in (red / "solve_out.json").read_text("utf-8").splitlines()]
    assert events[0]["type"] == "system"
    assert not (red / "cli_version.txt").exists()
    assert json.loads((run_dir / "metadata.json").read_text("utf-8"))["red_provider"] == "react"


@pytest.mark.parametrize("make", [claude_sample, react_sample])
@pytest.mark.parametrize("access", ["transcript", "transcript_cot"])
def test_run_folder_is_a_monitor_bundle(make, access: str, tmp_path: Path) -> None:
    run_dir = write_run_dir(make(), tmp_path)
    bundle = local_bundle(run_dir)
    assert bundle.setting == "cuda_kernel"
    assert bundle.side_task == "independent"
    assert bundle.task_config == "tinyllama"
    assert bundle.workspace is None
    with_cot = access == "transcript_cot"
    assert bundle.transcript(with_cot) == run_dir / "red_team" / (
        "solve_parsed.txt" if with_cot else "solve_parsed_no_cot.txt"
    )
    monitor_sample = bundle_to_sample(bundle, access)
    assert monitor_sample.metadata is not None
    assert monitor_sample.metadata["transcript_path"] == str(bundle.transcript(with_cot))
    assert monitor_sample.metadata["main_task_instruction"].startswith(
        "# CUDA Kernel Optimization Task"
    )

    task = researcharena_monitor(access=access, local_runs=str(tmp_path))
    assert [s.id for s in task.dataset] == [run_dir.name]


def test_a_sample_without_a_run_is_refused(tmp_path: Path) -> None:
    template = red_team_sample("cuda_kernel", "none", "1")
    assert template.metadata is not None
    sample = EvalSample(
        id="x", epoch=1, input="p", target="", messages=[], metadata=template.metadata
    )
    with pytest.raises(ValueError, match="no red-team run"):
        write_run_dir(sample, tmp_path)
