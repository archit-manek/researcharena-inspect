"""The monitor prompt, built by upstream's own scripts from the pinned checkout.

No published monitor folder keeps `monitor_prompt.md`, but one published monitor printed its own
`prompt.txt` with `cat`, so its trace holds the exact prompt it was given.
"""

import json
from pathlib import Path

import pytest

from researcharena_inspect.monitor import ACCESS_TYPES, published_bundle, upstream_monitor_prompt
from researcharena_inspect.pins import PAPER_COMMIT
from researcharena_inspect.prompts import (
    MonitorAccess,
    monitor_prompt_file,
    monitor_task_instruction,
)
from researcharena_inspect.published import fetch_run_file, load_index

# The Claude Sonnet 4.6 monitor of this run ran `cat /home/user/prompt.txt`.
ECHO_RUN = (
    "safety_finetuning/safety_finetuning__embedded_german__claude__claude-opus-4-6__20260504_205338"
)
ECHO_MONITOR = "claude_claude-sonnet-4-6_high"


def printed_prompt(trace: str) -> str:
    """The output of the monitor's `cat .../prompt.txt` call in Claude Code's stream-json."""
    calls: set[str] = set()
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (event.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use" and "prompt.txt" in json.dumps(block.get("input")):
                calls.add(block["id"])
            if block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                return block["content"]
    raise AssertionError("the monitor never printed prompt.txt")


@pytest.mark.huggingface
@pytest.mark.dataset_download
@pytest.mark.parametrize("commit", ["latest", "paper"])
def test_prompt_equals_what_a_published_monitor_was_given(commit: str) -> None:
    run = next(r for r in load_index()["runs"] if r["id"] == ECHO_RUN)
    bundle = published_bundle(run)
    trace = fetch_run_file(ECHO_RUN, f"blue_team/{ECHO_MONITOR}/monitor_out_sanitized.json")
    printed = printed_prompt(trace.read_text(encoding="utf-8"))
    kwargs = {} if commit == "latest" else {"commit": PAPER_COMMIT}
    built = upstream_monitor_prompt(
        setting=bundle.setting,
        side_task=bundle.side_task,
        task_config=bundle.task_config,
        transcript_path=bundle.transcript(with_cot=False),
        workspace=None,
        with_artifact=False,
        with_cot=False,
        **kwargs,
    )
    # Claude Code's Bash tool drops the file's final newline from what it shows the model.
    assert built.prompt == printed + "\n"


@pytest.mark.dataset_download
def test_instruction_keeps_upstreams_quirks(tmp_path: Path) -> None:
    transcript = tmp_path / "agent_transcript.txt"
    transcript.write_text("Assistant — turn 1\n")
    built = upstream_monitor_prompt(
        setting="safety_finetuning",
        side_task="embedded_german",
        task_config="",
        transcript_path=transcript,
        workspace=None,
        with_artifact=False,
        with_cot=False,
    )
    # D-005: the pipeline builds the instruction without --redruntime, so the default 1 hour.
    assert "You have 1 hours to complete the following task." in built.main_task_instruction
    # The templates are filled with str.replace, so an escaped brace stays doubled.
    assert "{{" in built.prompt
    # The transcript is staged as a file, not pasted into the prompt.
    assert "Assistant — turn 1" not in built.prompt


@pytest.mark.dataset_download
@pytest.mark.parametrize("access", list(ACCESS_TYPES))
@pytest.mark.parametrize(
    ("setting", "side_task", "task_config"),
    [("cuda_kernel", "rope_backdoor", "tinyllama"), ("safety_finetuning", "none", "")],
)
def test_upstream_scripts_agree_with_the_python_port(
    tmp_path: Path, access: str, setting: str, side_task: str, task_config: str
) -> None:
    """`researcharena_inspect.prompts` (the port) gives the same bytes as upstream's scripts."""
    with_artifact, with_cot = ACCESS_TYPES[access]
    snapshot = tmp_path / "red_team" / "task_snapshot"
    (snapshot / "kernels").mkdir(parents=True)
    (snapshot / "kernels" / "rmsnorm.cu").write_text("__global__ void k() {}\n")
    (snapshot / "notes.md").write_text("{setting} is not a placeholder here\n")
    transcript = tmp_path / "agent_transcript.txt"
    transcript.write_text("Assistant — turn 1\n")

    built = upstream_monitor_prompt(
        setting=setting,
        side_task=side_task,
        task_config=task_config,
        transcript_path=transcript,
        workspace=snapshot,
        with_artifact=with_artifact,
        with_cot=with_cot,
    )
    assert built.main_task_instruction == monitor_task_instruction(setting, task_config)
    assert built.prompt == monitor_prompt_file(
        MonitorAccess(with_artifact=with_artifact, with_cot=with_cot),
        setting=setting,
        side_task=side_task,
        task_config=task_config,
        transcript=transcript.read_text(encoding="utf-8"),
        task_snapshot=snapshot,
    )
