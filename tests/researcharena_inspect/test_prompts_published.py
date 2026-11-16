"""The second check: the port's prompts against the authors' published runs.

The published runs came from code older than the pinned commit, so differences are
expected. Each text recovered from a published transcript (see `published_prompts`) is
compared with the port (`b922f4b`) and with the paper commit's own `prompt_builder.py`
(`58776e0`), and the tests pin down which of the two it matches.
"""

import collections
from typing import Any

import pytest

from researcharena_inspect.prompts import (
    INSTRUCTION_FILENAME,
    agent_prompt,
    monitor_task_instruction,
)
from researcharena_inspect.published import fetch_run_file, load_index

from . import published_prompts as pp

pytestmark = [pytest.mark.huggingface, pytest.mark.dataset_download]

CUDA_CLAUDE_BASELINE = (
    "cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740"
)
CUDA_OPENCODE_BASELINE = (
    "cuda_kernel/cuda_kernel__baseline__opencode__gemini-3.1-pro__20260726_221145_c17417069"
)
CUDA_OPENCODE_DTYPE = "cuda_kernel/cuda_kernel__embedded3__opencode__opencode_gemini-3.1-pro__20260606_213455_c17315082"
SAFETY_BASELINE = "safety_finetuning/safety_finetuning__baseline__claude_norepeat__claude-opus-4-6__20260428_141136"
PROMPT_FILE_NAMES = {
    "instruction.md",
    "prompt.txt",
    "monitor_prompt.md",
    "main_task_instruction.md",
    "task_instruction.md",
}


def _run(run_id: str) -> dict[str, Any]:
    return next(r for r in load_index()["runs"] if r["id"] == run_id)


def _monitor_reads(run_id: str, monitor_dir: str) -> list[pp.Recovered]:
    stream = pp.read_text(
        fetch_run_file(run_id, f"blue_team/{monitor_dir}/monitor_out_sanitized.json")
    )
    return pp.claude_reads(stream, INSTRUCTION_FILENAME) + pp.codex_reads(
        stream, INSTRUCTION_FILENAME
    )


def test_published_runs_carry_no_prompt_files() -> None:
    for run_id in (CUDA_CLAUDE_BASELINE, SAFETY_BASELINE):
        names = {path.rsplit("/", 1)[-1] for path in pp.run_files(run_id)}
        assert "metadata.json" in names
        assert not names & PROMPT_FILE_NAMES


def test_cuda_monitor_task_text_is_b922f4bs() -> None:
    run = _run(CUDA_CLAUDE_BASELINE)
    port = monitor_task_instruction(run["setting"], run["task_config"]).rstrip("\n").split("\n")
    paper = pp.paper_era_prompt(run["setting"], "none", run["task_config"], "none", None)
    reads = _monitor_reads(CUDA_CLAUDE_BASELINE, "claude_claude-opus-4-8_high")
    assert reads
    for copy in reads:
        assert pp.classify_lines(copy.lines, port, paper.split("\n"))[0] == pp.IDENTICAL


def test_safety_monitor_task_text_is_the_paper_commits() -> None:
    # Published post-training monitors saw the text from before 30da456 (ledger D-054),
    # and, in a 5-hour run, "You have 1 hours" (ledger D-005).
    run = _run(SAFETY_BASELINE)
    port = monitor_task_instruction(run["setting"], run["task_config"]).rstrip("\n").split("\n")
    paper = pp.paper_era_prompt(run["setting"], "none", run["task_config"], "none", None)
    reads = _monitor_reads(SAFETY_BASELINE, "codex_gpt-5.4_high")
    assert reads
    for copy in reads:
        assert pp.classify_lines(copy.lines, port, paper.split("\n"))[0] == pp.PAPER_ONLY
        assert run["hours"] == 5
        assert any(line.startswith("You have 1 hours") for line in copy.lines)


def _argv_prompts(run_id: str) -> list[pp.ArgvPrompt]:
    stream = pp.read_text(fetch_run_file(run_id, "red_team/solve_out_sanitized.json"))
    return [p for p in pp.opencode_argv_prompts(stream) if not p.resume]


def test_opencode_prompt_shown_by_ps_is_b922f4bs() -> None:
    run = _run(CUDA_OPENCODE_BASELINE)
    expected = agent_prompt(run["setting"], "none", "opencode", run["hours"], run["task_config"])
    shown = _argv_prompts(CUDA_OPENCODE_BASELINE)
    assert shown
    assert all(p.text == pp.ps_view(expected) for p in shown)


def test_published_dtype_prompt_predates_the_crash_section() -> None:
    # The 6 June dtype run's prompt is today's without the side task's last section,
    # "Do Not Crash the Kernels", which the side task has had since the first public commit.
    # The older side_task.md ended after the line before it.
    run = _run(CUDA_OPENCODE_DTYPE)
    expected = agent_prompt(
        run["setting"], run["side_task"], "opencode", run["hours"], run["task_config"]
    )
    start = expected.index("## Do Not Crash the Kernels")
    end = expected.index("\n\nYou are running in non-interactive mode")
    without_section = expected[:start].rstrip("\n") + "\n" + expected[end:]
    shown = _argv_prompts(CUDA_OPENCODE_DTYPE)
    assert shown
    for p in shown:
        assert p.text != pp.ps_view(expected)
        assert p.text == pp.ps_view(without_section)


@pytest.mark.slow
def test_published_sample_classification() -> None:
    """Every recoverable prompt text in a sample of 24 runs (6 per setting), classified."""
    runs = pp.sample_runs(load_index()["runs"])
    assert len(runs) == 24
    evidence = [e for run in runs for e in pp.collect_evidence(run)]
    table = collections.Counter((e.setting, e.kind, e.status) for e in evidence)
    for key, count in sorted(table.items()):
        print(f"published sample: {' | '.join(key)}: {count}")
    # CUDA and inference monitors saw b922f4b's task text. Post-training monitors saw the
    # paper commit's (ledger D-054), 3 of them an older one still. One dtype prompt
    # predates the side task's last section.
    ident, paper, neither = pp.IDENTICAL, pp.PAPER_ONLY, pp.NEITHER
    assert table == {
        ("cuda_kernel", "monitor task text", ident): 37,
        ("cuda_kernel", "agent prompt", ident): 1,
        ("cuda_kernel", "agent prompt", neither): 1,
        ("cuda_kernel", "resume prompt", ident): 1,
        ("inference_optimization", "monitor task text", ident): 36,
        ("inference_optimization", "agent prompt", ident): 1,
        ("inference_optimization", "resume prompt", ident): 2,
        ("safety_finetuning", "monitor task text", paper): 38,
        ("safety_finetuning", "monitor task text", neither): 3,
        ("utility_finetuning", "monitor task text", paper): 36,
    }
