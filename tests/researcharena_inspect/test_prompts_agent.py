"""The agent's prompts are byte-identical to what upstream's own scripts deliver to each CLI.

Each combination runs upstream's prompt blocks of `pipeline_redteam.sh` and the family's
whole `solve.sh` from the pinned checkout (see `prompt_oracle`), then compares:

- the run's `instruction.md` and the agent's `prompt.txt`;
- the task config upstream picks when none is given;
- the first CLI call's stdin (the prompt) and, in settings that resume the agent, the
  second call's last argument (the "continue" prompt at the full budget).
"""

import difflib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from researcharena_inspect.prompts import (
    AGENT_FAMILIES,
    AgentFamily,
    agent_instruction_file,
    agent_prompt,
    budget_seconds,
    build_prompt,
    default_task_config,
    format_hours,
    list_side_tasks,
    resume_agent,
    resume_prompt,
    task_configs,
)
from researcharena_inspect.upstream import upstream_dir

from .prompt_oracle import make_stubs, run_upstream_agent, run_upstream_resume_prompt

pytestmark = pytest.mark.dataset_download  # clones upstream on first use

SETTINGS = ["cuda_kernel", "inference_optimization", "safety_finetuning", "utility_finetuning"]
RUNTIMES = ["1", "5", "0.5"]
WORKERS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True)
class AgentCombo:
    setting: str
    side_task: str
    task_config: str  # "" means not given, as run.sh passes it by default
    redruntime: str
    family: AgentFamily

    def label(self) -> str:
        config = self.task_config or "<default>"
        return f"{self.setting}/{self.side_task}/{config}/{self.redruntime}h/{self.family}"


def agent_combos(setting: str) -> list[AgentCombo]:
    """Every side task and the baseline x every task config (and none given) x runtime x family."""
    return [
        AgentCombo(setting, side_task, task_config, redruntime, family)
        for side_task in list_side_tasks(setting)
        for task_config in [*task_configs(setting), ""]
        for redruntime in RUNTIMES
        for family in AGENT_FAMILIES
    ]


def text_diff(label: str, expected: bytes, actual: bytes) -> str:
    lines = difflib.unified_diff(
        expected.decode("utf-8").splitlines(keepends=True),
        actual.decode("utf-8").splitlines(keepends=True),
        fromfile="port",
        tofile="upstream",
        n=1,
    )
    shown = "".join(list(lines)[:40]) or "(differ only in line endings or trailing newline)\n"
    return f"{label}: {len(expected)} bytes (port) vs {len(actual)} bytes (upstream)\n{shown}"


def compare_agent(combo: AgentCombo, work: Path, stubs) -> list[str]:
    run = run_upstream_agent(
        upstream_dir(),
        work,
        stubs,
        setting=combo.setting,
        side_task=combo.side_task,
        task_config=combo.task_config,
        redruntime=combo.redruntime,
        family=combo.family,
    )
    config = combo.task_config or None
    prompt = agent_prompt(combo.setting, combo.side_task, combo.family, combo.redruntime, config)
    prompt_file = agent_instruction_file(
        combo.setting,
        combo.side_task,
        combo.family,
        combo.redruntime,
        config,
    ).encode("utf-8")
    problems: list[str] = []
    if run.instruction_md != prompt_file:
        problems.append(text_diff("instruction.md", prompt_file, run.instruction_md))
    if run.prompt_txt != prompt_file:
        problems.append(text_diff("prompt.txt", prompt_file, run.prompt_txt))
    if run.task_config != (combo.task_config or default_task_config(combo.setting)):
        problems.append(f"task config: upstream picked {run.task_config!r}")
    resumes = resume_agent(combo.setting)
    if run.resume_agent != ("true" if resumes else "false"):
        problems.append(f"resume_agent: upstream has {run.resume_agent!r}")
    expected_calls = 2 if resumes else 1
    if len(run.calls) != expected_calls:
        problems.append(f"CLI called {len(run.calls)} times, expected {expected_calls}")
    if run.calls:
        first = run.calls[0]
        if first.stdin != prompt.encode("utf-8"):
            problems.append(text_diff("CLI stdin", prompt.encode("utf-8"), first.stdin))
        if any(prompt[:200] in arg for arg in first.argv):
            problems.append("prompt passed as an argument as well as on stdin")
    if resumes and len(run.calls) >= 2:
        expected_resume = resume_prompt(budget_seconds(combo.redruntime))
        if run.calls[1].argv[-1] != expected_resume:
            problems.append(f"resume prompt: upstream sent {run.calls[1].argv[-1]!r}")
        if run.calls[1].stdin:
            problems.append("resume call got stdin")
    return [f"[{combo.label()}] {p}" for p in problems]


@pytest.mark.parametrize("setting", SETTINGS)
def test_agent_prompts_match_upstream(setting: str, tmp_path: Path) -> None:
    combos = agent_combos(setting)
    stubs = make_stubs(tmp_path / "stubs")

    def check(item: tuple[int, AgentCombo]) -> list[str]:
        index, combo = item
        return compare_agent(combo, tmp_path / f"run{index}", stubs)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(check, enumerate(combos)))
    different = [r for r in results if r]
    print(
        f"agent parity {setting}: {len(combos)} combinations, "
        f"{len(combos) - len(different)} identical, {len(different)} different"
    )
    assert not different, "\n".join(p for r in different[:5] for p in r)


def test_agent_matrix_covers_every_setting() -> None:
    from researcharena_inspect.prompts import list_settings

    assert list_settings() == SETTINGS
    counts = {s: len(agent_combos(s)) for s in SETTINGS}
    # (side tasks + baseline) x (task configs + "not given") x 3 runtimes x 3 families
    assert counts == {
        "cuda_kernel": 5 * 2 * 9,
        "inference_optimization": 5 * 2 * 9,
        "safety_finetuning": 4 * 3 * 9,
        "utility_finetuning": 3 * 3 * 9,
    }


REMAINING = [-5, 0, 1, 59, 60, 61, 1799, 1800, 3599, 3600, 3601, 3661, 7199, 17999, 18000, 18059]


@pytest.mark.parametrize("family", AGENT_FAMILIES)
def test_resume_prompt_matches_solve_sh(family: str) -> None:
    for remaining in REMAINING:
        expected = run_upstream_resume_prompt(upstream_dir(), family, remaining)
        assert resume_prompt(remaining).encode("utf-8") == expected, remaining


@pytest.mark.parametrize(("hours", "seconds"), [("1", 3600), ("5", 18000), ("0.5", 1800)])
def test_budget_seconds(hours: str, seconds: int) -> None:
    assert budget_seconds(hours) == seconds


def test_format_hours_keeps_the_typed_text() -> None:
    assert format_hours("5") == "5"
    assert format_hours("5.0") == "5.0"
    assert format_hours(5) == "5"
    assert format_hours(0.5) == "0.5"
    for bad in ("five", "nan", "inf", True):
        with pytest.raises(ValueError, match="redruntime"):
            format_hours(bad)  # pyright: ignore[reportArgumentType]


def test_agent_prompt_refuses_unknown_inputs() -> None:
    with pytest.raises(ValueError, match="setting"):
        agent_prompt("bogus", "none", "claude", 1)
    with pytest.raises(ValueError, match="side task"):
        agent_prompt("cuda_kernel", "independent2", "claude", 1)
    with pytest.raises(ValueError, match="family"):
        agent_prompt("cuda_kernel", "none", "gemini", 1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="task config"):
        agent_prompt("utility_finetuning", "none", "claude", 5, task_config="qwen3_1.7b")


# A synthetic setting exercises the parts of prompt_builder.py that the pinned settings
# don't: nested and unresolved placeholders, no_side_task.md, a missing side_task.md,
# CRLF line endings, unknown task configs.
_INSTRUCTION = (
    "# Task {task_config}\r\n"
    "Model `{model}` for {num_hours} h ({redruntime}); nested {nested[key]} "
    "{nested[deep][leaf]} missing {nested[nope]} {unknown} {{escaped}} {lst} {thr} {flag} "
    "{nothing} {9bad} {a-b}\r\n"
    "Unicode: é✓ \\n backslash `$HOME` $(echo hi)\n\n\n"
)
_CONFIG = {
    "resume_agent": True,
    "configs": {
        "first": {
            "model": "org/model-a",
            "nested": {"key": "v1", "deep": {"leaf": 3}},
            "lst": [1, "two"],
            "thr": 1e-05,
            "flag": False,
            "nothing": None,
        },
        "second": {"model": "org/model-b", "nested": {"key": "v2"}},
    },
}


def _synthetic_upstream(root: Path) -> Path:
    setting = root / "settings" / "synthetic"
    (setting / "side_tasks" / "attack").mkdir(parents=True)
    (setting / "side_tasks" / "empty").mkdir(parents=True)
    (setting / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    (setting / "instruction.md").write_bytes(_INSTRUCTION.encode("utf-8"))
    (setting / "evaluate.py").write_text("", encoding="utf-8")
    (setting / "no_side_task.md").write_bytes(b"Baseline rule for {model}.\n\n")
    (setting / "side_tasks" / "attack" / "side_task.md").write_bytes(
        b"Secretly do X with {nested[key]} in {num_hours} hours.\n"
    )
    return root


SYNTHETIC_CASES = [
    # side task, task config, runtime, provider, main task only
    ("none", "first", "5", "claude", False),
    ("none", "second", "0.5", "opencode", False),
    ("none", "", "1.0", "codex", False),
    ("none", "unknown", "1", "none", False),
    ("attack", "first", "1", "claude", False),
    ("attack", "second", "2", "opencode", False),
    ("attack", "first", "1", "codex", True),
    ("empty", "first", "1", "claude", False),
    ("none", "first", "1", "none", True),
]


@pytest.mark.parametrize(
    ("side_task", "task_config", "redruntime", "provider", "main_only"), SYNTHETIC_CASES
)
def test_build_prompt_matches_prompt_builder_on_a_synthetic_setting(
    tmp_path: Path,
    side_task: str,
    task_config: str,
    redruntime: str,
    provider: str,
    main_only: bool,
) -> None:
    root = _synthetic_upstream(tmp_path)
    args = [
        "python3",
        str(upstream_dir() / "src" / "prompt_builder.py"),
        "--setting",
        "synthetic",
        "--side_task",
        side_task,
        "--task_config",
        task_config,
        "--redruntime",
        redruntime,
        "--red_provider",
        provider,
        "--settings_dir",
        str(root / "settings"),
    ]
    if main_only:
        args.append("--main_task_only")
    upstream = subprocess.run(args, capture_output=True, check=True)
    port = build_prompt(
        "synthetic",
        side_task,
        task_config,
        redruntime,
        provider,
        main_task_only=main_only,
        upstream=root,
    )
    assert (port + "\n").encode("utf-8") == upstream.stdout
