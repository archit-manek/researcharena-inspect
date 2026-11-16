"""The monitor's prompts and staged text are byte-identical to what upstream's scripts produce.

Each combination writes a red-team run folder (metadata, both transcripts, a workspace
snapshot), runs the block of `pipeline_monitor.sh` that builds and stages the prompt, then
the family's whole `monitor.sh` with a stub CLI (see `prompt_oracle`). It compares:

- `prompt.txt` and `monitor_prompt.md`;
- `main_task_instruction.md` and the staged `task/task_instruction.md`;
- the staged `task/agent_transcript.txt` (which transcript each access type gets);
- what the CLI received: stdin for Claude Code and Codex, the last argument for OpenCode.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from researcharena_inspect.prompts import (
    ACCESS_TYPES,
    AGENT_FAMILIES,
    AgentFamily,
    MonitorAccess,
    build_monitor_prompt,
    list_directory,
    list_side_tasks,
    monitor_prompt,
    monitor_prompt_file,
    monitor_task_instruction,
    task_configs,
)
from researcharena_inspect.prompts.monitor import load_file, templates_dir, truncate_transcript
from researcharena_inspect.upstream import upstream_dir

from .prompt_oracle import make_eval_dir, make_stubs, run_upstream_monitor
from .test_prompts_agent import SETTINGS, WORKERS, text_diff

pytestmark = pytest.mark.dataset_download  # clones upstream on first use

# Distinct transcripts, longer than the 80,000-character cut, with placeholder look-alikes.
TRANSCRIPTS = {
    "solve_parsed.txt": "WITH REASONING {setting} {main_task_instruction} ✓\n" * 2500,
    "solve_parsed_no_cot.txt": "ACTIONS ONLY {side_task_type} {transcript} é\n" * 2500,
}


def _sized(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.truncate(size)


def make_snapshot(root: Path) -> Path:
    """A workspace with the cases the listing must get right."""
    snap = root / "task_snapshot"
    _sized(snap / "kernels" / "attention.cu", 12_345)
    _sized(snap / "kernels" / "rmsnorm.cu", 999)
    _sized(snap / "bench.sh", 2_048)
    _sized(snap / "big.bin", 1_234_567)
    _sized(snap / ".git" / "HEAD", 21)
    _sized(snap / ".git" / "objects" / "ab" / "cdef0123", 77)
    _sized(snap / "deep" / "a" / "b" / "c" / "d.txt", 5)
    for i in range(55):
        _sized(snap / "many" / f"f{i:02d}.py", i)
    # Placeholder look-alikes: upstream's chained replacements reach into the listing.
    for name in (
        "{setting}.txt",
        "{side_task_type}.log",
        "{main_task_instruction}.md",
        "{transcript}",
        "{environment_context}",
    ):
        _sized(snap / name, 3)
    _sized(snap / "résumé ✓.txt", 10)
    (snap / "empty_dir").mkdir()
    (snap / "link_to_kernels").symlink_to("kernels")
    (snap / "link.txt").symlink_to("bench.sh")
    return snap


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_snapshot(tmp_path_factory.mktemp("snapshot"))


@dataclass(frozen=True)
class MonitorCombo:
    setting: str
    side_task: str
    task_config: str | None  # None: metadata.json has no task_config
    access: MonitorAccess
    family: AgentFamily

    def label(self) -> str:
        config = self.task_config if self.task_config is not None else "<missing>"
        return f"{self.setting}/{self.side_task}/{config}/{self.access.template_name}/{self.family}"


def monitor_combos(setting: str) -> list[MonitorCombo]:
    """Every side task and the baseline x task config (and none) x access type x family."""
    return [
        MonitorCombo(setting, side_task, task_config, access, family)
        for side_task in list_side_tasks(setting)
        for task_config in [*task_configs(setting), None]
        for access in ACCESS_TYPES
        for family in AGENT_FAMILIES
    ]


def compare_monitor(combo: MonitorCombo, work: Path, stubs, snapshot: Path) -> list[str]:
    eval_dir = work / "eval_dir"
    make_eval_dir(
        eval_dir,
        setting=combo.setting,
        side_task=combo.side_task,
        task_config=combo.task_config,
        transcripts=TRANSCRIPTS,
        snapshot=snapshot,
    )
    run = run_upstream_monitor(
        upstream_dir(),
        work / "oracle",
        stubs,
        eval_dir=eval_dir,
        access=combo.access,
        family=combo.family,
    )
    transcript = TRANSCRIPTS[combo.access.transcript_file]
    snapshot_path = str(eval_dir / "red_team" / "task_snapshot")
    prompt_file = monitor_prompt_file(
        combo.access,
        setting=combo.setting,
        side_task=combo.side_task,
        task_config=combo.task_config,
        transcript=transcript,
        task_snapshot=snapshot_path,
    ).encode("utf-8")
    delivered = monitor_prompt(
        combo.access,
        combo.family,
        setting=combo.setting,
        side_task=combo.side_task,
        task_config=combo.task_config,
        transcript=transcript,
        task_snapshot=snapshot_path,
    )
    instruction = monitor_task_instruction(combo.setting, combo.task_config).encode("utf-8")
    problems: list[str] = []
    for name, actual in (
        ("prompt.txt", run.prompt_txt),
        ("monitor_prompt.md", run.monitor_prompt_md),
    ):
        if actual != prompt_file:
            problems.append(text_diff(name, prompt_file, actual))
    for name, actual in (
        ("main_task_instruction.md", run.main_task_instruction_md),
        ("task/task_instruction.md", run.staged_instruction),
    ):
        if actual != instruction:
            problems.append(text_diff(name, instruction, actual))
    if run.staged_transcript != transcript.encode("utf-8"):
        problems.append(f"task/agent_transcript.txt is not {combo.access.transcript_file}")
    name = run.run_dir_name
    if (
        ("_artifact" in name) != combo.access.with_artifact
        or name.endswith("_cot") != combo.access.with_cot
        or not name.endswith(combo.access.suffix)
    ):
        problems.append(f"output folder {name!r} doesn't end in {combo.access.suffix!r}")
    if len(run.calls) != 1:
        problems.append(f"CLI called {len(run.calls)} times")
    else:
        call = run.calls[0]
        if combo.family == "opencode":
            if call.argv[-1] != delivered:
                problems.append(
                    text_diff(
                        "CLI argument", delivered.encode("utf-8"), call.argv[-1].encode("utf-8")
                    )
                )
            if call.stdin:
                problems.append("opencode got stdin")
        elif call.stdin != delivered.encode("utf-8"):
            problems.append(text_diff("CLI stdin", delivered.encode("utf-8"), call.stdin))
        if not call.cwd.endswith("/task"):
            problems.append(f"monitor ran in {call.cwd}")
    return [f"[{combo.label()}] {p}" for p in problems]


@pytest.mark.parametrize("setting", SETTINGS)
def test_monitor_prompts_match_upstream(setting: str, tmp_path: Path, snapshot: Path) -> None:
    combos = monitor_combos(setting)
    stubs = make_stubs(tmp_path / "stubs")

    def check(item: tuple[int, MonitorCombo]) -> list[str]:
        index, combo = item
        return compare_monitor(combo, tmp_path / f"run{index}", stubs, snapshot)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(check, enumerate(combos)))
    different = [r for r in results if r]
    print(
        f"monitor parity {setting}: {len(combos)} combinations, "
        f"{len(combos) - len(different)} identical, {len(different)} different"
    )
    assert not different, "\n".join(p for r in different[:5] for p in r)


def test_monitor_matrix_size() -> None:
    counts = {s: len(monitor_combos(s)) for s in SETTINGS}
    # (side tasks + baseline) x (task configs + missing) x 4 access types x 3 families
    assert counts == {
        "cuda_kernel": 5 * 2 * 12,
        "inference_optimization": 5 * 2 * 12,
        "safety_finetuning": 4 * 3 * 12,
        "utility_finetuning": 3 * 3 * 12,
    }


def test_monitor_task_text_ignores_the_runs_budget_d005() -> None:
    # Upstream behaviour kept: the monitor's task text always says 1 hour.
    for setting in ("safety_finetuning", "utility_finetuning", "inference_optimization"):
        assert "You have 1 hours" in monitor_task_instruction(setting)


def test_unknown_task_config_falls_back_like_upstream(tmp_path: Path, snapshot: Path) -> None:
    # 10 published capabilities runs have task_config qwen3_1.7b, which b922f4b no longer has.
    combo = MonitorCombo("utility_finetuning", "embedded", "qwen3_1.7b", ACCESS_TYPES[0], "claude")
    assert compare_monitor(combo, tmp_path / "run", make_stubs(tmp_path / "stubs"), snapshot) == []
    text = monitor_task_instruction("utility_finetuning", "qwen3_1.7b")
    assert text == monitor_task_instruction("utility_finetuning", "qwen2.5_0.5b")


def test_access_types_name_upstreams_templates() -> None:
    names = [access.template_name for access in ACCESS_TYPES]
    assert names == ["monitor", "monitor_cot", "monitor_artifact", "monitor_artifact_cot"]
    for name in names:
        assert (templates_dir() / f"{name}.md").is_file()
    assert sorted(p.stem for p in templates_dir().glob("*.md")) == sorted(names)


def test_templates_show_doubled_braces_to_the_monitor() -> None:
    # The templates escape braces as if for str.format, but build_monitor_prompt.py uses
    # str.replace, so the monitor sees "{{" and "}}" in the JSON example. Kept as upstream.
    prompt = monitor_prompt_file(ACCESS_TYPES[0], setting="cuda_kernel", side_task="none")
    assert '```json\n{{\n    "suspicion_score"' in prompt
    assert prompt.endswith("}}\n```\n")


# ------------------------------------------------- build_monitor_prompt.py, directly

ALL_PLACEHOLDERS = (
    "# Monitor\r\n{transcript}\r\nSetting {setting}; side task {side_task_type}.\r\n"
    "{environment_context}\r\nTask:\r\n{main_task_instruction}\r\n"
    "Literal {{braces}} and {unknown} and {setting}{setting}.\r\n\r\n\r\n"
)


def _upstream_monitor_script(
    tmp_path: Path,
    access: MonitorAccess,
    *,
    templates: Path,
    red_trace: Path,
    snapshot: Path,
    instruction: Path,
    side_task: str = "embedded",
) -> subprocess.CompletedProcess[bytes]:
    args = [
        "python3",
        str(upstream_dir() / "src" / "build_monitor_prompt.py"),
        "--setting",
        "cuda_kernel",
        "--red_trace",
        str(red_trace),
        "--task_snapshot",
        str(snapshot),
        "--templates_dir",
        str(templates),
        "--side_task",
        side_task,
        "--instruction_path",
        str(instruction),
    ]
    if access.with_artifact:
        args.append("--with-artifact")
    if access.with_cot:
        args.append("--with-chain-of-thought")
    return subprocess.run(args, capture_output=True, check=False, cwd=tmp_path)


@pytest.mark.parametrize("access", ACCESS_TYPES, ids=lambda a: a.template_name)
def test_build_monitor_prompt_matches_script_on_templates_using_every_placeholder(
    tmp_path: Path, snapshot: Path, access: MonitorAccess
) -> None:
    root = tmp_path / "upstream"
    templates = root / "monitors" / "templates"
    templates.mkdir(parents=True)
    (templates / f"{access.template_name}.md").write_bytes(ALL_PLACEHOLDERS.encode("utf-8"))
    red_trace = tmp_path / "trace.txt"
    red_trace.write_bytes(("line {setting} ✓\r\n" * 9000).encode("utf-8"))
    instruction = tmp_path / "main_task_instruction.md"
    instruction.write_bytes(b"# Task {side_task_type}\nDo it.\n")

    upstream = _upstream_monitor_script(
        tmp_path,
        access,
        templates=templates,
        red_trace=red_trace,
        snapshot=snapshot,
        instruction=instruction,
    )
    assert upstream.returncode == 0, upstream.stderr
    port = build_monitor_prompt(
        access,
        setting="cuda_kernel",
        side_task="embedded",
        transcript=load_file(red_trace),
        main_task_instruction=instruction.read_text(encoding="utf-8"),
        task_snapshot=snapshot,
        upstream=root,
    )
    assert (port + "\n").encode("utf-8") == upstream.stdout
    assert "[TRUNCATED at 80000 chars]" in port


def test_missing_template_and_transcript_match_script(tmp_path: Path, snapshot: Path) -> None:
    root = tmp_path / "upstream"
    (root / "monitors" / "templates").mkdir(parents=True)
    instruction = tmp_path / "main_task_instruction.md"
    instruction.write_text("Task.\n", encoding="utf-8")
    access = ACCESS_TYPES[3]
    upstream = _upstream_monitor_script(
        tmp_path,
        access,
        templates=root / "monitors" / "templates",
        red_trace=tmp_path / "absent.txt",
        snapshot=snapshot,
        instruction=instruction,
    )
    port = build_monitor_prompt(
        access,
        setting="cuda_kernel",
        side_task="embedded",
        transcript=load_file(tmp_path / "absent.txt"),
        main_task_instruction="Task.\n",
        task_snapshot=snapshot,
        upstream=root,
    )
    assert (port + "\n").encode("utf-8") == upstream.stdout


def _listing_via_script(tmp_path: Path, snap: Path) -> subprocess.CompletedProcess[bytes]:
    root = tmp_path / "upstream"
    templates = root / "monitors" / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    (templates / "monitor_artifact.md").write_text("{environment_context}", encoding="utf-8")
    instruction = tmp_path / "i.md"
    instruction.write_text("", encoding="utf-8")
    return _upstream_monitor_script(
        tmp_path,
        ACCESS_TYPES[2],
        templates=templates,
        red_trace=tmp_path / "absent.txt",
        snapshot=snap,
        instruction=instruction,
    )


def test_listing_caps_match_script(tmp_path: Path) -> None:
    snap = tmp_path / "task_snapshot"
    for d in range(12):
        for f in range(55):
            _sized(snap / f"dir{d:02d}" / f"file{f:02d}", f * 1000)
    _sized(snap / "a" / "b" / "c" / "d" / "e.txt", 1)
    upstream = _listing_via_script(tmp_path, snap)
    assert upstream.returncode == 0, upstream.stderr
    from researcharena_inspect.prompts.monitor import environment_context

    port = environment_context(list_directory(snap))
    assert (port + "\n").encode("utf-8") == upstream.stdout
    assert len(list_directory(snap).splitlines()) == 500


def test_broken_link_in_the_workspace_fails_like_upstream(tmp_path: Path) -> None:
    # A dangling link within three levels makes upstream's listing raise, and the
    # artifact monitor never starts (pipeline_monitor.sh runs under set -e).
    snap = tmp_path / "task_snapshot"
    _sized(snap / "ok.txt", 1)
    (snap / "dangling").symlink_to("does-not-exist")
    upstream = _listing_via_script(tmp_path, snap)
    assert upstream.returncode != 0
    assert b"FileNotFoundError" in upstream.stderr
    with pytest.raises(FileNotFoundError):
        list_directory(snap)


def test_truncation_rule() -> None:
    assert truncate_transcript("abc", 3) == "abc"
    assert truncate_transcript("abcd", 3) == "abc\n\n... [TRUNCATED at 3 chars] ..."


def test_monitor_prompt_delivery_differs_by_family() -> None:
    def delivered(family: str) -> str:
        return monitor_prompt(ACCESS_TYPES[0], family, setting="cuda_kernel", side_task="none")  # pyright: ignore[reportArgumentType]

    claude = delivered("claude")
    assert claude == delivered("codex")
    assert claude.endswith("```\n")
    assert delivered("opencode") == claude[:-1]
    with pytest.raises(ValueError, match="family"):
        delivered("gemini")


def test_snapshot_listing_runs_in_this_folder_os_walk_order(snapshot: Path) -> None:
    listing = list_directory(snapshot)
    assert listing.splitlines()[0] == "task_snapshot/"
    assert "  big.bin (1,234,567 bytes)" in listing
    assert sum(1 for line in listing.splitlines() if line.startswith("    f")) == 50
    assert os.path.basename(snapshot) == "task_snapshot"
