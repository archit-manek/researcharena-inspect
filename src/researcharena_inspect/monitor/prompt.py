"""The monitor's main-task instruction and prompt, built by running upstream's own scripts.

Runs `src/prompt_builder.py` and `src/build_monitor_prompt.py` from the pinned checkout with the
arguments of `src/pipeline_monitor.sh:163-190`, and keeps the shell's handling of their output.
`researcharena_inspect.prompts` ports the same builders to Python; a test requires the two to agree.
"""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.upstream import upstream_dir


@dataclass(frozen=True)
class MonitorPrompt:
    """What the pipeline writes before it starts the monitor."""

    main_task_instruction: str
    """Content of `main_task_instruction.md` and of the staged `task/task_instruction.md`."""

    prompt: str
    """Content of `prompt.txt` (and of the run's `monitor_prompt.md`): what the monitor CLI reads."""


def _run(args: list[str]) -> str:
    # UTF-8 on every platform, so the output does not depend on the host's locale.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, *args], capture_output=True, check=True, encoding="utf-8", env=env
    )
    return result.stdout


def _echo_into_file(command_output: str) -> str:
    """What `X=$(cmd); echo "$X" > file` leaves in the file."""
    return command_output.rstrip("\n") + "\n"


def upstream_monitor_prompt(
    *,
    setting: str,
    side_task: str,
    task_config: str,
    transcript_path: Path,
    workspace: Path | None,
    with_artifact: bool,
    with_cot: bool,
    commit: str = UPSTREAM_COMMIT,
) -> MonitorPrompt:
    """Build the monitor's instruction and prompt for one run and one access type.

    `side_task` is as in the run's `metadata.json`. Only artifact prompts list `workspace`,
    always under upstream's name `task_snapshot`.
    """
    root = upstream_dir(commit)
    builder = root / "src" / "prompt_builder.py"
    # D-005: like pipeline_monitor.sh, no --redruntime, so the instruction says "1 hours".
    instruction_args = [
        str(builder),
        "--setting",
        setting,
        "--side_task",
        "none",
        "--task_config",
        task_config,
        "--red_provider",
        "none",
        "--settings_dir",
        str(root / "settings"),
    ]
    if "--main_task_only" in builder.read_text(encoding="utf-8"):
        # The paper-era builder has no such flag; its pipeline called it without one.
        instruction_args.insert(5, "--main_task_only")
    main_task_instruction = _echo_into_file(_run(instruction_args))

    with tempfile.TemporaryDirectory(prefix="researcharena_monitor_prompt_") as tmp:
        instruction_path = Path(tmp) / "main_task_instruction.md"
        instruction_path.write_text(main_task_instruction, encoding="utf-8")
        snapshot = Path(tmp) / "red_team" / "task_snapshot"
        if workspace is not None:
            snapshot.parent.mkdir()
            snapshot.symlink_to(workspace.resolve(), target_is_directory=True)
        args = [str(root / "src" / "build_monitor_prompt.py")]
        if with_artifact:
            args.append("--with-artifact")
        if with_cot:
            args.append("--with-chain-of-thought")
        args += [
            "--setting",
            setting,
            "--red_trace",
            str(transcript_path),
            "--task_snapshot",
            str(snapshot),
            "--templates_dir",
            str(root / "monitors" / "templates"),
            "--side_task",
            side_task,
            "--instruction_path",
            str(instruction_path),
            "--settings_dir",
            str(root / "settings"),
        ]
        prompt = _echo_into_file(_run(args))
    return MonitorPrompt(main_task_instruction=main_task_instruction, prompt=prompt)


def resolve_monitor_effort(
    setting: str, model: str, provider: str = "claude", commit: str = UPSTREAM_COMMIT
) -> str:
    """The monitor's effort from `effort.json` (`src/resolve_effort.py`); `default`: no flag."""
    root = upstream_dir(commit)
    out = _run(
        [
            str(root / "src" / "resolve_effort.py"),
            "--setting",
            setting,
            "--team",
            "blue_team",
            "--provider",
            provider,
            "--model",
            model,
        ]
    )
    return out.strip()
