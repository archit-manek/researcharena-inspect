"""What the monitor finds in its sandbox, staged as `src/pipeline_monitor.sh:163-273` stages it.

The monitor's home is `/home/user` and it starts in `/home/user/task`. Upstream writes, in order:

1. `main_task_instruction.md` (home, `:174`);
2. `task/agent_transcript.txt` (`:262`);
3. `task/task_instruction.md`, a copy of 1 (`:263`);
4. with artifact access, the agent's final workspace into `task/` (`:269`), which can overwrite
   2 and 3 (D-019);
5. `prompt.txt` (home, `:273`).

Upstream's launch scripts and the CLI's credentials are not staged (D-066).
"""

import io
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

SANDBOX_HOME = "/home/user"
"""The monitor's home, upstream's job folder (`--home "$BLUE_JOB_DIR:/home/user"`)."""

SANDBOX_TASK_DIR = f"{SANDBOX_HOME}/task"
"""The monitor's working directory (`--pwd /home/user/task`)."""


@dataclass(frozen=True)
class StagedFile:
    """One file write. `content` is the bytes, or a host file whose bytes are copied."""

    path: str
    """Path relative to the monitor's home."""
    content: bytes | Path
    mode: int = 0o644
    """Permission bits: `cp` keeps the source's; `echo > file` gives 0644."""


def workspace_files(workspace: Path, *, links_outside: bool = False) -> list[tuple[str, Path]]:
    """Files under `workspace` as (relative POSIX path, host path), sorted.

    D-111: symlinked files are followed (the Hugging Face cache stores files as symlinks);
    symlinked and empty folders are left out. A link to a file outside the workspace is followed
    only if `links_outside`: a restored workspace's links were made by the agent, and following
    them on the host would hand host files to the monitor.
    """
    inside = workspace.resolve()
    files = []
    for root, dirs, names in os.walk(workspace):
        dirs.sort()
        for name in sorted(names):
            host = Path(root) / name
            if not host.is_file():
                continue
            leaves = host.is_symlink() and not host.resolve().is_relative_to(inside)
            if leaves and not links_outside:  # D-111
                continue
            rel = PurePosixPath(*host.relative_to(workspace).parts)
            files.append((str(rel), host))
    return files


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def staged_files(
    *,
    transcript: Path,
    main_task_instruction: str,
    prompt: str,
    workspace: Path | None,
    with_artifact: bool,
    workspace_links_outside: bool = False,
) -> list[StagedFile]:
    """The monitor's files in the order upstream writes them; later writes win."""
    instruction = main_task_instruction.encode("utf-8")
    staged = [StagedFile("main_task_instruction.md", instruction)]
    if transcript.is_file():  # `cp ... 2>/dev/null || true`: a missing transcript is skipped
        staged.append(StagedFile("task/agent_transcript.txt", transcript, _mode(transcript)))
    staged.append(StagedFile("task/task_instruction.md", instruction))
    if with_artifact and workspace is not None and workspace.is_dir():
        found = workspace_files(workspace, links_outside=workspace_links_outside)
        staged += [StagedFile(f"task/{rel}", host, _mode(host)) for rel, host in found]
    staged.append(StagedFile("prompt.txt", prompt.encode("utf-8")))
    return staged


def final_tree(staged: list[StagedFile]) -> dict[str, tuple[bytes, int]]:
    """The monitor's files after every write: relative path -> (bytes, permission bits).

    An overwritten file keeps its first mode, as `cp` onto an existing file does.
    """
    tree: dict[str, tuple[bytes, int]] = {}
    for item in staged:
        data = item.content.read_bytes() if isinstance(item.content, Path) else item.content
        tree[item.path] = (data, tree[item.path][1] if item.path in tree else item.mode)
    return tree


def _tar(tree: dict[str, tuple[bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path, (data, mode) in tree.items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


async def write_to_sandbox(staged: list[StagedFile], home: str = SANDBOX_HOME) -> None:
    """Write the staged files into the current sample's sandbox under `home`."""
    archive = "/tmp/researcharena_monitor_inputs.tar"
    await sandbox().write_file(archive, _tar(final_tree(staged)))
    result = await sandbox().exec(
        [
            "sh",
            "-c",
            f'mkdir -p "{home}/task" && tar -xf {archive} -C "{home}" && rm -f {archive}',
        ]
    )
    if not result.success:
        raise RuntimeError(f"staging the monitor's files failed: {result.stderr}")


def sample_staged_files(metadata: dict[str, Any]) -> list[StagedFile]:
    """The monitor's files for a sample made by `monitor.task.bundle_to_sample`."""
    workspace = metadata.get("workspace")
    return staged_files(
        transcript=Path(metadata["transcript_path"]),
        main_task_instruction=metadata["main_task_instruction"],
        prompt=metadata["monitor_prompt"],
        workspace=Path(workspace) if workspace else None,
        with_artifact=metadata["with_artifact"],
        workspace_links_outside=metadata.get("workspace_links_outside", False),
    )


@solver
def stage_monitor_inputs() -> Solver:
    """Write the monitor's inputs (from the sample's metadata) into its sandbox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        await write_to_sandbox(sample_staged_files(state.metadata))
        return state

    return solve
