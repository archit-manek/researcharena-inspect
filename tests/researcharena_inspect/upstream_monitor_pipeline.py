"""Run upstream's `src/pipeline_monitor.sh` for real, with a probe in place of the monitor CLI.

The script runs unchanged from a clean export of the pinned commit, inside a Debian container
(it needs bash >= 4.4 and coreutils `timeout`). With no Apptainer image it takes its own host
branch (`pipeline_monitor.sh:311-327`) and runs `monitors/claude/monitor.sh` as the monitor. In
the export, and only there, that one file is replaced by a probe that copies the job folder the
pipeline staged to `/capture/staged`, then writes a judgement or prints output as the test asks.
No model is called and no credentials exist in the container (`HOME` is an empty folder).
"""

import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.upstream import upstream_dir

IMAGE = "python:3.12-slim-bookworm"

PROBE = """#!/usr/bin/env bash
# Test probe standing in for the monitor CLI (see tests/.../upstream_monitor_pipeline.py).
set -euo pipefail
JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /capture/staged
cp -a "$JOB_DIR/." /capture/staged/
pwd > /capture/cwd.txt
case "${PROBE_MODE:-none}" in
    task) printf '%s' "$PROBE_TEXT" > "$JOB_DIR/task/judgement.json" ;;
    home) printf '%s' "$PROBE_TEXT" > "$JOB_DIR/judgement.json" ;;
    stdout) printf '%s' "$PROBE_TEXT" ;;
    none) ;;
esac
"""

# Launch helpers and credentials the pipeline puts in the monitor's home; the port does not
# stage them (Inspect starts the monitor itself).
LAUNCH_FILES = {"monitor_solve.sh", "update_agent_cli.sh", "check_effort.py"}


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


def export_arena(dest: Path, commit: str = UPSTREAM_COMMIT) -> Path:
    """A clean copy of upstream at `commit` with the probe as the Claude monitor."""
    archive = subprocess.run(
        ["git", "-C", str(upstream_dir(commit)), "archive", "--format=tar", commit],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(dest, filter="tar")
    probe = dest / "monitors" / "claude" / "monitor.sh"
    probe.write_text(PROBE, encoding="utf-8")
    probe.chmod(0o755)
    return dest


@dataclass
class PipelineResult:
    staged: dict[str, tuple[bytes, int]]
    """The monitor's home as the pipeline staged it: relative path -> (bytes, mode)."""
    results: dict | None
    """The run's `results.json`, or None if the pipeline wrote none."""
    judgement: str | None
    """The run's `judgement.json` as the pipeline left it."""
    monitor_prompt: str
    cwd: str
    returncode: int
    log: str


def _read_tree(root: Path) -> dict[str, tuple[bytes, int]]:
    tree = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rel = path.relative_to(root).as_posix()
            tree[rel] = (path.read_bytes(), path.stat().st_mode & 0o777)
    return tree


def run_pipeline(
    work: Path,
    arena: Path,
    eval_dir: Path,
    *,
    with_artifact: bool,
    with_cot: bool,
    model: str = "claude-sonnet-4-6",
    probe_mode: str = "none",
    probe_text: str = "",
) -> PipelineResult:
    """Run `pipeline_monitor.sh <eval_dir> 0.01 <artifact> claude <model> <cot>` in Docker.

    `arena`, `eval_dir` and the capture folder must be inside `work`, which is mounted at /work.
    """
    capture = work / "capture"
    for stale in (capture, eval_dir / "blue_team"):
        if stale.exists():
            shutil.rmtree(stale)
    capture.mkdir()
    home = work / "home"
    home.mkdir(exist_ok=True)

    def inside(p: Path) -> str:
        return "/work/" + p.relative_to(work).as_posix()

    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{work}:/work",
        "-v",
        f"{capture}:/capture",
        "-e",
        "HOME=/work/home",
        "-e",
        f"ARENA_ROOT={inside(arena)}",
        "-e",
        "ARENA_CONTAINER_SIF=/nonexistent/arena.sif",
        "-e",
        f"PROBE_MODE={probe_mode}",
        "-e",
        f"PROBE_TEXT={probe_text}",
        IMAGE,
        "bash",
        f"{inside(arena)}/src/pipeline_monitor.sh",
        inside(eval_dir),
        "0.01",
        "true" if with_artifact else "false",
        "claude",
        model,
        "true" if with_cot else "false",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    suffix = ("_artifact" if with_artifact else "") + ("_cot" if with_cot else "")
    run_dir = eval_dir / "blue_team" / f"claude_{model}_high{suffix}"
    results_path = run_dir / "results.json"
    judgement_path = run_dir / "judgement.json"
    prompt_path = run_dir / "monitor_prompt.md"
    cwd_path = capture / "cwd.txt"
    return PipelineResult(
        staged=_read_tree(capture / "staged") if (capture / "staged").is_dir() else {},
        results=json.loads(results_path.read_text()) if results_path.is_file() else None,
        judgement=judgement_path.read_text() if judgement_path.is_file() else None,
        monitor_prompt=prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else "",
        cwd=cwd_path.read_text().strip() if cwd_path.is_file() else "",
        returncode=proc.returncode,
        log=proc.stdout + proc.stderr,
    )
