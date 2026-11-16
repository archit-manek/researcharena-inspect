"""Replaying a workspace: a given `task/` in place of the agent's work, then scored as usual.

A replayed sample is staged as any run is; in the agent's slot the staged `task/` is replaced by
the given files, as the agent's user; then the scorer runs on it exactly as after an agent. The
files come from a published run's `red_team/task_snapshot/` (Hugging Face, pinned revision) or
from a local run folder in upstream's layout.

A published `task_snapshot/` is the workspace after the agent **and** after upstream's scorer
ran in it (WP2, discrepancy 4): `evaluate.py` returns the speedup in
`workspace/final_results.json`, `final_results.json` or `workspace/last_bench.json` without
benchmarking anything when one says `compile_status` PASS (`evaluate.py:32-53`, ledger D-001),
and `bench.py` reuses `workspace/baseline_tps.txt` as the baseline if it exists
(`bench.py:190-194`), whether the agent or a previous scoring wrote it. So two modes:

- `as_published`: the snapshot as it is. Upstream's own re-evaluation (`src/reeval_only.sh`)
  does this; for the 16 runs with `last_bench.json` it only reads that file back, and elsewhere
  a cached baseline measured on another GPU is divided into this GPU's agent throughput.
- `fresh` (the fair re-score, the default): the snapshot without those four files, so
  `evaluate.py` compiles the kernels and measures the baseline and the agent's kernels on this
  GPU, in one session, as it would have for an agent that never ran `bench.py`. The side-task
  evaluators read none of the four files, so their verdicts are the same in both modes.
"""

import io
import json
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from inspect_ai.util import sandbox

from researcharena_inspect.environment.layout import AGENT_UID, AGENT_USER, TASK_DIR
from researcharena_inspect.environment.staging import UMASK

ReplayMode = Literal["as_published", "fresh"]
REPLAY_MODES: tuple[ReplayMode, ...] = ("as_published", "fresh")

SCORER_CACHE_FILES = (
    "workspace/final_results.json",
    "final_results.json",
    "workspace/last_bench.json",
    "workspace/baseline_tps.txt",
)
"""Files `evaluate.py` (`:33-37`) and `bench.py` (`:190-194`) read as earlier results."""

SNAPSHOT = "red_team/task_snapshot"


@dataclass(frozen=True)
class ReplaySource:
    """Where a replayed workspace comes from, and the run it belongs to."""

    source: str
    """A published run id (`cuda_kernel/<run>`) or a local run folder."""
    setting: str
    side_task: str
    """Upstream's side task (`none` for a baseline)."""
    task_config: str | None
    redruntime: str
    published: bool


def replay_source(source: str) -> ReplaySource:
    """Describe a replay source: a local run folder (with `metadata.json`) or a published run."""
    path = Path(source)
    if path.is_dir():
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return ReplaySource(
            source=str(path),
            setting=meta["setting"],
            side_task=meta["side_task"],
            task_config=meta.get("task_config"),
            redruntime=str(meta.get("redruntime_hours", "1")),
            published=False,
        )
    from researcharena_inspect.published import load_index

    runs = {run["id"]: run for run in load_index()["runs"]}
    if source not in runs:
        raise ValueError(f"{source!r} is neither a run folder nor a published run id")
    run = runs[source]
    return ReplaySource(
        source=source,
        setting=run["setting"],
        side_task="none" if run["side_task"] == "baseline" else run["side_task"],
        task_config=run.get("task_config"),
        redruntime=str(run.get("hours", 1)),
        published=True,
    )


def published_snapshot_files(run_id: str) -> dict[str, bytes]:
    """Every file of a published run's `task_snapshot/`, by path relative to it."""
    from researcharena_inspect.monitor.bundle import fetch_workspace

    root = fetch_workspace(run_id)
    if not root.is_dir():
        raise FileNotFoundError(f"no published task_snapshot for {run_id}")
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def local_snapshot_files(run_dir: Path) -> dict[str, bytes]:
    """Every file under a local run folder's `red_team/task_snapshot/` (links followed)."""
    root = run_dir / SNAPSHOT
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def snapshot_files(source: ReplaySource) -> dict[str, bytes]:
    if source.published:
        return published_snapshot_files(source.source)
    return local_snapshot_files(Path(source.source))


def select_files(files: Mapping[str, bytes], mode: ReplayMode) -> dict[str, bytes]:
    """The files to put in place: all of them, or without the scorer's caches (`fresh`)."""
    if mode not in REPLAY_MODES:
        raise ValueError(f"replay mode must be one of {REPLAY_MODES}, got {mode!r}")
    if mode == "as_published":
        return dict(files)
    return {k: v for k, v in files.items() if k not in SCORER_CACHE_FILES}


def snapshot_tar(files: Mapping[str, bytes]) -> bytes:
    """A tar of the files in `task/`'s layout, with the modes upstream's staging gives.

    644, and `chmod +x` on top-level `*.sh` and `*.py` (`pipeline_redteam.sh:185`): Hugging
    Face keeps no modes.
    """
    buffer = io.BytesIO()
    dirs: set[str] = set()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for rel in sorted(files):
            parts = PurePosixPath(rel).parts
            for i in range(1, len(parts)):
                folder = "/".join(parts[:i])
                if folder not in dirs:
                    dirs.add(folder)
                    info = tarfile.TarInfo(folder)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o777 & ~UMASK
                    info.uid = info.gid = AGENT_UID
                    tar.addfile(info)
            info = tarfile.TarInfo(rel)
            top_level_script = len(parts) == 1 and rel.endswith((".sh", ".py"))
            info.mode = (0o777 if top_level_script else 0o666) & ~UMASK
            info.uid = info.gid = AGENT_UID
            info.size = len(files[rel])
            tar.addfile(info, io.BytesIO(files[rel]))
    return buffer.getvalue()


SCRATCH = "/var/tmp/researcharena-replay.tar"


async def replay_workspace(files: Mapping[str, bytes], mode: ReplayMode) -> dict[str, Any]:
    """Replace the staged `task/` with `files` as the agent's user.

    Returns:
        What was put in place: the mode, the files written and the cache files left out.
    """
    chosen = select_files(files, mode)
    box = sandbox()
    await box.write_file(SCRATCH, snapshot_tar(chosen))
    clear = await box.exec(
        ["bash", "-c", f"set -e; find {TASK_DIR} -mindepth 1 -delete; chmod 644 {SCRATCH}"],
        user="root",
    )
    if not clear.success:
        raise RuntimeError(f"clearing {TASK_DIR} failed: {clear.stderr}")
    unpack = await box.exec(
        ["bash", "-c", f"umask 022; tar -xpmf {SCRATCH} --no-same-owner -C {TASK_DIR}"],
        user=AGENT_USER,
    )
    if not unpack.success:
        raise RuntimeError(f"writing the replayed workspace failed: {unpack.stderr}")
    await box.exec(["rm", "-f", SCRATCH], user="root")
    return {
        "mode": mode,
        "files": sorted(chosen),
        "left_out": sorted(set(files) - set(chosen)),
    }


__all__ = [
    "REPLAY_MODES",
    "SCORER_CACHE_FILES",
    "ReplayMode",
    "ReplaySource",
    "local_snapshot_files",
    "published_snapshot_files",
    "replay_source",
    "replay_workspace",
    "select_files",
    "snapshot_files",
    "snapshot_tar",
]
