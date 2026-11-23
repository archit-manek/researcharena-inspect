"""The agent's final workspace (`red_team/task_snapshot/`), kept in the `.eval` log (D-110).

Upstream keeps the folder in the run folder (`src/pipeline_redteam.sh:433-438`) as Phase 4 leaves
it: the evaluators read it and write into it (`__pycache__/`, `workspace/last_bench.json`). It is
what the authors published and what an artifact monitor gets (`pipeline_monitor.sh:265-269`).
The port archives it from the sandbox after the last attempt, once leftover processes are killed
and the score is decided, keeps the archive in the sample's store, and
`agents.record.write_run_dir` unpacks it.

Not carried: owners, hard links (`cp -r` makes none), FIFOs, sockets and device files (listed in
the record), and setuid, setgid and sticky bits on restore.
"""

import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from inspect_ai.util import Store, StoreModel
from pydantic import Field

from researcharena_inspect.environment.evaluation import (
    SNAPSHOT_DIR,
    evaluator_sandbox,
    stop_agent_processes,
)
from researcharena_inspect.environment.layout import IMAGE_PATH

logger = logging.getLogger(__name__)

SNAPSHOT_NAME = "task_snapshot"
"""The folder's name in the run folder and the archive's top folder."""

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
"""Largest archive kept in the log (20 MiB, gzipped). The published CUDA snapshots are 32 to
62 KB gzipped; a post-training `final_model` is gigabytes and is not kept."""

ARCHIVE_SCRIPT = Path(__file__).with_name("snapshot_archive.py")

_SCRATCH = "/var/tmp/researcharena-snapshot"
_ARCHIVE_TIMEOUT = 1800

METADATA_KEY = "task_snapshot"
"""Key of the record (without the archive) in the sample's metadata."""


class TaskSnapshot(StoreModel):
    """The agent's final workspace as the run keeps it, in the sample's store."""

    status: str = ""
    """`kept`, `too_large`, `off` (turned off), `error`, or "" (never attempted)."""
    taken: str = "after_evaluation"
    """When it was archived: after the last `evaluate.py` attempt."""
    source: str = SNAPSHOT_DIR
    archive: str = ""
    """The gzipped tar, base64-encoded; its top folder is `task_snapshot`."""
    archive_bytes: int | None = None
    archive_sha256: str | None = None
    max_bytes: int | None = None
    files: int | None = None
    dirs: int | None = None
    links: int | None = None
    raw_bytes: int | None = None
    """Total size of the regular files."""
    largest: list[list[Any]] = Field(default_factory=list)
    """The largest files, as [path, size]."""
    skipped: list[list[str]] = Field(default_factory=list)
    """Entries not archived, as [path, kind]: FIFOs, sockets, device files."""
    excluded: list[list[Any]] = Field(default_factory=list)
    """Top-level folders left out on purpose, as [name, files, bytes]: a post-training run's
    `final_model/`, which the submission store keeps instead (D-172)."""
    seconds: float | None = None
    """How long the step took, from stopping leftover processes to reading the archive."""
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """The record without the archive, for the sample's metadata."""
        # `excluded` only when non-empty, so other settings' records keep their keys.
        return {
            k: v
            for k, v in self.model_dump().items()
            if k != "archive" and not (k == "excluded" and not v)
        }


async def keep_task_snapshot(
    store: Store,
    max_bytes: int = DEFAULT_MAX_BYTES,
    *,
    enabled: bool = True,
    exclude: tuple[str, ...] = (),
) -> TaskSnapshot:
    """Archive the sandbox's `task_snapshot` into the sample's `store` and return the record.

    Call after the last evaluation attempt. Never raises: a failure is `status` `error`, so the
    score is never lost to it. Top-level names in `exclude` are left out and counted.
    """
    record = TaskSnapshot(store=store)
    record.max_bytes = max_bytes
    if not enabled:
        record.status = "off"
        return record
    start = time.monotonic()
    try:
        await _archive_in_sandbox(record, max_bytes, exclude)
    except Exception as e:  # the score must not depend on the snapshot
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"
        record.archive = ""
        logger.warning("task_snapshot not kept: %s", record.error)
    finally:
        record.seconds = round(time.monotonic() - start, 2)
        with contextlib.suppress(Exception):
            await evaluator_sandbox().exec(["rm", "-rf", _SCRATCH], user="root")
    if record.status == "too_large":
        logger.warning(
            "task_snapshot not kept: the archive passed %d bytes (%s bytes in %s files)",
            max_bytes,
            record.raw_bytes,
            record.files,
        )
    return record


async def _archive_in_sandbox(
    record: TaskSnapshot, max_bytes: int, exclude: tuple[str, ...] = ()
) -> None:
    # The evaluator's leftovers (the agent's user) could still write into the folder;
    # upstream's evaluator container is gone at this point.
    await stop_agent_processes(evaluator_sandbox())
    box = evaluator_sandbox()
    prepared = await box.exec(["install", "-d", "-m", "700", _SCRATCH], user="root")
    if not prepared.success:
        raise RuntimeError(f"creating {_SCRATCH} failed: {prepared.stderr}")
    script = f"{_SCRATCH}/snapshot_archive.py"
    output = f"{_SCRATCH}/{SNAPSHOT_NAME}.tar.gz"
    await box.write_file(script, ARCHIVE_SCRIPT.read_bytes())
    argv = ["python3", script, SNAPSHOT_DIR, SNAPSHOT_NAME, output, str(max_bytes)]
    if exclude:
        argv.append(",".join(exclude))
    result = await box.exec(
        argv,
        user="root",
        env={"PATH": IMAGE_PATH},
        timeout=_ARCHIVE_TIMEOUT,
    )
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(
            f"the archiver printed nothing (exit {result.returncode}): {result.stderr}"
        )
    report = json.loads(lines[-1])
    for key in ("files", "dirs", "links", "raw_bytes", "largest", "skipped", "excluded"):
        if key in report:
            setattr(record, key, report[key])
    record.status = report["status"]
    if record.status == "error":
        record.error = report.get("error")
        return
    if record.status != "kept":
        return
    data = await box.read_file(output, text=False)
    if len(data) != report["archive_bytes"]:
        raise RuntimeError(f"read {len(data)} bytes of a {report['archive_bytes']}-byte archive")
    record.archive_bytes = len(data)
    record.archive_sha256 = hashlib.sha256(data).hexdigest()
    record.archive = base64.b64encode(data).decode("ascii")


def snapshot_bytes(record: TaskSnapshot) -> bytes:
    """The kept archive's bytes, checked against its SHA-256."""
    if record.status != "kept" or not record.archive:
        raise ValueError(f"no archive kept (status {record.status or 'not attempted'!r})")
    data = base64.b64decode(record.archive)
    if hashlib.sha256(data).hexdigest() != record.archive_sha256:
        raise ValueError("the archive does not match its SHA-256")
    return data


def _member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or parts[0] != SNAPSHOT_NAME
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValueError(f"unexpected path in the archive: {name!r}")
    return path


def restore_snapshot(archive: bytes, parent: Path) -> dict[str, Any]:
    """Unpack an archive from `keep_task_snapshot` to `parent/task_snapshot`.

    Link targets are written as they were and never followed. Only paths below `task_snapshot`,
    each inside a folder the archive made earlier, are accepted.

    Returns:
        Counts of what was restored, and the paths with dropped mode bits or not restored.

    Raises:
        FileExistsError: `parent/task_snapshot` exists.
        ValueError: the archive holds a path outside `task_snapshot`, a path twice, or an entry
            whose folder it did not make first.
    """
    dest = parent / SNAPSHOT_NAME
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    parent.mkdir(parents=True, exist_ok=True)
    folders: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    report: dict[str, Any] = {"files": 0, "dirs": 0, "links": 0, "dropped_bits": [], "skipped": []}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            path = _member_path(member.name)
            key = path.as_posix()
            if key in seen:
                raise ValueError(f"{key!r} is in the archive twice")
            seen.add(key)
            if len(path.parts) > 1 and path.parent.as_posix() not in folders:
                raise ValueError(f"{key!r} comes before its folder")
            if len(path.parts) == 1 and not member.isdir():
                raise ValueError(f"{key!r} is not a folder")
            target = parent / Path(*path.parts)
            mode = member.mode & 0o777
            if member.mode & 0o7000:
                report["dropped_bits"].append(key)
            if member.isdir():
                target.mkdir(mode=0o700)
                folders[key] = (mode, int(member.mtime))
                report["dirs"] += 1
            elif member.isreg():
                source = tar.extractfile(member)
                assert source is not None
                with open(target, "xb") as handle:
                    while block := source.read(1 << 20):
                        handle.write(block)
                os.chmod(target, mode)
                os.utime(target, (member.mtime, member.mtime))
                report["files"] += 1
            elif member.issym():
                os.symlink(member.linkname, target)
                if os.utime in os.supports_follow_symlinks:
                    os.utime(target, (member.mtime, member.mtime), follow_symlinks=False)
                report["links"] += 1
            else:
                report["skipped"].append(key)
    # Folder modes and times last, deepest first, so read-only folders could be filled.
    for key in sorted(folders, key=lambda k: k.count("/"), reverse=True):
        mode, mtime = folders[key]
        target = parent / Path(*PurePosixPath(key).parts)
        os.chmod(target, mode)
        os.utime(target, (mtime, mtime))
    return report


__all__ = [
    "DEFAULT_MAX_BYTES",
    "METADATA_KEY",
    "SNAPSHOT_NAME",
    "TaskSnapshot",
    "keep_task_snapshot",
    "restore_snapshot",
    "snapshot_bytes",
]
