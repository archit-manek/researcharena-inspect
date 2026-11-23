"""Replaying a workspace: a given `task/` in place of the agent's work, then scored as usual.

The files come from a published run's `red_team/task_snapshot/` or a local run folder. A
published snapshot was taken after upstream's scorer ran in it, and CUDA's `evaluate.py` and
`bench.py` read back results left there (D-001; `evaluate.py:32-53`, `bench.py:190-194`). So
`as_published` replays it as it is, as upstream's `src/reeval_only.sh` does, and `fresh` (D-098,
the default) leaves those files out so this GPU measures. Other settings read back nothing.
Side-task evaluators read none of these files, so side verdicts are the same in both modes.

Warnings:
- A published inference snapshot's compile caches (`.vllm_cache/` and the like) may hold what
  the published evaluation wrote (D-110). vLLM keys them by version, so one from another
  version should go unused (not checked; `c17312128` and `c17360448` carry them).
- Post-training models are published apart from the trajectories and put back in place
  (`published_model_files`, D-172). Files go in by pieces or bounded tars, so the host never
  holds a whole model in memory.
- A published `independent` run's evidence is not published, so its replayed side score is
  that of a run that uploaded nothing.
"""

import gzip
import hashlib
import io
import json
import os
import shlex
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from inspect_ai.util import sandbox

from researcharena_inspect.environment.layout import AGENT_UID, AGENT_USER, TASK_DIR
from researcharena_inspect.environment.staging import UMASK
from researcharena_inspect.scoring.submission_store import MODEL_FOLDER, MODEL_SETTINGS
from researcharena_inspect.scoring.transfer import copy_in

ReplayMode = Literal["as_published", "fresh"]
REPLAY_MODES: tuple[ReplayMode, ...] = ("as_published", "fresh")

SCORER_CACHE_FILES = (
    "workspace/final_results.json",
    "final_results.json",
    "workspace/last_bench.json",
    "workspace/baseline_tps.txt",
)
"""Files `cuda_kernel`'s `evaluate.py` (`:33-37`) and `bench.py` (`:190-194`) read as earlier
results."""

SCORER_CACHE_FILES_BY_SETTING: dict[str, tuple[str, ...]] = {
    "cuda_kernel": SCORER_CACHE_FILES,
    "inference_optimization": (),
    "utility_finetuning": (),
    "safety_finetuning": (),
}
"""The files `fresh` leaves out, by setting; a setting not listed keeps CUDA's list."""

SNAPSHOT = "red_team/task_snapshot"

LARGE_FILE_BYTES = 32 * 1024 * 1024
"""Files this large or larger go into the sandbox in pieces (`transfer.copy_in`)."""

TAR_BATCH_BYTES = 64 * 1024 * 1024
"""Most file bytes in one tar of smaller files."""

FileSource = bytes | Path
"""A replayed file's contents, or the host file that holds them."""


@dataclass(frozen=True)
class ReplaySource:
    """Where a replayed workspace comes from, and the run it belongs to."""

    source: str
    """A published run id (`cuda_kernel/<run>`) or a local run folder."""
    setting: str
    side_task: str
    """Upstream's side task folder (`none` for a baseline; `embedded3` for a published
    inference run's `independent2`)."""
    task_config: str | None
    redruntime: str
    published: bool


def replay_source(source: str) -> ReplaySource:
    """Describe a replay source: a local run folder (with `metadata.json`) or a published run."""
    from researcharena_inspect.scoring.metrics_json import upstream_side_task

    path = Path(source)
    if path.is_dir():
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return ReplaySource(
            source=str(path),
            setting=meta["setting"],
            side_task=upstream_side_task(meta["setting"], meta["side_task"]),
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
        # D-050: the published runs' `independent2` is upstream's `embedded3`.
        side_task=upstream_side_task(run["setting"], run["side_task"]),
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


class ReplayError(RuntimeError):
    """The replayed workspace is missing, incomplete, unverified or over the caps.

    Never replaced by an empty workspace: that would score as a genuine "no model" 0.0 (D-172).
    """


REPLAY_MAX_BYTES = 24 * 1024**3
"""Cap on a replayed workspace (the store's 20 GiB cap plus room)."""

REPLAY_MAX_FILES = 100_000


@dataclass
class ReplayInput:
    """What a replay puts in place, and where it came from (for the replay's record)."""

    files: dict[str, "FileSource"]
    provenance: dict[str, Any]
    evidence: Path | None = None
    """A stored run's `side_task_evidence/` folder, for the hook service (`services`)."""


LISTED_SETTINGS: tuple[str, ...] = ("utility_finetuning", "safety_finetuning")
"""Settings whose published models a pinned listing lists (`pins.OUTPUT_MODELS_LISTINGS`)."""


@cache
def model_listing(setting: str = "utility_finetuning") -> dict[str, Any]:
    """The pinned listing of a setting's published models, checked against its pinned SHA-256.

    One entry per `final_model/` file: size and SHA-256 (LFS files) or git blob id (small files).
    """
    from researcharena_inspect.pins import OUTPUT_MODELS_LISTINGS, OUTPUT_MODELS_REVISION

    path, sha256 = OUTPUT_MODELS_LISTINGS[setting]
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ReplayError(f"{path} does not match its pinned SHA-256")
    data = json.loads(raw)
    if data["revision"] != OUTPUT_MODELS_REVISION:
        raise ReplayError("the model listing is not for the pinned revision")
    return data


def _git_blob_id(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def published_model_files(run_id: str) -> tuple[dict[str, Path], dict[str, Any]]:
    """A published post-training run's `final_model/` files, checked against the pinned listing.

    The local copy must hold exactly the listed files, each with its size and hash. Empty for a
    run that saved no model.

    Returns:
        The files by path relative to `task_snapshot/`, and their provenance (dataset,
        revision, file count and bytes) for the replay's record.

    Raises:
        ReplayError: a file is missing, extra, of another size or another hash, or `run_id`
            is not a published run of a setting the listing covers.
    """
    from inspect_evals_utils.huggingface import snapshot_download

    from researcharena_inspect.pins import OUTPUT_MODELS_DATASET, OUTPUT_MODELS_REVISION

    setting = run_id.split("/", 1)[0]
    if setting not in LISTED_SETTINGS:
        raise ReplayError(f"{run_id!r}: no pinned listing of published models for {setting!r}")
    runs = model_listing(setting)["runs"]  # keyed by the published run id, `<setting>/<run>`
    if run_id not in runs:
        from researcharena_inspect.published import load_index

        if run_id not in {run["id"] for run in load_index()["runs"]}:
            raise ReplayError(f"{run_id!r} is not a published run")
    # No entry: the run saved no `final_model/` (D-212).
    listed = {e["path"]: e for e in runs.get(run_id, [])}
    provenance: dict[str, Any] = {
        "dataset": OUTPUT_MODELS_DATASET,
        "revision": OUTPUT_MODELS_REVISION,
        "files": len(listed),
        "bytes": sum(e["size"] for e in listed.values()),
    }
    if not listed:
        return {}, provenance
    prefix = f"{run_id}/{SNAPSHOT}/{MODEL_FOLDER}"
    root = Path(
        snapshot_download(
            repo_id=OUTPUT_MODELS_DATASET,
            repo_type="dataset",
            revision=OUTPUT_MODELS_REVISION,
            allow_patterns=[f"{prefix}/*"],
        )
    )
    base = root / run_id / SNAPSHOT
    model = base / MODEL_FOLDER
    found = (
        {p.relative_to(base).as_posix(): p for p in sorted(model.rglob("*")) if p.is_file()}
        if model.is_dir()
        else {}
    )
    if set(found) != set(listed):
        raise ReplayError(
            f"{run_id}: the local model does not match the pinned listing: missing "
            f"{sorted(set(listed) - set(found))[:10]}, not listed {sorted(set(found) - set(listed))[:10]}"
        )
    for rel, path in found.items():
        entry = listed[rel]
        if path.stat().st_size != entry["size"]:
            raise ReplayError(
                f"{run_id}: {rel} has {path.stat().st_size} bytes, pinned {entry['size']}"
            )
        if "sha256" in entry:
            ok = _sha256(path) == entry["sha256"]
        else:
            ok = _git_blob_id(path) == entry["git_blob"]
        if not ok:
            raise ReplayError(f"{run_id}: {rel} does not match the pinned listing's hash")
    provenance["verified"] = True
    return found, provenance


def _check_caps(files: Mapping[str, "FileSource"], where: str) -> None:
    total = sum(_size(v) for v in files.values())
    if len(files) > REPLAY_MAX_FILES or total > REPLAY_MAX_BYTES:
        raise ReplayError(
            f"{where}: {len(files)} files, {total} bytes; caps {REPLAY_MAX_FILES} files, "
            f"{REPLAY_MAX_BYTES} bytes"
        )


def local_run_files(run_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """The files of a local run folder's `red_team/task_snapshot/`, checked.

    A link is followed only to a regular file inside the run folder, so no file outside it is
    read on the host; other links are listed in `refused_links`. A folder the submission store
    wrote must be complete and match its `manifest.json`.

    Raises:
        ReplayError: an empty, unfinished or altered snapshot, or one over the caps.
    """
    run_dir = run_dir.resolve()
    root = run_dir / SNAPSHOT
    if not root.is_dir() or root.is_symlink():
        raise ReplayError(f"{run_dir} has no {SNAPSHOT}/ folder: nothing to replay")
    files: dict[str, Path] = {}
    refused: list[dict[str, str]] = []
    for folder, dirs, names in os.walk(root, followlinks=False):
        here = Path(folder)
        for name in sorted(dirs):
            if (here / name).is_symlink():
                refused.append(
                    {
                        "path": (here / name).relative_to(root).as_posix(),
                        "reason": "a link to a folder",
                    }
                )
        for name in sorted(names):
            path = here / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                real = Path(os.path.realpath(path))
                if not real.is_relative_to(run_dir) or not real.is_file():
                    refused.append({"path": rel, "reason": "a link out of the run folder"})
                    continue
            elif not path.is_file():
                refused.append({"path": rel, "reason": "not a regular file"})
                continue
            files[rel] = path
    if not files:
        raise ReplayError(f"{run_dir}: {SNAPSHOT}/ holds no file: nothing to replay")
    if run_dir.name.endswith(".incomplete") or (run_dir / ".transfer").exists():
        raise ReplayError(f"{run_dir} is a store that did not finish")
    _check_caps(files, str(run_dir))
    provenance: dict[str, Any] = {"run_folder": str(run_dir), "refused_links": refused}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "stored":
            raise ReplayError(
                f"{run_dir}: the store's status is {manifest.get('status')!r}; only a "
                "complete store ('stored') is replayed"
            )
        listed = {e["path"]: e for e in manifest.get("entries", []) if e.get("kind") == "file"}
        links = {e["path"] for e in manifest.get("entries", []) if e.get("kind") == "link"}
        on_disk = set(files) - links
        if on_disk != set(listed):
            raise ReplayError(
                f"{run_dir}: files differ from the manifest: missing "
                f"{sorted(set(listed) - on_disk)[:10]}, not listed {sorted(on_disk - set(listed))[:10]}"
            )
        for rel, entry in listed.items():
            path = files[rel]
            if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
                raise ReplayError(f"{run_dir}: {rel} differs from the manifest")
        provenance["manifest"] = {"status": manifest["status"], "files": len(listed)}
    return files, provenance


def replay_input(source: "ReplaySource") -> ReplayInput:
    """The replayed workspace's files (host paths, read when put in place) and their provenance.

    Raises:
        ReplayError: a missing, incomplete or altered source, conflicting files, or the caps.
    """
    if source.published:
        from researcharena_inspect.monitor.bundle import fetch_workspace
        from researcharena_inspect.pins import HF_DATASET, HF_REVISION

        root = fetch_workspace(source.source)
        if not root.is_dir():
            raise ReplayError(f"no published task_snapshot for {source.source}")
        files: dict[str, FileSource] = {
            path.relative_to(root).as_posix(): path
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        provenance: dict[str, Any] = {"workspace": {"dataset": HF_DATASET, "revision": HF_REVISION}}
        if source.setting in MODEL_SETTINGS:
            model, provenance["model"] = published_model_files(source.source)
            clash = sorted(set(files) & set(model))
            if clash or any(k.startswith(f"{MODEL_FOLDER}/") for k in files):
                raise ReplayError(f"{source.source}: the trajectories already hold {MODEL_FOLDER}/")
            files.update(model)
        _check_caps(files, source.source)
        return ReplayInput(files, provenance)
    run_dir = Path(source.source)
    local, provenance = local_run_files(run_dir)
    evidence = run_dir / "side_task_evidence"
    return ReplayInput(dict(local), provenance, evidence if evidence.is_dir() else None)


def snapshot_sources(source: "ReplaySource") -> dict[str, "FileSource"]:
    """The replayed workspace's files (`replay_input(source).files`)."""
    return replay_input(source).files


def _size(value: FileSource) -> int:
    return len(value) if isinstance(value, bytes) else value.stat().st_size


def _content(value: FileSource) -> bytes:
    return value if isinstance(value, bytes) else value.read_bytes()


def _mode(rel: str) -> int:
    """644, and `chmod +x` on top-level `*.sh` and `*.py` (`pipeline_redteam.sh:185`)."""
    top_level_script = len(PurePosixPath(rel).parts) == 1 and rel.endswith((".sh", ".py"))
    return (0o777 if top_level_script else 0o666) & ~UMASK


def select_files(
    files: Mapping[str, FileSource], mode: ReplayMode, setting: str = "cuda_kernel"
) -> dict[str, FileSource]:
    """The files to put in place: all of them, or without the setting's scorer caches (`fresh`)."""
    if mode not in REPLAY_MODES:
        raise ValueError(f"replay mode must be one of {REPLAY_MODES}, got {mode!r}")
    if mode == "as_published":
        return dict(files)
    left_out = SCORER_CACHE_FILES_BY_SETTING.get(setting, SCORER_CACHE_FILES)
    return {k: v for k, v in files.items() if k not in left_out}


def snapshot_tar(files: Mapping[str, FileSource], dirs: set[str] | None = None) -> bytes:
    """A tar of the files in `task/`'s layout, with the modes of `_mode` (Hugging Face keeps none).

    `dirs` holds the folders already written, shared between several tars of one workspace.
    """
    buffer = io.BytesIO()
    dirs = set() if dirs is None else dirs
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
            info.mode = _mode(rel)
            info.uid = info.gid = AGENT_UID
            content = _content(files[rel])
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _batches(files: Mapping[str, FileSource]) -> list[dict[str, FileSource]]:
    """The smaller files in tars of at most `TAR_BATCH_BYTES` (one for a small workspace)."""
    batches: list[dict[str, FileSource]] = [{}]
    total = 0
    for rel in sorted(files):
        size = _size(files[rel])
        if batches[-1] and total + size > TAR_BATCH_BYTES:
            batches.append({})
            total = 0
        batches[-1][rel] = files[rel]
        total += size
    return batches


SCRATCH = "/var/tmp/researcharena-replay.tar"
PIECE_SCRATCH = "/var/tmp/researcharena-replay-piece"


async def replay_workspace(
    files: Mapping[str, FileSource], mode: ReplayMode, setting: str = "cuda_kernel"
) -> dict[str, Any]:
    """Replace the staged `task/` with `files` as the agent's user.

    Returns:
        The mode, the files written, the files left out, and for large files (`streamed`)
        each one's size and SHA-256 as the sandbox computed it.
    """
    chosen = select_files(files, mode, setting)
    large = {k: v for k, v in chosen.items() if _size(v) >= LARGE_FILE_BYTES}
    small = {k: v for k, v in chosen.items() if k not in large}
    box = sandbox()
    clear = await box.exec(
        ["bash", "-c", f"set -e; find {TASK_DIR} -mindepth 1 -delete"], user="root"
    )
    if not clear.success:
        raise RuntimeError(f"clearing {TASK_DIR} failed: {clear.stderr}")
    dirs: set[str] = set()
    for batch in _batches(small):
        await box.write_file(SCRATCH, snapshot_tar(batch, dirs))
        readable = await box.exec(["chmod", "644", SCRATCH], user="root")
        if not readable.success:
            raise RuntimeError(f"preparing the replayed workspace failed: {readable.stderr}")
        unpack = await box.exec(
            ["bash", "-c", f"umask 022; tar -xpmf {SCRATCH} --no-same-owner -C {TASK_DIR}"],
            user=AGENT_USER,
        )
        if not unpack.success:
            raise RuntimeError(f"writing the replayed workspace failed: {unpack.stderr}")
    await box.exec(["rm", "-f", SCRATCH], user="root")
    streamed = []
    for rel in sorted(large):
        source = large[rel]
        dest = f"{TASK_DIR}/{rel}"
        parent = str(PurePosixPath(dest).parent)
        made = await box.exec(
            ["bash", "-c", f"umask 022; mkdir -p {shlex.quote(parent)}"], user=AGENT_USER
        )
        if not made.success:
            raise RuntimeError(f"creating {parent} failed: {made.stderr}")
        if isinstance(source, bytes):
            await box.write_file(PIECE_SCRATCH, source)
            moved = await box.exec(
                [
                    "bash",
                    "-c",
                    f"chmod 644 {PIECE_SCRATCH} && cat {PIECE_SCRATCH} > {shlex.quote(dest)}",
                ],
                user="root",
            )
            await box.exec(["chown", f"{AGENT_UID}:{AGENT_UID}", dest], user="root")
            if not moved.success:
                raise RuntimeError(f"writing {dest} failed: {moved.stderr}")
            digest = hashlib.sha256(source).hexdigest()
        else:
            digest = await copy_in(box, source, dest, user=AGENT_USER, scratch=PIECE_SCRATCH)
        chmod = await box.exec(["chmod", f"{_mode(rel):o}", dest], user=AGENT_USER)
        if not chmod.success:
            raise RuntimeError(f"setting the mode of {dest} failed: {chmod.stderr}")
        streamed.append({"path": rel, "bytes": _size(source), "sha256": digest})
    await box.exec(["rm", "-f", PIECE_SCRATCH], user="root")
    record: dict[str, Any] = {
        "mode": mode,
        "files": sorted(chosen),
        "left_out": sorted(set(files) - set(chosen)),
    }
    if streamed:
        record["streamed"] = streamed
    return record


__all__ = [
    "REPLAY_MODES",
    "SCORER_CACHE_FILES",
    "SCORER_CACHE_FILES_BY_SETTING",
    "ReplayMode",
    "ReplaySource",
    "local_snapshot_files",
    "published_model_files",
    "published_snapshot_files",
    "replay_source",
    "replay_workspace",
    "select_files",
    "snapshot_files",
    "snapshot_sources",
    "snapshot_tar",
]
