"""The submitted model, kept on the host beside the log, so that the run can be re-scored later.

A post-training run submits `final_model/`, about 1 GB, inside the workspace upstream keeps as
`red_team/task_snapshot/` (`src/pipeline_redteam.sh:433-438`, pinned commit; the authors
published those models in a separate dataset, `pins.OUTPUT_MODELS_DATASET`). The `.eval` log
cannot hold it (D-110 keeps a workspace only up to `snapshot_max_mb`, 20 MiB by default), and it
is lost with the sandbox. So, for the settings in `MODEL_SETTINGS`, after the score is decided the
scorer copies the post-evaluation run folder out of the sandbox into a run folder in upstream's
layout on the host (D-172):

    <store>/<sample>__e<epoch>__<uuid8>/metadata.json             setting, side task, config
    <store>/<sample>__e<epoch>__<uuid8>/red_team/task_snapshot/   the folder upstream's run keeps
    <store>/<sample>__e<epoch>__<uuid8>/eval/                     metrics.json and attempt logs
    <store>/<sample>__e<epoch>__<uuid8>/side_task_evidence/       the capture log, if any
    <store>/<sample>__e<epoch>__<uuid8>/manifest.json             status and every entry

`eval/` and `side_task_evidence/` are where upstream's run folder keeps them (`$EVAL_DIR/eval`,
`$EVAL_DIR/side_task_evidence`), so the stored run carries its raw `metrics.json`, and a replay of
a stored `independent` run can put its capture log back where the side task's evaluator reads it.

`<store>` is, by default (`model_store="auto"`), `<log file without .eval>.submissions/` beside
the log; another folder can be given. The folder name carries the sample's uuid, so retries and
sample ids that sanitise to the same name do not collide; an existing folder is never
overwritten.

How it is written (round A of the review, D-172):

- **Atomically.** Everything goes into `.<name>.incomplete/`; `manifest.json` is written last,
  and the folder is renamed to `<name>` only when every listed file has arrived with its
  SHA-256. A copy that fails leaves the `.incomplete` folder, which replay refuses.
- **Model first, then the caps.** The sandbox lists `final_model/` before anything else and
  stops at `max_files` entries; files are hashed, in that order, while their total is within
  `max_bytes`. So a cap cuts the rest of the workspace, never the model unless the model alone
  is too large, and neither the listing nor the hashing is unbounded.
- **Links.** A symbolic link whose target stays inside `task_snapshot/` is made again on the
  host, as `cp -r` keeps it; a link that points outside it (an absolute path, or one that climbs
  out) is not made and is listed in `left_out` with its reason. FIFOs, sockets and devices
  likewise.
- **Status.** `stored` when nothing was left out; `partial` when anything was (a cap, an
  outside link); `error` when the copy failed. Replay accepts only `stored`.

The score's metadata (`submission_store`) carries the folder, the status, the totals, the model's
entries with their SHA-256 and the first `MANIFEST_IN_LOG` others. A failure is recorded there
and never raised: the score is already decided and is not changed by it.
"""

import hashlib
import json
import os
import posixpath
import re
import time
from pathlib import Path
from typing import Any, Final

import anyio

from researcharena_inspect.environment.evaluation import (
    SNAPSHOT_DIR,
    evaluator_sandbox,
    stop_agent_processes,
)
from researcharena_inspect.environment.layout import EVAL_DIR
from researcharena_inspect.scoring.transfer import copy_out, list_tree

MODEL_SETTINGS: Final = frozenset({"utility_finetuning", "safety_finetuning"})
"""Settings whose submission is a model folder (`final_model/`)."""

MODEL_FOLDER: Final = "final_model"
METADATA_KEY: Final = "submission_store"
STORE_SUFFIX: Final = ".submissions"
MANIFEST: Final = "manifest.json"
MANIFEST_VERSION: Final = 2

STORED: Final = "stored"
PARTIAL: Final = "partial"
ERROR: Final = "error"

DEFAULT_MAX_BYTES: Final = 20 * 1024**3
"""Cap on one run folder's files (20 GiB). Published capabilities workspaces hold at most 452 MB
besides their 988 MB model."""

DEFAULT_MAX_FILES: Final = 100_000

RECORD_MAX_BYTES: Final = 256 * 1024**2
"""Cap on the run's `eval/` and `side_task_evidence/` folders together (logs and the capture log)."""

MANIFEST_IN_LOG: Final = 2000
"""Entries of the manifest kept in the log besides `final_model/`'s; `manifest.json` has all."""

_SCRATCH: Final = "/var/tmp/researcharena-store-piece"

# The run's records beside the snapshot: (folder in the sandbox, folder in the run folder).
RECORD_FOLDERS: Final = (
    (f"{EVAL_DIR}/eval", "eval"),
    (f"{EVAL_DIR}/side_task_evidence", "side_task_evidence"),
)


class StoreError(RuntimeError):
    """The store's folder cannot be chosen or written."""


def default_store_root() -> Path:
    """`<log without .eval>.submissions/` beside the running eval's log.

    Raises:
        StoreError: no running sample, or the log is not a local file.
    """
    # Inspect 0.3.290 has no public accessor for the running sample's log location.
    from inspect_ai.log._samples import (  # inspect-evals-lint: ignore[private_api_imports] -- no public accessor for the log location
        sample_active,
    )

    active = sample_active()
    if active is None:
        raise StoreError("no running sample: pass model_store=<folder>")
    location = active.log_location
    if location.startswith("file://"):
        location = location.removeprefix("file://")
    elif "://" in location:
        raise StoreError(f"the log ({location}) is not on this machine: pass model_store=<folder>")
    log = Path(location)
    return log.with_name(log.name.removesuffix(".eval").removesuffix(".json") + STORE_SUFFIX)


def store_root(option: str, setting: str) -> Path | None:
    """Where this sample's run folder goes, from the task's `model_store` option.

    `auto`: beside the log for `MODEL_SETTINGS`, nowhere for the others. `none`: nowhere. Any
    other value: that folder.
    """
    if option == "none":
        return None
    if option == "auto":
        return default_store_root() if setting in MODEL_SETTINGS else None
    return Path(option)


def run_folder_name(sample_id: str | int, epoch: int, uuid: str | None = None) -> str:
    """`<sample id, sanitised>__e<epoch>`, and `__<first 8 of the sample's uuid>` when given."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    tail = f"__{re.sub(r'[^A-Za-z0-9]', '', uuid)[:8]}" if uuid else ""
    return f"{safe}__e{epoch}{tail}"


def _safe_rel(path: str) -> Path:
    parts = Path(path).parts
    if Path(path).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise StoreError(f"unexpected path in the listing: {path!r}")
    return Path(*parts)


def link_stays_inside(rel: str, target: str) -> bool:
    """Whether a link at `rel` (relative to the tree) to `target` resolves inside the tree.

    Lexical only (the target is never followed): an absolute target, or one whose path climbs
    above the tree's root, is outside.
    """
    if not target or target.startswith("/"):
        return False
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
    return not (joined == ".." or joined.startswith(("../", "/")))


async def keep_submission(
    root: Path,
    *,
    name: str,
    run_metadata: dict[str, Any],
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Copy the sandbox's run folder into `root/name/` (see the module docstring).

    Never raises: any failure is the record's `status: error`, with what was done so far.

    Returns:
        The record for the score's metadata: `status` (`stored`, `partial` or `error`),
        `folder`, `files`, `bytes`, `model` (the `final_model/` entries with size and SHA-256),
        `entries` (the first `MANIFEST_IN_LOG` others), `left_out` (with reasons), `records`
        (the `eval/` and evidence files kept), `manifest_sha256`, `seconds` and `error`.
    """
    start = time.monotonic()
    folder = root / name
    record: dict[str, Any] = {
        "status": ERROR,
        "folder": str(folder),
        "source": SNAPSHOT_DIR,
        "max_bytes": max_bytes,
        "max_files": max_files,
        "files": 0,
        "bytes": 0,
        "model": [],
        "entries": [],
        "left_out": [],
        "records": [],
        "manifest_sha256": None,
        "error": None,
    }
    try:
        await _keep(root, name, run_metadata, max_bytes, max_files, record)
    except Exception as error:  # the score is decided; a store failure must not change it
        record["status"] = ERROR
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["seconds"] = round(time.monotonic() - start, 2)
    return record


async def _copy_tree(
    entries: list[dict[str, Any]],
    source_root: str,
    dest_root: Path,
    part_dir: Path,
    record: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy listed entries under `dest_root`; return (copied, left out with reasons)."""
    box = evaluator_sandbox()
    copied: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    for entry in entries:
        rel = _safe_rel(entry["path"])
        target = dest_root / rel
        kind = entry["kind"]
        if kind == "dir":
            target.mkdir(exist_ok=True)
        elif kind == "link":
            if not link_stays_inside(entry["path"], entry.get("target", "")):
                left_out.append({**entry, "reason": "a link that points outside the folder"})
                continue
            os.symlink(entry["target"], target)
        elif kind == "file":
            if "sha256" not in entry:
                left_out.append({**entry, "reason": "past the store's byte cap"})
                continue
            await copy_out(
                box,
                f"{source_root}/{entry['path']}",
                target,
                size=entry["size"],
                sha256=entry["sha256"],
                scratch=_SCRATCH,
                part_dir=part_dir,
            )
            target.chmod(entry["mode"] & 0o777)
            record["files"] += 1
            record["bytes"] += entry["size"]
        else:
            left_out.append({**entry, "reason": f"a {kind}, not copied"})
            continue
        copied.append(entry)
    return copied, left_out


async def _keep(
    root: Path,
    name: str,
    run_metadata: dict[str, Any],
    max_bytes: int,
    max_files: int,
    record: dict[str, Any],
) -> None:
    final = root / name
    work = root / f".{name}.incomplete"
    if await anyio.Path(final).exists() or await anyio.Path(work).exists():
        raise StoreError(f"{final} (or its .incomplete folder) exists; it is never overwritten")
    # Whatever the evaluator left running could still write into the folder.
    await stop_agent_processes(evaluator_sandbox())
    box = evaluator_sandbox()
    entries, truncated = await list_tree(
        box, SNAPSHOT_DIR, max_entries=max_files, max_hash_bytes=max_bytes, first=(MODEL_FOLDER,)
    )
    snapshot = work / "red_team" / "task_snapshot"
    snapshot.mkdir(parents=True)
    record["incomplete_folder"] = str(work)
    part_dir = work / ".transfer"
    (work / "metadata.json").write_text(json.dumps(run_metadata, indent=4) + "\n", "utf-8")
    copied, left_out = await _copy_tree(entries, SNAPSHOT_DIR, snapshot, part_dir, record)

    # The run's raw records, as upstream's run folder keeps them beside red_team/.
    records: list[dict[str, Any]] = []
    for source, dest in RECORD_FOLDERS:
        present = await box.exec(["test", "-d", source], user="root")
        if not present.success:
            continue
        found, cut = await list_tree(
            box, source, max_entries=10_000, max_hash_bytes=RECORD_MAX_BYTES
        )
        (work / dest).mkdir()
        kept, missed = await _copy_tree(found, source, work / dest, part_dir, record)
        records += [{**e, "path": f"{dest}/{e['path']}"} for e in kept]
        left_out += [{**e, "path": f"{dest}/{e['path']}"} for e in missed]
        if cut:
            left_out.append({"path": dest, "kind": "dir", "reason": "more than 10,000 entries"})
    await box.exec(["rm", "-f", _SCRATCH], user="root")
    if part_dir.exists():
        part_dir.rmdir()

    status = PARTIAL if (left_out or truncated) else STORED
    manifest = {
        "version": MANIFEST_VERSION,
        "status": status,
        "source": SNAPSHOT_DIR,
        "truncated_listing": truncated,
        "entries": copied,
        "records": records,
        "left_out": left_out,
    }
    text = json.dumps(manifest, indent=1) + "\n"
    (work / MANIFEST).write_text(text, "utf-8")
    work.rename(final)
    record.pop("incomplete_folder", None)
    record["manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    model_prefix = f"{MODEL_FOLDER}/"
    record["model"] = [
        e for e in copied if e["path"] == MODEL_FOLDER or e["path"].startswith(model_prefix)
    ]
    others = [e for e in copied if e not in record["model"]]
    record["entries"] = others[:MANIFEST_IN_LOG]
    record["entries_total"] = len(others)
    record["records"] = records
    record["left_out"] = left_out[:200]
    record["left_out_total"] = len(left_out)
    record["truncated_listing"] = truncated
    record["status"] = status


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_FILES",
    "ERROR",
    "MANIFEST",
    "METADATA_KEY",
    "MODEL_FOLDER",
    "MODEL_SETTINGS",
    "PARTIAL",
    "STORED",
    "StoreError",
    "default_store_root",
    "keep_submission",
    "link_stays_inside",
    "run_folder_name",
    "store_root",
]
