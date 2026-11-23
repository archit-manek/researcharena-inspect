"""A post-training run's folder, model included, copied to the host beside the log (D-172).

The submitted `final_model/` (about 1 GB) is too large for the `.eval` log and is lost with the
sandbox. For `MODEL_SETTINGS` (any setting if `model_store` names a folder), once the score is
decided, the scorer copies the run folder in upstream's layout to
`<store>/<sample>__e<epoch>__<uuid8>/`: `metadata.json`, `red_team/task_snapshot/`
(`src/pipeline_redteam.sh:433-438`), `eval/`, `side_task_evidence/` and `manifest.json`. By
default `<store>` is `<log without .eval>.submissions/`. `eval/` and `side_task_evidence/` let a
replay of an `independent` run put its capture log back.

- Everything is written under `.<name>.incomplete/` and renamed once every listed file has
  arrived with its SHA-256; replay refuses an `.incomplete` folder. Nothing is overwritten.
- `final_model/` is listed and hashed first, so the caps cut the rest of the workspace first.
- A link pointing outside `task_snapshot/`, and FIFOs, sockets and devices, are not made; they
  are listed in `left_out`.
- Status: `stored`, `partial` (something left out) or `error`. Replay accepts only `stored`.

A failure is recorded in the score's metadata and never raised.
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
    """Where this sample's run folder goes, from the `model_store` option.

    `auto`: beside the log for `MODEL_SETTINGS`, nowhere for the others; `none`; or a folder.
    """
    if option == "none":
        return None
    if option == "auto":
        return default_store_root() if setting in MODEL_SETTINGS else None
    return Path(option)


def run_folder_name(sample_id: str | int, epoch: int, uuid: str | None = None) -> str:
    """`<sample id, sanitised>__e<epoch>__<uuid8>`.

    The uuid keeps apart retries and sample ids that sanitise to the same name.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    tail = f"__{re.sub(r'[^A-Za-z0-9]', '', uuid)[:8]}" if uuid else ""
    return f"{safe}__e{epoch}{tail}"


def _safe_rel(path: str) -> Path:
    parts = Path(path).parts
    if Path(path).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise StoreError(f"unexpected path in the listing: {path!r}")
    return Path(*parts)


def link_stays_inside(rel: str, target: str) -> bool:
    """Whether a link at `rel` to `target` stays inside the tree, lexically (never followed)."""
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
    """Copy the sandbox's run folder into `root/name/`. Never raises: a failure is `error`.

    Returns:
        The record for the score's metadata: `status`, `folder`, totals, `model` (the
        `final_model/` entries with SHA-256), the first `MANIFEST_IN_LOG` other `entries`,
        `left_out` with reasons, `records` (`eval/` and evidence), `manifest_sha256` and `error`.
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
    # The evaluator's leftovers could still write into the folder.
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

    # `eval/` and `side_task_evidence/`, beside `red_team/` as upstream keeps them.
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
