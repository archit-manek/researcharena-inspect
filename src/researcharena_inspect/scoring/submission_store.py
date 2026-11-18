"""The submitted model, kept on the host beside the log, so that the run can be re-scored later.

A post-training run submits `final_model/`, about 1 GB, inside the workspace upstream keeps as
`red_team/task_snapshot/` (`src/pipeline_redteam.sh:433-438`, pinned commit; the authors
published those models in a separate dataset, `pins.OUTPUT_MODELS_DATASET`). The `.eval` log
cannot hold it (D-110 keeps a workspace only up to `snapshot_max_mb`, 20 MiB by default), and it
is lost with the sandbox. So, for the settings in `MODEL_SETTINGS`, after the score is decided the
scorer copies the post-evaluation `task_snapshot/` out of the sandbox, `final_model/` first, into
a run folder in upstream's layout on the host (D-172):

    <store>/<sample id>__e<epoch>/metadata.json             the run's setting, side task, config
    <store>/<sample id>__e<epoch>/red_team/task_snapshot/   the folder upstream's run keeps
    <store>/<sample id>__e<epoch>/manifest.json             every entry: path, kind, size, SHA-256

Symbolic links are listed in the manifest with their target and not made on the host.

`<store>` is, by default (`model_store="auto"`), `<log file without .eval>.submissions/` beside
the log; another folder can be given. The run folder is what `agents.replay` takes as a local run
(`-T agent=replay -T replay=<run folder>`), so a stored run can be re-scored, and what the monitor's
local bundles read. The score's metadata (`submission_store`) carries the folder, the totals and
the manifest of `final_model/` (and the first `MANIFEST_IN_LOG` entries of the rest).

Every copy is bounded (`transfer`: pieces of 64 MiB, a time limit per exec, SHA-256 checked end
to end), the store's total is capped (`max_bytes`, `max_files`; `final_model/` is copied first, so a
cap cuts the rest of the workspace, not the model, unless the model alone is too large), and an
existing folder is never overwritten. A failure is recorded in the metadata (`status: error`) and
never raised: the score is already decided and is not changed by it.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Final

import anyio
from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.evaluation import SNAPSHOT_DIR, stop_agent_processes
from researcharena_inspect.scoring.transfer import copy_out, list_tree

MODEL_SETTINGS: Final = frozenset({"utility_finetuning", "safety_finetuning"})
"""Settings whose submission is a model folder (`final_model/`)."""

MODEL_FOLDER: Final = "final_model"
METADATA_KEY: Final = "submission_store"
STORE_SUFFIX: Final = ".submissions"

DEFAULT_MAX_BYTES: Final = 20 * 1024**3
"""Cap on one run folder's files (20 GiB). Published capabilities workspaces hold at most 452 MB
besides their 988 MB model."""

DEFAULT_MAX_FILES: Final = 100_000

MANIFEST_IN_LOG: Final = 2000
"""Entries of the manifest kept in the log besides `final_model/`'s; `manifest.json` has all."""

_SCRATCH: Final = "/var/tmp/researcharena-store-piece"


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


def run_folder_name(sample_id: str | int, epoch: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    return f"{safe}__e{epoch}"


def _safe_rel(path: str) -> Path:
    parts = Path(path).parts
    if Path(path).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise StoreError(f"unexpected path in the listing: {path!r}")
    return Path(*parts)


async def keep_submission(
    root: Path,
    *,
    name: str,
    run_metadata: dict[str, Any],
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_files: int = DEFAULT_MAX_FILES,
) -> dict[str, Any]:
    """Copy the sandbox's `task_snapshot/` into `root/name/` (see the module docstring).

    Never raises: any failure is the record's `status: error`, with the files copied so far.

    Returns:
        The record for the score's metadata: `status` (`stored`, `partial` or `error`),
        `folder`, `files`, `bytes`, `model` (the `final_model/` entries with size and SHA-256),
        `entries` (the first `MANIFEST_IN_LOG` others), `left_out` (paths not copied because of
        a cap), `manifest_sha256`, `seconds` and `error`.
    """
    start = time.monotonic()
    folder = root / name
    record: dict[str, Any] = {
        "status": "error",
        "folder": str(folder),
        "source": SNAPSHOT_DIR,
        "max_bytes": max_bytes,
        "max_files": max_files,
        "files": 0,
        "bytes": 0,
        "model": [],
        "entries": [],
        "left_out": [],
        "manifest_sha256": None,
        "error": None,
    }
    try:
        await _keep(folder, run_metadata, max_bytes, max_files, record)
    except Exception as error:  # the score is decided; a store failure must not change it
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["seconds"] = round(time.monotonic() - start, 2)
    return record


async def _keep(
    folder: Path,
    run_metadata: dict[str, Any],
    max_bytes: int,
    max_files: int,
    record: dict[str, Any],
) -> None:
    if await anyio.Path(folder).exists():
        raise StoreError(f"{folder} exists; it is never overwritten")
    # Whatever the evaluator left running could still write into the folder.
    await stop_agent_processes()
    box = sandbox_env()
    entries, truncated = await list_tree(
        box, SNAPSHOT_DIR, max_entries=max_files, first=(MODEL_FOLDER,)
    )
    snapshot = folder / "red_team" / "task_snapshot"
    snapshot.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps(run_metadata, indent=4) + "\n", "utf-8")
    total = 0
    copied: list[dict[str, Any]] = []
    left_out: list[str] = []
    for entry in entries:
        rel = _safe_rel(entry["path"])
        target = snapshot / rel
        if entry["kind"] == "dir":
            target.mkdir(exist_ok=True)
        elif entry["kind"] == "link":
            # Listed with its target in the manifest, not made: a link on the host could point
            # anywhere there, and a replay that read it would carry host files into a sandbox.
            pass
        elif entry["kind"] == "file":
            if total + entry["size"] > max_bytes:
                left_out.append(entry["path"])
                continue
            await copy_out(
                box,
                f"{SNAPSHOT_DIR}/{entry['path']}",
                target,
                size=entry["size"],
                sha256=entry["sha256"],
                scratch=_SCRATCH,
            )
            target.chmod(entry["mode"] & 0o777)
            total += entry["size"]
            record["files"] += 1
            record["bytes"] = total
        else:
            left_out.append(entry["path"])
            continue
        copied.append(entry)
    await box.exec(["rm", "-f", _SCRATCH], user="root")
    manifest = {
        "source": SNAPSHOT_DIR,
        "truncated_listing": truncated,
        "entries": copied,
        "left_out": left_out,
    }
    text = json.dumps(manifest, indent=1) + "\n"
    (folder / "manifest.json").write_text(text, "utf-8")
    record["manifest_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    model_prefix = f"{MODEL_FOLDER}/"
    record["model"] = [
        e for e in copied if e["path"] == MODEL_FOLDER or e["path"].startswith(model_prefix)
    ]
    others = [e for e in copied if e not in record["model"]]
    record["entries"] = others[:MANIFEST_IN_LOG]
    record["entries_total"] = len(others)
    record["left_out"] = left_out[:200]
    record["left_out_total"] = len(left_out)
    record["truncated_listing"] = truncated
    record["status"] = "partial" if (left_out or truncated) else "stored"


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_FILES",
    "METADATA_KEY",
    "MODEL_FOLDER",
    "MODEL_SETTINGS",
    "StoreError",
    "default_store_root",
    "keep_submission",
    "run_folder_name",
    "store_root",
]
