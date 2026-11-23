"""The inference evaluator's `artifacts/` folder, kept in the score's metadata (D-138).

Upstream writes it beside the submission
(`inference_optimization/evaluation_code/common.py:33-45`) and keeps it in the run folder; it
holds what `metrics.json` leaves out (timings, quality examples, server logs). A long log keeps
both ends, so a server's start-up and last lines survive. Other evaluators write no such folder,
and then nothing is added.
"""

import shlex
from typing import Any, Final

from researcharena_inspect.environment.evaluation import evaluator_sandbox
from researcharena_inspect.environment.layout import EVAL_DIR

ARTIFACTS_DIR: Final = f"{EVAL_DIR}/artifacts"
METADATA_KEY: Final = "evaluation_artifacts"
JSON_LIMIT_BYTES: Final = 512 * 1024
LOG_END_BYTES: Final = 32 * 1024

READ_TIMEOUT_SECONDS: Final = 60
"""Bound on reading one file in the sandbox."""

# Reads only what is kept (seeking for a log's tail), so a multi-GB server log costs 64 KiB.
_READ = """
import os, sys
path, n, end = sys.argv[1], int(sys.argv[2]), sys.argv[3]
size = os.path.getsize(path)
with open(path, "rb") as f:
    if end == "head":
        data = f.read(n)
    elif size <= 2 * n:
        data = f.read(2 * n)
    else:
        head = f.read(n)
        f.seek(size - n)
        data = head + b"\\n[... %d bytes not kept ...]\\n" % (size - 2 * n) + f.read(n)
sys.stdout.write(data.decode("utf-8", "replace"))
"""


async def read_artifacts(folder: str = ARTIFACTS_DIR) -> dict[str, Any] | None:
    """Every file directly in `folder`, as kept text; None if there is no such folder.

    Never raises: it runs after the score is decided, so a failure is recorded under `error`.

    Returns:
        `{"folder": ..., "files": {name: {"bytes", "kept", "truncated", "text"}}}`, plus
        `error` if listing or reading failed.
    """
    files: dict[str, Any] = {}
    try:
        return await _read(folder, files)
    except Exception as error:  # an exec or sandbox failure must not lose the score
        return {"folder": folder, "error": f"{type(error).__name__}: {error}", "files": files}


async def _read(folder: str, files: dict[str, Any]) -> dict[str, Any] | None:
    box = evaluator_sandbox()
    q = shlex.quote(folder)
    listing = await box.exec(
        ["bash", "-c", f"[ -d {q} ] || exit 3; find {q} -maxdepth 1 -type f -printf '%s %f\\n'"],
        user="root",
    )
    if listing.returncode == 3:
        return None
    if not listing.success:
        return {"folder": folder, "error": listing.stderr, "files": files}
    for line in sorted(listing.stdout.splitlines()):
        size_text, _, name = line.partition(" ")
        if not name or not size_text.isdigit():
            continue
        size = int(size_text)
        is_json = name.endswith(".json")
        limit = JSON_LIMIT_BYTES if is_json else 2 * LOG_END_BYTES
        end_bytes = JSON_LIMIT_BYTES if is_json else LOG_END_BYTES
        try:
            read = await box.exec(
                [
                    "python3",
                    "-c",
                    _READ,
                    f"{folder}/{name}",
                    str(end_bytes),
                    "head" if is_json else "ends",
                ],
                user="root",
                timeout=READ_TIMEOUT_SECONDS,
            )
            text = read.stdout if read.success else f"[not read: {read.stderr.strip()}]"
        except Exception as error:  # this file only: the others are still read
            text = f"[not read: {type(error).__name__}: {error}]"
        files[name] = {
            "bytes": size,
            "kept": min(size, limit),
            "truncated": size > limit,
            "text": text,
        }
    return {"folder": folder, "files": files}


__all__ = ["ARTIFACTS_DIR", "METADATA_KEY", "read_artifacts"]
