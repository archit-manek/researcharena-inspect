"""The evaluator's `artifacts/` folder, kept in the score's metadata.

`settings/inference_optimization/evaluation_code/common.py:33-45` (pinned commit) writes beside
the submission, in `<run>/artifacts/` (`/mnt/eval/artifacts` here): one JSON file per gate and
side-task probe (`startup_check.json`, `speed_results.json`, ...) and every server's stdout and
stderr (`<name>_stdout.log`, `<name>_stderr.log`). `metrics.json` keeps only part of it: the
baseline's and the submission's timings, the quality examples and why a server did not start
are only here. Upstream's run folder keeps the folder; the port keeps it in the score (JSON
files whole up to `JSON_LIMIT_BYTES`, logs' last `LOG_TAIL_BYTES`). CUDA's evaluator writes no
such folder, and then nothing is added.
"""

import shlex
from typing import Any, Final

from inspect_ai.util import sandbox

from researcharena_inspect.environment.layout import EVAL_DIR

ARTIFACTS_DIR: Final = f"{EVAL_DIR}/artifacts"
METADATA_KEY: Final = "evaluation_artifacts"
JSON_LIMIT_BYTES: Final = 512 * 1024
LOG_TAIL_BYTES: Final = 16 * 1024

_READ = (
    "import sys; path, n, end = sys.argv[1], int(sys.argv[2]), sys.argv[3]; "
    "data = open(path, 'rb').read(); data = data[:n] if end == 'head' else data[-n:]; "
    "sys.stdout.write(data.decode('utf-8', 'replace'))"
)


async def read_artifacts(folder: str = ARTIFACTS_DIR) -> dict[str, Any] | None:
    """Every file directly in `folder`, as kept text; None if there is no such folder.

    Returns:
        `{"folder": ..., "files": {name: {"bytes", "kept", "truncated", "text"}}}`.
    """
    box = sandbox()
    q = shlex.quote(folder)
    listing = await box.exec(
        ["bash", "-c", f"[ -d {q} ] || exit 3; find {q} -maxdepth 1 -type f -printf '%s %f\\n'"],
        user="root",
    )
    if listing.returncode == 3:
        return None
    if not listing.success:
        return {"folder": folder, "error": listing.stderr, "files": {}}
    files: dict[str, Any] = {}
    for line in sorted(listing.stdout.splitlines()):
        size_text, _, name = line.partition(" ")
        if not name or not size_text.isdigit():
            continue
        size = int(size_text)
        is_json = name.endswith(".json")
        limit = JSON_LIMIT_BYTES if is_json else LOG_TAIL_BYTES
        # A JSON file's start, a log's end; decoded leniently (a log can hold any bytes).
        read = await box.exec(
            ["python3", "-c", _READ, f"{folder}/{name}", str(limit), "head" if is_json else "tail"],
            user="root",
        )
        text = read.stdout if read.success else f"[not read: {read.stderr.strip()}]"
        files[name] = {
            "bytes": size,
            "kept": min(size, limit),
            "truncated": size > limit,
            "text": text,
        }
    return {"folder": folder, "files": files}


__all__ = ["ARTIFACTS_DIR", "METADATA_KEY", "read_artifacts"]
