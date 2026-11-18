"""Upstream's CUDA baseline measurement, alone, in a process of its own (D-160).

This file is copied into the sandbox and run there with the image's Python; it is not imported
by the package. `red_team_scorer(cuda_baseline_process=True)` runs it once, as the evaluator's
user, in the evaluator's folder and environment, before upstream's `evaluate.py`:

    python3 cuda_baseline.py --task-dir /mnt/eval/red_team/task_snapshot

It imports the workspace's own `bench` and `config` from `--task-dir`, as `evaluate.py` does
(`evaluate.py:57-60` at the pin), and does what `bench.run_benchmark` does when no cached
baseline exists and `starter_kernels/` is not found (`bench.py:160-164`, `:189-210`,
`:238-242`), which is always the case in the port (D-002):

1. `apply_seed(config.seed)` and `os.makedirs(config.workspace_dir)`.
2. Removes `<workspace_dir>/baseline_tps.txt` if it exists (keeping its text for the record):
   under the option the baseline is always this measurement, never a file already in the
   workspace.
3. Loads the model with `bench.load_model_and_tokenizer(config)` and times plain PyTorch with
   `bench.benchmark_generation(model, tokenizer, config, None)`: no kernel is compiled,
   loaded or injected.
4. Writes `f"{tps:.4f}"` and a newline to `<workspace_dir>/baseline_tps.txt`, as
   `bench.py:240-241` does.

`evaluate.py` then finds that file and `bench.py` skips its own baseline pass
(`bench.py:192-194`, "(cached)"), so the agent's kernels are benchmarked in a process where no
baseline pass has run. If `starter_kernels/` were found, upstream would compile and inject the
starter kernels for its baseline; this script does not reproduce that branch and fails instead,
so the scorer falls back to upstream's own in-process baseline.

The last line printed is `CUDA_BASELINE_RESULT=<json>`; the exit status is 0 only if the file
was written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any

RESULT_PREFIX = "CUDA_BASELINE_RESULT="
CACHE_NAME = "baseline_tps.txt"
TEXT_LIMIT = 4096


def _read_text(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read(TEXT_LIMIT).decode("utf-8", errors="replace")
    except OSError as e:
        return f"<unreadable: {e}>"


def _remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def measure(task_dir: str) -> dict[str, Any]:
    """Steps 1-4 above; returns the record printed on the result line."""
    record: dict[str, Any] = {"status": "error", "task_dir": task_dir}
    sys.path.insert(0, task_dir)
    import bench  # pyright: ignore[reportMissingImports] (the workspace's own)
    from config import ArenaConfig  # pyright: ignore[reportMissingImports]

    config = ArenaConfig()
    bench.apply_seed(config.seed)
    os.makedirs(config.workspace_dir, exist_ok=True)
    cache = os.path.join(config.workspace_dir, CACHE_NAME)
    record["cache_path"] = cache
    record["removed_before"] = None
    if os.path.lexists(cache):
        record["removed_before"] = _read_text(cache)
        _remove(cache)

    arena_root = os.environ.get("ARENA_ROOT", os.path.join(bench.ROOT, ".."))
    starter_dir = os.path.join(arena_root, "starter_kernels")
    if os.path.isdir(starter_dir):
        raise RuntimeError(
            f"starter_kernels found at {starter_dir}: upstream would benchmark the starter "
            "kernels, which this script does not reproduce"
        )
    print(f"WARNING: starter_kernels not found at {starter_dir}")
    print("Using vanilla PyTorch as baseline (no kernel injection)")
    model, tokenizer = bench.load_model_and_tokenizer(config)
    results = bench.benchmark_generation(model, tokenizer, config, None)
    tps = results["avg_tokens_per_sec"]
    print(f"BASELINE_TPS={tps:.2f}")
    with open(cache, "w") as f:
        f.write(f"{tps:.4f}\n")
    print(f"Baseline TPS cached to {cache}")
    record.update(
        status="measured",
        baseline_tps=tps,
        cache_text=_read_text(cache),
        benchmark={k: v for k, v in results.items() if isinstance(v, (int, float, str))},
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upstream's CUDA baseline, in a process of its own"
    )
    parser.add_argument("--task-dir", required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    try:
        record = measure(args.task_dir)
    except BaseException as e:
        traceback.print_exc()
        record = {"status": "error", "task_dir": args.task_dir, "error": f"{type(e).__name__}: {e}"}
    record["seconds"] = round(time.monotonic() - start, 1)
    print(RESULT_PREFIX + json.dumps(record), flush=True)
    return 0 if record["status"] == "measured" else 1


if __name__ == "__main__":
    sys.exit(main())
