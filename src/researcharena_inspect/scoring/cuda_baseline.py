"""Upstream's CUDA baseline measurement, alone, in a process of its own (D-160).

Copied into the sandbox and run with the image's Python before `evaluate.py`, as the evaluator's
user; not imported by the package. With the workspace's own `bench` and `config`
(`evaluate.py:57-60`), it does what `bench.run_benchmark` does when there is no cached baseline
and no `starter_kernels/` (`bench.py:160-164`, `:189-210`, `:238-242`), always the case in the
port (D-002): it times plain PyTorch and writes `baseline_tps.txt` as `bench.py` does, after
removing any file already there. `bench.py` then reads that file as cached (`:192-194`) and skips
its own baseline pass. If `starter_kernels/` exists, this script fails rather than reproduce that
branch, and the scorer falls back to upstream's in-process baseline.

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
    """Measure and cache the baseline; return the record printed on the result line."""
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
