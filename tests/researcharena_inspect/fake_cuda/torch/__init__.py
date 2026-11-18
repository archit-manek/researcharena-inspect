"""A stand-in for the parts of PyTorch that upstream's `bench.py` and `evaluate.py` touch.

Not imported by the package. Tests copy `torch/`, `transformers/` and `inject.py` from this
folder into a workspace (`task_snapshot/`), which `evaluate.py` and `cuda_baseline.py` put first
on `sys.path`, so upstream's real `evaluate.py` and `bench.py` run on a CPU against it.

What it models is the fault the D-160 option is about: a generation with the agent's kernels
injected raises "CUDA error: an illegal memory access was encountered" if a generation without
them (the baseline pass) has already run **in the same process**. A fresh process has no such
history, so the same kernels then run. `fake_cuda.json` in the workspace can change the rule:
`{"fault": "after_baseline" | "always" | "never" | "baseline"}` (`baseline`: the plain-PyTorch
pass itself fails).

Timing is deterministic: `time.perf_counter` is replaced by a clock that only `generate`
advances, by `0.01` s per new token without kernels and `0.005` s with them, so the baseline is
exactly 100 tokens/s and the agent's kernels give a speedup of 2.0. Every call that matters
prints a line `FAKE_CUDA {"event": ..., "pid": ...}`, so a test can tell which process did what.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAULT_TEXT = "CUDA error: an illegal memory access was encountered"
SECONDS_PER_TOKEN = {False: 0.01, True: 0.005}


def _config() -> dict[str, Any]:
    try:
        with open(os.path.join(WORKSPACE, "fake_cuda.json")) as f:
            return json.load(f)
    except OSError:
        return {}


FAULT = _config().get("fault", "after_baseline")
STATE: dict[str, Any] = {"clock": 0.0, "baseline_pass_ran": False}


def _perf_counter() -> float:
    return STATE["clock"]


time.perf_counter = _perf_counter


def event(kind: str, **fields: Any) -> None:
    print("FAKE_CUDA " + json.dumps({"event": kind, "pid": os.getpid(), **fields}), flush=True)


def advance(seconds: float) -> None:
    STATE["clock"] += seconds


float16 = "float16"
bfloat16 = "bfloat16"


def manual_seed(seed: int) -> None:
    event("manual_seed", seed=seed)


class no_grad:  # noqa: N801 (PyTorch's name)
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        event("cuda.manual_seed_all", seed=seed)

    @staticmethod
    def synchronize() -> None:
        return None

    @staticmethod
    def empty_cache() -> None:
        event("cuda.empty_cache")


cuda = _Cuda()
