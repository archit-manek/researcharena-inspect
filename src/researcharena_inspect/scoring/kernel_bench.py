"""Benchmark a chosen subset of the workspace's kernels, in the sandbox (part D's diagnostic).

This file is copied into the sandbox and run there with the image's Python; it is not imported
by the package. It uses upstream's own code from a copy of the agent's `task/` folder, unchanged:
`config.ArenaConfig`, `bench.load_model_and_tokenizer`, `bench.benchmark_generation`,
`bench.apply_seed` and `inject.inject_agent_kernels`. It builds the extension once with the
arguments `bench.compile_kernels` uses (the same name, sources and flags) into a fixed folder, so
later runs load it without rebuilding.

    python3 kernel_bench.py build --task DIR --build-dir DIR
    python3 kernel_bench.py bench --task DIR --build-dir DIR --kernels rmsnorm,attention
        [--sync-each] [--vanilla-first] [--prompts N] [--max-new-tokens N] [--warmup N]

`bench` injects only the named kernels (`all`, `none` or a comma list): `inject_agent_kernels`
asks `hasattr(agent_ops, name)` for each kernel, and the module is wrapped so that only the named
ones are visible. With `--sync-each`, every kernel call is followed by
`torch.cuda.synchronize()` and the first tensor shapes of each kernel are recorded, so a fault is
attributed to the kernel call that caused it. The last line is `KERNEL_BENCH_RESULT=<json>`.
"""

# Adapted from https://github.com/aisa-group/ResearchArena at b922f4b (Copyright (c) 2026 AI Safety and Alignment Group, MIT licence)

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from typing import Any

KERNELS = ("rmsnorm", "rotary_embedding", "fused_mlp", "attention")
RESULT_PREFIX = "KERNEL_BENCH_RESULT="
MAX_SHAPES = 8


def parse_kernels(text: str) -> list[str]:
    """`all`, `none` or a comma list of kernel names, in `KERNELS` order."""
    if text == "all":
        return list(KERNELS)
    if text == "none":
        return []
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = sorted(set(names) - set(KERNELS))
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}")
    return [k for k in KERNELS if k in names]


def _describe(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"shape": list(shape), "dtype": str(getattr(value, "dtype", ""))}
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return type(value).__name__


class OnlyKernels:
    """The compiled module with only `kernels` visible (and, optionally, each call checked)."""

    def __init__(
        self, ops: Any, kernels: list[str], sync_each: bool = False, synchronize: Any = None
    ) -> None:
        self._ops = ops
        self._kernels = list(kernels)
        self._sync_each = sync_each
        self._synchronize = synchronize
        self.calls: dict[str, int] = dict.fromkeys(kernels, 0)
        self.shapes: dict[str, list[Any]] = {k: [] for k in kernels}
        self.fault: dict[str, Any] | None = None

    def __dir__(self) -> list[str]:
        return list(self._kernels)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name not in self._kernels:
            raise AttributeError(name)
        function = getattr(self._ops, name)
        if not self._sync_each:
            return function

        def checked(*args: Any) -> Any:
            self.calls[name] += 1
            described = [_describe(a) for a in args]
            if described not in self.shapes[name] and len(self.shapes[name]) < MAX_SHAPES:
                self.shapes[name].append(described)
            try:
                out = function(*args)
                if self._synchronize is not None:
                    self._synchronize()
                return out
            except Exception as error:
                if self.fault is None:
                    self.fault = {
                        "kernel": name,
                        "call": self.calls[name],
                        "args": described,
                        "error": f"{type(error).__name__}: {error}",
                    }
                raise

        return checked


def _load_extension(task: str, build_dir: str) -> Any:
    from torch.utils.cpp_extension import load

    # bench.compile_kernels's arguments (bench.py:121-146), with a fixed build folder.
    sources = [os.path.join(task, "bindings.cpp")]
    sources.extend(sorted(glob.glob(os.path.join(task, "kernels", "*.cu"))))
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name="agent_ops",
        sources=sources,
        build_directory=build_dir,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def _versions() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception as error:
        out["transformers"] = f"unavailable: {error}"
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    try:
        ops = _load_extension(args.task, args.build_dir)
    except Exception as error:
        return {"compile_status": "FAIL", "error": str(error), "seconds": time.time() - start}
    functions = [f for f in dir(ops) if not f.startswith("_")]
    print(f"COMPILE_STATUS=PASS\nCOMPILE_FUNCTIONS={','.join(functions)}", flush=True)
    return {"compile_status": "PASS", "functions": functions, "seconds": time.time() - start}


def bench(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, args.task)
    import gc

    import torch
    from bench import apply_seed, benchmark_generation, load_model_and_tokenizer
    from config import ArenaConfig
    from inject import inject_agent_kernels

    kernels = parse_kernels(args.kernels)
    config = ArenaConfig()
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.warmup is not None:
        config.warmup_runs = args.warmup
    result: dict[str, Any] = {
        "kernels": kernels,
        "sync_each": args.sync_each,
        "vanilla_first": args.vanilla_first,
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "max_new_tokens": config.max_new_tokens,
        "warmup_runs": config.warmup_runs,
        "prompts": args.prompts or len(config.eval_prompts),
        **_versions(),
    }
    apply_seed(config.seed)
    ops = _load_extension(args.task, args.build_dir)
    if args.vanilla_first:
        # The order run_benchmark uses: a vanilla model first, deleted, then the agent's.
        model, tokenizer = load_model_and_tokenizer(config)
        result["vanilla"] = benchmark_generation(model, tokenizer, config, args.prompts)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    visible = OnlyKernels(ops, kernels, args.sync_each, torch.cuda.synchronize)
    model, tokenizer = load_model_and_tokenizer(config)
    result["injected"] = inject_agent_kernels(model, visible, verbose=True)
    try:
        result["benchmark"] = benchmark_generation(model, tokenizer, config, args.prompts)
        result["status"] = "ok"
    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result["fault"] = visible.fault
    if args.sync_each:
        result["calls"] = visible.calls
        result["shapes"] = visible.shapes
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "bench"])
    parser.add_argument("--task", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--kernels", default="all")
    parser.add_argument("--sync-each", action="store_true")
    parser.add_argument("--vanilla-first", action="store_true")
    parser.add_argument("--prompts", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        result = build(args) if args.action == "build" else bench(args)
    except Exception as error:
        result = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    failed = result.get("status") == "error" or result.get("compile_status") == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
