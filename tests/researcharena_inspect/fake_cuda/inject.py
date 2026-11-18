"""A stand-in for upstream's `inject.py`, which walks a real model's layers: marks the fake
model as running with the agent's kernels when the compiled module has any of them."""

from __future__ import annotations

from typing import Any

KERNELS = ("rmsnorm", "rotary_embedding", "fused_mlp", "attention")


def inject_agent_kernels(model: Any, agent_ops: Any, verbose: bool = False) -> dict[str, bool]:
    injected = {k: hasattr(agent_ops, k) for k in KERNELS}
    model.injected = any(injected.values())
    return injected
