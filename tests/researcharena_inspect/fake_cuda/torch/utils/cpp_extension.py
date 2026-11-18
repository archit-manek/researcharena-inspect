"""`load` of `torch.utils.cpp_extension`: "compiles" by naming the kernels it finds."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from torch import event

KERNELS = ("rmsnorm", "rotary_embedding", "fused_mlp", "attention")


def load(name: str, sources: list[str], **kwargs: Any) -> Any:
    found = sorted(os.path.basename(s)[:-3] for s in sources if s.endswith(".cu"))
    event("compile", name=name, kernels=found)
    return SimpleNamespace(**{k: (lambda *a, **k: None) for k in KERNELS if k in found})
