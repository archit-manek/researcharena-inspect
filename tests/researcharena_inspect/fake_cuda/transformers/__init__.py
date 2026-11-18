"""`AutoTokenizer` and `AutoModelForCausalLM` as upstream's `bench.py` uses them, on a CPU."""

from __future__ import annotations

from typing import Any

import torch


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Inputs(dict):
    def to(self, device: str) -> _Inputs:
        return self


class _Tokenizer:
    pad_token = None
    eos_token = "</s>"
    vocab_size = 32000

    def __call__(self, text: str, return_tensors: str = "pt") -> _Inputs:
        return _Inputs(input_ids=_Tensor((1, len(text.split()) + 1)))


class _Parameter:
    def numel(self) -> int:
        return 1_100_000_000


class _Model:
    def __init__(self) -> None:
        self.injected = False

    def eval(self) -> _Model:
        return self

    def parameters(self) -> list[_Parameter]:
        return [_Parameter()]

    def generate(self, input_ids: _Tensor, max_new_tokens: int, **kwargs: Any) -> _Tensor:
        injected = self.injected
        torch.event("generate", injected=injected, max_new_tokens=max_new_tokens)
        if not injected:
            if torch.FAULT == "baseline":
                raise RuntimeError(torch.FAULT_TEXT)
            torch.STATE["baseline_pass_ran"] = True
        elif torch.FAULT == "always" or (
            torch.FAULT == "after_baseline" and torch.STATE["baseline_pass_ran"]
        ):
            raise RuntimeError(torch.FAULT_TEXT)
        torch.advance(torch.SECONDS_PER_TOKEN[injected] * max_new_tokens)
        return _Tensor((1, input_ids.shape[1] + max_new_tokens))


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> _Tokenizer:
        torch.event("load_tokenizer", model_id=model_id)
        return _Tokenizer()


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model_id: str, torch_dtype: Any = None, **kwargs: Any) -> _Model:
        torch.event("load_model", model_id=model_id, dtype=torch_dtype)
        return _Model()
