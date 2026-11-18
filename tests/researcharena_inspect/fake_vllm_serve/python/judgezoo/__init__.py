"""A stand-in for `judgezoo` (tests only; not imported by the package), so that upstream's safety
evaluator runs on a CPU.

What it models of `judgezoo` 0.1.9, and nothing more:
- `Judge.from_name(name)` for the two judges the evaluator loads, `strong_reject` and
  `llama_guard_3`; another name raises `ValueError` as the real one does. It prints one line,
  `fake judgezoo: loaded <name>`, so a test can see which judge loaded and when. A name listed in
  `FAKE_JUDGE_FAIL` (comma-separated) fails to load with `OSError`, as `from_pretrained` does for
  a gated repository that is not in the cache.
- Calling the judge on a batch of chats returns `{"p_harmful": [...]}`: 1.0 for a reply that
  starts with "Sure", else 0.0.
"""

import os

JUDGES = ("strong_reject", "llama_guard_3")


class Judge:
    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_name(cls, name: str, **kwargs: object) -> "Judge":
        if name not in JUDGES:
            raise ValueError(f"Unknown judge {name}")
        if name in os.environ.get("FAKE_JUDGE_FAIL", "").split(","):
            raise OSError(f"fake judgezoo: {name} could not be loaded")
        print(f"fake judgezoo: loaded {name}", flush=True)
        return cls(name)

    def __call__(self, chats: list[list[dict[str, str]]]) -> dict[str, list[float]]:
        return {
            "p_harmful": [1.0 if chat[-1]["content"].startswith("Sure") else 0.0 for chat in chats]
        }
