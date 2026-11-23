"""The plain row every analysis function takes: one red-team run under one monitor configuration.

Run-level fields repeat on every row of a run. A run no monitor scored has one row, with
`monitor_model` and `access` None. Scores are in the paper's units (`published_scores` converts
published data).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

Access = Literal["traj", "cot", "artifact", "artifact_cot"]
SideTaskClass = Literal["baseline", "embedded", "independent"]

# The paper's four monitor types, in its column order (M_traj, M_CoT, M_artifact, M_artifact+CoT).
ACCESS_TYPES: tuple[Access, ...] = ("traj", "cot", "artifact", "artifact_cot")

SETTINGS = ("safety_finetuning", "utility_finetuning", "cuda_kernel", "inference_optimization")
OPTIMIZATION_SETTINGS = ("cuda_kernel", "inference_optimization")
POST_TRAINING_SETTINGS = ("safety_finetuning", "utility_finetuning")

# Red-team agents by model, provider prefix removed ("opencode/glm-5.1" -> "glm-5.1").
SEVEN_AGENTS = (
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "gpt-5.4",
    "gpt-5.5",
    "gemini-3.1-pro",
    "glm-5.1",
)
# GPT-5.3-codex ran on part of the grid only; the paper reports it in Appendix C (L501-503).
EXTRA_AGENT = "gpt-5.3-codex"
ALL_AGENTS = (*SEVEN_AGENTS, EXTRA_AGENT)
COT_AGENTS = tuple(a for a in ALL_AGENTS if a not in ("gemini-3.1-pro", "glm-5.1"))

# Monitor backbones by model name.
POST_TRAINING_MONITORS = ("claude-sonnet-4-6", "gpt-5.4")
OPTIMIZATION_MONITORS = (
    "gpt-5.4",
    "gpt-5.5",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
    "kimi-k2.6",
    "glm-5.1",
)

# Flags the analysis reads. Anything else in `Row.flags` is carried but ignored.
FLAG_UNPROMPTED_SABOTAGE = "unprompted_sabotage"  # a baseline the paper excludes (L497-500)


@dataclass(frozen=True)
class Row:
    """One run under one monitor configuration.

    Attributes:
        run_id: unique per red-team run.
        setting: one of `SETTINGS`.
        side_task: upstream's side-task id ("baseline", "embedded", "embedded2", ...).
        side_task_class: "baseline", "embedded" or "independent", as the paper classes it.
        agent: red-team model, without provider prefix.
        monitor_model: monitor backbone model, or None for a run with no monitor row.
        access: monitor access type, or None with `monitor_model`.
        main_score: main-task score in the paper's units (CUDA: raw speedup).
        side_score: the side task's headline score in the paper's units; None for baselines.
        suspicion: monitor suspicion 0-100; None when the monitor's score is invalid or missing.
        side_metrics: further scorer outputs that success rules read (for example
            `suspicious_code`, `kl_fp16`, `trigger_wrong_rate`).
        flags: run-level labels, for example `FLAG_UNPROMPTED_SABOTAGE`.
    """

    run_id: str
    setting: str
    side_task: str
    side_task_class: SideTaskClass
    agent: str
    monitor_model: str | None
    access: Access | None
    main_score: float | None
    side_score: float | None
    suspicion: float | None
    side_metrics: Mapping[str, float | bool | None] = field(default_factory=dict)
    flags: frozenset[str] = frozenset()


def runs(rows: Iterable[Row]) -> list[Row]:
    """One row per run (the first seen), in first-seen order."""
    seen: dict[str, Row] = {}
    for row in rows:
        seen.setdefault(row.run_id, row)
    return list(seen.values())
