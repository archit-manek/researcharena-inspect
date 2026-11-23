"""The tasks' epoch reducer: upstream's -1 ("no score") is left out of the mean (D-123).

Upstream writes -1 for a score it does not have (`pipeline_monitor.sh:474-475`, and the failure
record of `pipeline_redteam.sh:526-530`) and for a monitor without a usable judgement. Inspect's
`mean` would average it in as a number (2.0 and a failed epoch give 0.5). Per field:

- the mean of the values other than -1; -1 if only -1 is present, None if only None;
- a boolean field (`evaluation_failed`) is the share of epochs where it was true;
- an unscored epoch (NaN) is skipped, as `mean` skips it.
"""

import math
import statistics
from typing import Any

from inspect_ai.scorer import Score, ScoreReducer, Value, score_reducer

NO_SCORE: float = -1.0
"""Upstream's value for "no score"."""


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def reduce_field(values: list[Any]) -> Any:
    """One field (or a scalar score) over the epochs."""
    present = [v for v in values if v is not None and not _is_nan(v)]
    if not present:
        return None if any(v is None for v in values) else float("nan")
    if all(isinstance(v, bool) for v in present):
        return statistics.mean(float(v) for v in present)
    numbers = [
        float(v)
        for v in present
        if isinstance(v, int | float) and not isinstance(v, bool) and v != NO_SCORE
    ]
    if numbers:
        return statistics.mean(numbers)
    if any(isinstance(v, int | float) and v == NO_SCORE for v in present):
        return NO_SCORE
    raise ValueError(f"cannot reduce non-numeric score values over epochs: {present!r}")


def _same(items: list[Any]) -> Any:
    return items[0] if all(item == items[0] for item in items) else None


@score_reducer
def scored_epochs_mean() -> ScoreReducer:
    """Mean over the epochs that have a score; -1 ("no score") only if none has one."""

    def reduce(scores: list[Score]) -> Score:
        scored = [s for s in scores if not _is_nan(s.value)]
        if not scored:
            return Score(value=float("nan"))
        first = scored[0].value
        if isinstance(first, dict):
            dicts = [s.value for s in scored if isinstance(s.value, dict)]
            if len(dicts) != len(scored) or any(d.keys() != first.keys() for d in dicts):
                raise ValueError("epochs of one sample have differing score fields")
            value: Value = {key: reduce_field([d[key] for d in dicts]) for key in first}
        else:
            value = reduce_field([s.value for s in scored])
        metadata = dict(scored[0].metadata or {})
        if len(scored) > 1:
            metadata["epoch_values"] = [s.value for s in scored]
        return Score(
            value=value,
            answer=_same([s.answer for s in scored]),
            explanation=_same([s.explanation for s in scored]),
            metadata=metadata,
        )

    return reduce


__all__ = ["NO_SCORE", "reduce_field", "scored_epochs_mean"]
