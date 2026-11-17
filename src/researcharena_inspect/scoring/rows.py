"""Scored runs as the wave 1 analysis's rows, live or published, read the same way.

`analysis.published_rows.rows_from_index` builds rows for the authors' runs from the website
index and a snapshot of their `raw_metrics`. A port run has no index entry: its row comes from
the score's metadata, where `red_team_scorer` keeps the parsed `metrics.json`. Both go through
`metrics_json.read_metrics`; only the main score's decoding differs (D-004, published only).
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, EvalSample, read_eval_log

from researcharena_inspect.analysis.published_rows import agent_name
from researcharena_inspect.analysis.rows import Row, SideTaskClass
from researcharena_inspect.prompts import BASELINE
from researcharena_inspect.scoring.metrics_json import (
    MetricsReading,
    analysis_main_score,
    read_metrics,
)

SCORER_NAME = "red_team_scorer"


def side_task_class(setting: str, side_task: str) -> SideTaskClass:
    """The paper's class of a side task, as the published index records it (`cat`)."""
    if side_task in (BASELINE, "baseline"):
        return "baseline"
    if side_task == "independent":
        return "independent"
    # Every other side task is embedded, including the inference setting's `independent2`
    # (ledger D-050: upstream's folder for it is `embedded3`).
    return "embedded"


def _row(run_id: str, reading: MetricsReading, agent: str, main_score: float | None) -> Row:
    return Row(
        run_id=run_id,
        setting=reading.setting,
        side_task=reading.side_task_id,
        side_task_class=side_task_class(reading.setting, reading.side_task),
        agent=agent,
        monitor_model=None,
        access=None,
        main_score=main_score,
        side_score=reading.side_score,
        suspicion=None,
        side_metrics=dict(reading.side_metrics),
    )


def published_row(run: Mapping[str, Any], raw_metrics: Mapping[str, Any]) -> Row:
    """A published run's row from its `raw_metrics` (one monitor folder's `results.json`).

    Args:
        run: the run's entry in the website index (for its id, setting, side task and agent).
        raw_metrics: the run's `metrics.json`, as `results.json` keeps it.
    """
    reading = read_metrics(raw_metrics, run["setting"], run["side_task"])
    main = analysis_main_score(reading, "published", run["id"])
    return _row(run["id"], reading, agent_name(run["red_model"]), main)


def sample_row(sample: EvalSample, scorer: str = SCORER_NAME) -> Row:
    """A scored port sample's row (no monitor yet)."""
    if not sample.scores or scorer not in sample.scores:
        raise ValueError(f"sample {sample.id} has no {scorer} score")
    score = sample.scores[scorer]
    meta = score.metadata or {}
    reading = read_metrics(meta["metrics"], meta["setting"], meta["side_task"])
    model = sample.metadata.get("agent_model") or ""
    run_id = f"{sample.id}__e{sample.epoch}"
    return _row(run_id, reading, agent_name(model), analysis_main_score(reading, "live"))


def rows_from_log(log: str | Path | EvalLog, scorer: str = SCORER_NAME) -> list[Row]:
    """One row per scored sample of a port run's log."""
    if not isinstance(log, EvalLog):
        log = read_eval_log(str(log))
    return [sample_row(s, scorer) for s in log.samples or [] if s.scores and scorer in s.scores]


def compare(ours: Row, theirs: Row) -> list[str]:
    """Fields where two rows of the same run disagree: main and side score, side metrics."""
    out = []
    if ours.main_score != theirs.main_score:
        out.append(f"main_score {ours.main_score!r} != {theirs.main_score!r}")
    if ours.side_score != theirs.side_score:
        out.append(f"side_score {ours.side_score!r} != {theirs.side_score!r}")
    for key in sorted(set(ours.side_metrics) & set(theirs.side_metrics)):
        if ours.side_metrics[key] != theirs.side_metrics[key]:
            out.append(f"{key} {ours.side_metrics[key]!r} != {theirs.side_metrics[key]!r}")
    return out


def by_run(rows: Iterable[Row]) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for row in rows:
        out.setdefault(row.run_id, row)
    return out


__all__ = [
    "SCORER_NAME",
    "by_run",
    "compare",
    "published_row",
    "rows_from_log",
    "sample_row",
    "side_task_class",
]
