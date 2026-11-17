"""Scoring a red-team run as upstream's pipeline does, and reading `metrics.json`.

- `metrics_json`: the one place where `metrics.json` (a live run's, or a published run's
  `raw_metrics`) becomes upstream's two scores, the side task's headline value and the fields
  the wave 1 analysis's success rules read.
- `scorer`: `red_team_scorer`, Phase 4 of `src/pipeline_redteam.sh` in the sample's sandbox.
- `rows`: a scored Inspect log as `analysis.Row`s, read the same way as the published runs.
"""

from researcharena_inspect.scoring.metrics_json import (
    FAILED_METRICS_RECORD,
    SIDE_SCORE_FIELDS,
    MetricsReading,
    analysis_main_score,
    parse_metrics,
    read_metrics,
)
from researcharena_inspect.scoring.scorer import (
    Attempt,
    EvaluationRun,
    free_gpu_memory,
    red_team_scorer,
    run_evaluation,
    score_value,
)

__all__ = [
    "FAILED_METRICS_RECORD",
    "SIDE_SCORE_FIELDS",
    "Attempt",
    "EvaluationRun",
    "MetricsReading",
    "analysis_main_score",
    "free_gpu_memory",
    "parse_metrics",
    "read_metrics",
    "red_team_scorer",
    "run_evaluation",
    "score_value",
]
