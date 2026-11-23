"""The paper's analysis as code: success rules, detection rate, AUROC and table reproduction.

Upstream's code has none of it (D-035); it follows the paper's definitions and is checked against
the authors' published runs (`reproduce`).
"""

from researcharena_inspect.analysis.metrics import (
    Proportion,
    auroc,
    bootstrap_auroc_sd,
    detection_rate,
)
from researcharena_inspect.analysis.rows import Row
from researcharena_inspect.analysis.rules import (
    THRESHOLDS,
    exclusion_reason,
    main_task_success,
    run_success,
    side_task_success,
)

__all__ = [
    "THRESHOLDS",
    "Proportion",
    "Row",
    "auroc",
    "bootstrap_auroc_sd",
    "detection_rate",
    "exclusion_reason",
    "main_task_success",
    "run_success",
    "side_task_success",
]
