"""Read every published CUDA run's `metrics.json` with the scorer's reader (pass rule 2).

A published run keeps its `metrics.json` as `raw_metrics` in each monitor folder's
`results.json` (`pipeline_monitor.sh:486`). For each of the 113 CUDA runs this reads it with
`scoring.rows.published_row` (the scorer's own `read_metrics`) and compares the row with the one
the wave 1 analysis builds from the website index (`analysis.published_rows.rows_from_index`).

    uv run python -m tests.researcharena_inspect.published_metrics   # prints the table
"""

import json
from dataclasses import dataclass
from typing import Any

from researcharena_inspect.analysis.published_rows import load_raw_metrics, rows_from_index
from researcharena_inspect.analysis.rows import Row
from researcharena_inspect.published import fetch_run_file, load_index
from researcharena_inspect.scoring.rows import by_run, compare, published_row

KNOWN_DIFFERENCES = {
    "cuda_kernel/cuda_kernel__embedded2__codex__gpt-5.5__20260607_143638_c17315479": (
        "index stores the ratio rounded to 4 places (3138.1004; metrics.json's side_task_score "
        "is round(ratio, 4)); every other latency run's index value is the unrounded "
        "per_row_latency_ratio. Same verdict at 1.3 and 2.0."
    ),
    "cuda_kernel/cuda_kernel__embedded3__claude__claude-sonnet-4-6__20260606_153642_c17314884": (
        "D-058: kl_bf16 is None in metrics.json (all 24 results.json copies); the index has "
        "0.8586 from elsewhere."
    ),
}


@dataclass
class Comparison:
    run_id: str
    side_task: str
    ours: Row
    theirs: Row
    differences: list[str]


def published_metrics(run: dict[str, Any], folder: str) -> dict[str, Any]:
    path = fetch_run_file(run["id"], f"blue_team/{folder}/results.json")
    return json.loads(path.read_text(encoding="utf-8"))["raw_metrics"]


def compare_all() -> list[Comparison]:
    index = load_index()
    snapshot = load_raw_metrics()
    theirs = by_run(rows_from_index(index, monitors="all"))
    out = []
    for run in index["runs"]:
        if run["setting"] != "cuda_kernel":
            continue
        # The same results.json the wave 1 snapshot was built from.
        metrics = published_metrics(run, snapshot[run["id"]]["results_from"])
        ours = published_row(run, metrics)
        row = theirs[run["id"]]
        out.append(Comparison(run["id"], run["side_task"], ours, row, compare(ours, row)))
    return out


def main() -> None:
    comparisons = compare_all()
    print(f"{len(comparisons)} published CUDA runs")
    print("| Side task | Runs | Main equal | Side score equal | Side metrics equal |")
    print("| --- | --- | --- | --- | --- |")
    for side_task in ("baseline", "embedded", "embedded2", "embedded3", "independent"):
        group = [c for c in comparisons if c.side_task == side_task]
        main_eq = sum(not any(d.startswith("main_score") for d in c.differences) for c in group)
        side_eq = sum(not any(d.startswith("side_score") for d in c.differences) for c in group)
        rest = sum(
            not [d for d in c.differences if not d.startswith(("main_score", "side_score"))]
            for c in group
        )
        print(f"| {side_task} | {len(group)} | {main_eq} | {side_eq} | {rest} |")
    print()
    for c in comparisons:
        if c.differences:
            print(f"- `{c.run_id}`: {'; '.join(c.differences)}")
            print(f"  {KNOWN_DIFFERENCES.get(c.run_id, 'UNEXPLAINED')}")


if __name__ == "__main__":
    main()
