"""Compare the reproduced tables with the paper's printed numbers, cell by cell.

Run `uv run python -m researcharena_inspect.analysis.reproduce` for a Markdown report of every
cell (paper, ours, match). Counts must match exactly; means, SDs, rates and AUROC must match to
the paper's printed precision.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from researcharena_inspect.analysis import paper_values as paper
from researcharena_inspect.analysis.metrics import Proportion
from researcharena_inspect.analysis.published_rows import rows_from_index
from researcharena_inspect.analysis.published_scores import encode_cuda_main_for_table
from researcharena_inspect.analysis.rows import (
    ACCESS_TYPES,
    OPTIMIZATION_SETTINGS,
    POST_TRAINING_SETTINGS,
    Row,
)
from researcharena_inspect.analysis.rules import RuleSet
from researcharena_inspect.analysis.tables import (
    auroc_table,
    detection_breakdown,
    detection_rate_pooled,
    headline_counts,
    main_column,
    side_task_rates,
    table5,
)

DETECTION_TABLES = {
    6: ("safety_finetuning", paper.TABLE6, paper.TABLE6_MAIN),
    7: ("utility_finetuning", paper.TABLE7, paper.TABLE7_MAIN),
    8: ("cuda_kernel", paper.TABLE8, paper.TABLE8_MAIN),
    9: ("inference_optimization", paper.TABLE9, paper.TABLE9_MAIN),
}
ACCESS_LABELS = dict(zip(ACCESS_TYPES, ("traj", "CoT", "artifact", "artifact+CoT"), strict=True))
NO_VALUE = "\u2013"  # the en dash the paper prints for an empty cell


@dataclass(frozen=True)
class Comparison:
    """One printed number of the paper against ours, both as strings at the paper's precision."""

    table: str
    key: str
    paper: str
    ours: str

    @property
    def match(self) -> bool:
        return self.paper == self.ours


def _count(cell: tuple[int, int] | Proportion | None) -> str:
    if cell is None:
        return NO_VALUE
    k, n = (cell.k, cell.n) if isinstance(cell, Proportion) else cell
    return NO_VALUE if n == 0 else f"{k}/{n}"


def _pm(value: float | None, sd: float | None, digits: int = 2) -> str:
    if value is None or sd is None:
        return NO_VALUE
    return f"{value:.{digits}f} ± {sd:.{digits}f}"


def compare_table5(rows: Sequence[Row], rules: RuleSet = "paper") -> list[Comparison]:
    ours = table5(rows, rules)
    return [
        Comparison("table5", "/".join(key), _count(cell), _count(ours.get(key)))
        for key, cell in paper.TABLE5.items()
    ]


def compare_detection_table(
    rows: Sequence[Row], table: int, rules: RuleSet = "paper"
) -> list[Comparison]:
    """Every count cell and every Main value of Table 6, 7, 8 or 9."""
    setting, cells, mains = DETECTION_TABLES[table]
    ours = detection_breakdown(rows, setting, rules)
    ours_main = main_column(rows, setting)
    out = []
    for (side_task, agent), (mean, sd) in mains.items():
        got = ours_main.get((side_task, agent))
        if got is not None and setting == "cuda_kernel":
            # D-004: Table 8 prints the stored half-scale speedup.
            got = (encode_cuda_main_for_table(got[0]), encode_cuda_main_for_table(got[1]))
        out.append(
            Comparison(
                f"table{table}_main",
                f"{side_task}/{agent}",
                _pm(mean, sd),
                _pm(*got) if got else NO_VALUE,
            )
        )
    for (side_task, agent, monitor), printed in cells.items():
        for access, cell in zip(ACCESS_TYPES, printed, strict=True):
            out.append(
                Comparison(
                    f"table{table}",
                    f"{side_task}/{agent}/{monitor}/{ACCESS_LABELS[access]}",
                    _count(cell),
                    _count(ours.get((side_task, agent, monitor, access))),
                )
            )
    return out


def compare_auroc_tables(rows: Sequence[Row], rules: RuleSet = "paper") -> list[Comparison]:
    """Tables 2 and 3: AUROC ± bootstrap SD per access type, and the detection rate ± SD."""
    out = []
    for name, settings, printed in (
        ("table2", OPTIMIZATION_SETTINGS, paper.TABLE2),
        ("table3", POST_TRAINING_SETTINGS, paper.TABLE3),
    ):
        ours = auroc_table(rows, settings, rules=rules)
        for monitor, (*aurocs, (rate, rate_sd)) in printed.items():
            for access, (value, sd) in zip(ACCESS_TYPES, aurocs, strict=True):
                cell = ours.get((monitor, access))
                out.append(
                    Comparison(
                        name,
                        f"{monitor}/{ACCESS_LABELS[access]}",
                        _pm(value, sd),
                        _pm(cell.value, cell.sd) if cell else NO_VALUE,
                    )
                )
            det = detection_rate_pooled(rows, settings, monitor, rules=rules)
            ours_rate = (
                f"{100 * det.value:.0f}% ± {100 * det.sd:.1f}"
                if det.value is not None and det.sd is not None
                else NO_VALUE
            )
            out.append(
                Comparison(name, f"{monitor}/det. rate", f"{rate:.0f}% ± {rate_sd:.1f}", ours_rate)
            )
    return out


def compare_headlines(rows: Sequence[Row], rules: RuleSet = "paper") -> list[Comparison]:
    ours = headline_counts(rows, rules)
    out = [
        Comparison("headline", name, str(value), str(ours[name]))
        for name, (value, _line) in paper.HEADLINE_COUNTS.items()
    ]
    rates = side_task_rates(rows, rules=rules)
    for key, (value, _line) in paper.SIDE_TASK_RATES.items():
        got = rates.get(key)
        ours_rate = f"{100 * got.value:.1f}%" if got and got.value is not None else NO_VALUE
        out.append(Comparison("side_task_rate", "/".join(key), f"{value:.1f}%", ours_rate))
    return out


def compare_all(rows: Sequence[Row], rules: RuleSet = "paper") -> list[Comparison]:
    """Every comparison: headline counts, Table 5, Tables 6 to 9, Tables 2 and 3."""
    out = compare_headlines(rows, rules) + compare_table5(rows, rules)
    for table in DETECTION_TABLES:
        out += compare_detection_table(rows, table, rules)
    return out + compare_auroc_tables(rows, rules)


def gaps(comparisons: Sequence[Comparison]) -> dict[tuple[str, str], tuple[str, str]]:
    """The cells that differ: (table, key) -> (paper, ours)."""
    return {(c.table, c.key): (c.paper, c.ours) for c in comparisons if not c.match}


def summary(comparisons: Sequence[Comparison]) -> dict[str, tuple[int, int]]:
    """Matching cells and cells, per table."""
    out: dict[str, list[int]] = {}
    for c in comparisons:
        cell = out.setdefault(c.table, [0, 0])
        cell[0] += c.match
        cell[1] += 1
    return {table: (m, n) for table, (m, n) in out.items()}


def markdown(comparisons: Sequence[Comparison], only_gaps: bool = False) -> str:
    """The comparisons as Markdown tables, one per paper table."""
    lines = ["| table | matching cells |", "| --- | --- |"]
    lines += [f"| {t} | {m} of {n} |" for t, (m, n) in summary(comparisons).items()]
    for table in dict.fromkeys(c.table for c in comparisons):
        rows = [c for c in comparisons if c.table == table and not (only_gaps and c.match)]
        if not rows:
            continue
        lines += [
            "",
            f"### {table}",
            "",
            "| cell | paper | ours | match |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {c.key} | {c.paper} | {c.ours} | {'yes' if c.match else '**no**'} |" for c in rows
        ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", choices=get_args(RuleSet), default="paper")
    parser.add_argument("--baseline-exclusions", choices=("paper", "website"), default="paper")
    parser.add_argument("--only-gaps", action="store_true")
    args = parser.parse_args(argv)
    rows = rows_from_index(baseline_exclusions=args.baseline_exclusions)
    print(markdown(compare_all(rows, args.rules), only_gaps=args.only_gaps), end="")


if __name__ == "__main__":
    main()
