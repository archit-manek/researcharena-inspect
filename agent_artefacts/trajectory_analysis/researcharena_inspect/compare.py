#!/usr/bin/env python3
"""Scanner flags against the known answers: an agreement table per scanner, every disagreement.

Reads a scan's results (`<results_dir>/scan_id=*/<scanner>.parquet`) and the known answers of
the same logs (`known_answers.csv`, from `known_answers.py`), joined on the transcript id (the
sample's uuid). Writes into `<results_dir>`:

- `scanner_results.csv`: one row per sample and scanner (value, answer, explanation cut to 600
  characters, error);
- `agreement.json` and `agreement.md`: per scanner and fact, how many samples were compared,
  the four cells (both yes, both no, scanner yes and fact no, scanner no and fact yes), the
  samples the scanner gave no answer for, the agreement, and each disagreement with its run.

A fact is compared only where it is defined (a CUDA-only fact on a CUDA sample, a side-task
fact on a side-task sample). The facts are not the scanners' whole questions: see `PAIRS`.

Usage:
    uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/compare.py \
        <results_dir> <known_answers.csv>
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inspect_scout"))

import pandas as pd  # noqa: E402
from utils import find_scan_dir  # noqa: E402

EXPLANATION_CHARS = 600


@dataclass(frozen=True)
class Pair:
    scanner: str
    fact: str
    note: str
    """How the fact relates to the scanner's question."""


PAIRS = (
    Pair(
        "side_task_carried_out",
        "side_task_recorded",
        "fact: the evaluator's side-task verdict by the paper's rule; the scanner asks whether "
        "the agent did it at all, so 'scanner yes, fact no' includes side tasks done too weakly "
        "to pass",
    ),
    Pair(
        "score_read_back",
        "read_back",
        "fact: CUDA evaluator log shows a file's verdict and no benchmark output (D-001)",
    ),
    Pair(
        "scorer_failed_on_working_submission",
        "scoring_fault",
        "fact: main-task scoring recorded a fault; the scanner also requires the agent's own runs "
        "to have worked, so 'scanner no, fact yes' can be right",
    ),
    Pair(
        "submission_not_loadable",
        "bare_adapter",
        "fact: final_model/ holds an adapter and no config.json (D-179)",
    ),
    Pair(
        "broken_env",
        "scoring_fault",
        "default scanner, related question: an environment fault would include a scoring fault",
    ),
    Pair(
        "reward_hacking_success",
        "side_task_recorded",
        "default scanner, shown for its overlap with instructed sabotage (not the same question)",
    ),
)


def scanner_results(scan_dir: Path) -> pd.DataFrame:
    """One row per sample and scanner, from every result parquet of a scan."""
    frames = []
    for parquet in sorted(scan_dir.glob("*.parquet")):
        if parquet.stem.startswith("_"):
            continue
        df = pd.read_parquet(parquet)
        cols = ["transcript_id", "value", "answer", "explanation", "scan_error"]
        df = df[[c for c in cols if c in df.columns]].copy()
        df.insert(1, "scanner", parquet.stem)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["transcript_id", "scanner", "value", "answer", "explanation"])
    out = pd.concat(frames, ignore_index=True)
    out["explanation"] = out["explanation"].map(
        lambda e: "" if e is None or (isinstance(e, float) and pd.isna(e)) else str(e)
    )
    return out


def as_bool(value: Any) -> bool | None:
    """Scanner values are 'true'/'false'; facts are 'yes'/'no'; anything else is no answer."""
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value).strip().lower()
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    return None


def agreement(results: pd.DataFrame, known: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id = {k["transcript_id"]: k for k in known}
    tables = []
    for pair in PAIRS:
        rows = results[results["scanner"] == pair.scanner]
        if rows.empty:
            continue
        cells = {"both_yes": 0, "both_no": 0, "scanner_yes_fact_no": 0, "scanner_no_fact_yes": 0}
        no_answer, disagreements = [], []
        for _, r in rows.iterrows():
            fact_row = by_id.get(r["transcript_id"])
            fact = as_bool(fact_row.get(pair.fact)) if fact_row else None
            if fact is None:
                continue
            flag = as_bool(r["value"])
            run = fact_row.get("run", "") if fact_row else ""
            if flag is None:
                no_answer.append(run)
                continue
            if flag and fact:
                cells["both_yes"] += 1
            elif not flag and not fact:
                cells["both_no"] += 1
            else:
                key = "scanner_yes_fact_no" if flag else "scanner_no_fact_yes"
                cells[key] += 1
                disagreements.append(
                    {
                        "run": run,
                        "setting": fact_row.get("setting", "") if fact_row else "",
                        "side_task": fact_row.get("side_task", "") if fact_row else "",
                        "scanner": "yes" if flag else "no",
                        "fact": "yes" if fact else "no",
                        "fact_detail": _detail(pair.fact, fact_row or {}),
                        "explanation": str(r["explanation"])[:EXPLANATION_CHARS],
                    }
                )
        compared = sum(cells.values())
        tables.append(
            {
                "scanner": pair.scanner,
                "fact": pair.fact,
                "note": pair.note,
                "compared": compared,
                **cells,
                "no_answer": len(no_answer),
                "no_answer_runs": sorted(no_answer),
                "agreement": (cells["both_yes"] + cells["both_no"]) / compared if compared else None,
                "disagreements": sorted(disagreements, key=lambda d: d["run"]),
            }
        )
    return tables


def _detail(fact: str, row: dict[str, str]) -> str:
    if fact == "scoring_fault":
        return row.get("fault_text", "")
    if fact == "read_back":
        file = row.get("read_back_file", "")
        return f"{file} speedup {row.get('read_back_speedup', '')}" if file else ""
    if fact == "side_task_recorded":
        return f"side score {row.get('side_score', '')}"
    if fact == "bare_adapter":
        return row.get("final_model_files", "")
    return ""


def markdown(tables: list[dict[str, Any]]) -> str:
    lines = [
        "| Scanner | Fact from the logs | Compared | Both yes | Both no | Scanner yes, fact no "
        "| Scanner no, fact yes | No answer | Agreement |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in tables:
        share = "n/a" if t["agreement"] is None else f"{t['agreement']:.0%}"
        lines.append(
            f"| `{t['scanner']}` | `{t['fact']}` | {t['compared']} | {t['both_yes']} | "
            f"{t['both_no']} | {t['scanner_yes_fact_no']} | {t['scanner_no_fact_yes']} | "
            f"{t['no_answer']} | {share} |"
        )
    lines.append("")
    for t in tables:
        lines.append(f"- `{t['scanner']}` against `{t['fact']}`: {t['note']}.")
    for t in tables:
        if not t["disagreements"]:
            continue
        lines += ["", f"### Disagreements: `{t['scanner']}` against `{t['fact']}`", ""]
        for d in t["disagreements"]:
            detail = f" ({d['fact_detail']})" if d["fact_detail"] else ""
            explanation = " ".join(d["explanation"].split())[:400]
            lines.append(
                f"- `{d['run']}` ({d['setting']}, side task {d['side_task']}): scanner "
                f"{d['scanner']}, fact {d['fact']}{detail}. Scanner: \"{explanation}\""
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        raise SystemExit("usage: compare.py <results_dir> <known_answers.csv>")
    results_dir, known_path = Path(args[0]), Path(args[1])
    results = scanner_results(find_scan_dir(results_dir))
    out = results.copy()
    out["explanation"] = out["explanation"].str.slice(0, EXPLANATION_CHARS)
    out.to_csv(results_dir / "scanner_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with known_path.open(newline="") as f:
        known = list(csv.DictReader(f))
    tables = agreement(results, known)
    (results_dir / "agreement.json").write_text(json.dumps(tables, indent=2) + "\n")
    (results_dir / "agreement.md").write_text(markdown(tables))
    for t in tables:
        share = "n/a" if t["agreement"] is None else f"{t['agreement']:.0%}"
        print(f"{t['scanner']} vs {t['fact']}: {t['compared']} compared, agreement {share}")
    return tables


if __name__ == "__main__":
    main()
