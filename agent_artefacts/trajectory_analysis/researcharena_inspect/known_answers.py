#!/usr/bin/env python3
"""Known answers: one row per sample, the facts the scanners are asked about, from files only.

Reads each live `.eval` log and takes every fact from the scorer's own records (see
`ra_common`), with no model:

- `side_task_recorded`: the evaluator's verdict on the side task by the paper's rule
  (`analysis.rules.side_task_success`); empty for a baseline.
- `read_back`: CUDA only (D-001): the main score came from a results file, not a benchmark run.
  `read_back_file` and `read_back_speedup` name the file in the kept workspace.
- `scoring_fault` and `fault_text`: main-task scoring failed (no metrics.json, CUDA's "Main task
  evaluation error", an inference gate error).
- `bare_adapter`: post-training only (D-179): final_model/ holds an adapter and no config.json.
- `main_success`: the paper's main-task rule for the setting (`analysis.rules`); a failed
  evaluation is a failure.
- `compactions`, `task_prompt_in_messages` and `earlier_conversations_added`: what the scanners
  can see. The default scanners read the sample's messages, which after a Claude Code
  compaction may start from its summary, without the task prompt (and so without the side
  task's instruction); this eval's scanners get the earlier conversations put back in front
  (`ra_common.scanner_messages`).

Usage:
    uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/known_answers.py \
        --manifest full --logs-root <notebook>/docs/audit/data --out known_answers.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ra_common as rc  # noqa: E402

COLUMNS = [
    "transcript_id",
    "run",
    "log",
    "sample_id",
    "epoch",
    "task_version",
    "model",
    "agent",
    "group",
    "setting",
    "side_task",
    "redruntime",
    "main_task_score",
    "side_score",
    "evaluation_failed",
    "main_success",
    "side_task_recorded",
    "read_back",
    "read_back_file",
    "read_back_speedup",
    "scoring_fault",
    "fault_text",
    "side_probe_error",
    "bare_adapter",
    "final_model_files",
    "compactions",
    "task_prompt_in_messages",
    "earlier_conversations_added",
    "sample_error",
    "sample_limit",
]


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def task_prompt_in_messages(sample: Any) -> bool:
    """Whether the sample's first prompt appears in a user message of its messages."""
    return rc.prompt_in(sample.messages, rc.prompt_key(sample.input))


def sample_row(sample: Any, log_header: Any, entry: rc.LogEntry | None = None) -> dict[str, Any]:
    """The known-answer row of one scored sample."""
    from researcharena_inspect.analysis.rules import side_task_success
    from researcharena_inspect.scoring.rows import sample_row as analysis_row

    meta = sample.metadata or {}
    setting = str(meta.get("setting", ""))
    side_task = str(meta.get("side_task", ""))
    score = sample.scores.get(rc.SCORER_NAME) if sample.scores else None
    value = rc.score_dict(score.value) if score else {}
    smeta: dict[str, Any] = dict(score.metadata or {}) if score else {}
    agent = meta.get("agent") or (log_header.eval.task_args or {}).get("agent")
    model = str(log_header.eval.model)
    row: dict[str, Any] = {
        "transcript_id": sample.uuid,
        "run": entry.run if entry else "",
        "log": entry.rel if entry else "",
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "task_version": log_header.eval.task_version,
        "model": model,
        "agent": agent,
        "group": rc.group_name(model, agent),
        "setting": setting,
        "side_task": side_task,
        "redruntime": meta.get("redruntime"),
        "main_task_score": value.get("main_task_score"),
        "side_score": value.get("side_score"),
        "evaluation_failed": bool(value.get("evaluation_failed")) if score else None,
        "sample_error": (sample.error.message.splitlines()[0][:200] if sample.error else None),
        "sample_limit": sample.limit.type if sample.limit else None,
        "compactions": sum(getattr(e, "event", None) == "compaction" for e in sample.events),
        "task_prompt_in_messages": task_prompt_in_messages(sample),
        "earlier_conversations_added": rc.scanner_messages(sample, model)[1],
    }
    if score is None:
        return row
    row["main_success"] = rc.main_task_success(setting, value)
    side = None if rc.is_baseline(side_task) else side_task_success(analysis_row(sample))
    row["side_task_recorded"] = side
    if setting in rc.READ_BACK_SETTINGS:
        read_back = rc.cuda_read_back(smeta)
        row["read_back"] = read_back
        if read_back:
            row["read_back_file"], row["read_back_speedup"] = rc.snapshot_results_file(
                sample.store or {}
            )
    fault = rc.scoring_fault(setting, smeta)
    row["scoring_fault"] = fault is not None
    row["fault_text"] = fault
    row["side_probe_error"] = rc.side_probe_error(smeta)
    if setting in rc.POST_TRAINING:
        row["bare_adapter"] = rc.bare_adapter(smeta)
        files = rc.final_model_files(smeta)
        row["final_model_files"] = (
            " ".join(sorted(f.removeprefix("final_model/") for f in files))
            if files is not None
            else None
        )
    return row


def rows_for_logs(entries: Iterable[rc.LogEntry]) -> list[dict[str, Any]]:
    from inspect_ai.log import read_eval_log

    rows = []
    for entry in entries:
        log = read_eval_log(str(entry.path))
        for sample in log.samples or []:
            rows.append(sample_row(sample, log, entry))
    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in COLUMNS})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", default="full", help="manifest name or path")
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    entries, empty = rc.resolve_logs(rc.read_manifest(rc.manifest_path(args.manifest)), args.logs_root)
    for pattern in empty:
        print(f"Note: no log matches {pattern}")
    rows = rows_for_logs(entries)
    write_csv(rows, args.out)
    print(f"{len(rows)} samples from {len(entries)} logs -> {args.out}")


if __name__ == "__main__":
    main()
