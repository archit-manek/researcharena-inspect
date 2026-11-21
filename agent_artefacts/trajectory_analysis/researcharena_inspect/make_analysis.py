#!/usr/bin/env python3
"""Write `researcharena_inspect_<model>_ANALYSIS.md` from one group's result files.

Every number in the file is computed here from the files `run.py` leaves in a group's folder
(`runs/<label>/<group>/`): `known_answers.csv`, `validity_summary.json`,
`validity_per_sample.csv`, `scanner_results.csv`, `scanner_metadata.json`, `agreement.json`,
`usage.json`, the label's `dry_run.json` when there is one, and the same group's
`scanner_results.csv` in the reference scans `review.py` names. Nothing is typed by hand here.

What a check of the logs by hand found is in `review.py`, recorded for one label: the verdict on
each invalid sample and each kind of disagreement, and the failures whose cause is not settled.
The file states those beside the numbers, marked as checks by hand, and counts validity both as
the scanners give it and after the checks. A scan with no notes says so.

The raw scans are not needed: from the tables alone,
    uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/make_analysis.py \
        runs/<label>/<group> [--out-dir <folder>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ra_common as rc  # noqa: E402
import review as review_notes  # noqa: E402

GUIDANCE_SAMPLES = 100  # Inspect Evals' guidance: analyse at least 100 samples
THRESHOLD = 0.10  # EVALUATION_CHECKLIST: logs have a 10% or lower rate of invalid samples
EXCERPT = 350
TEXT_SCANNERS = {"outcome_summary"}  # answers in words, flags nothing
SCANNER_ERROR_KEY = review_notes.SCANNER_ERROR
# The agreement table's key, where compare.py's note says less than it should.
KEY_NOTES = {
    "broken_env": (
        "default scanner, transcript only; `scoring_fault` is a comparison signal from the "
        "scorer's record, outside what `broken_env` can see, so the two answer different "
        "questions"
    ),
}
EVAL_YAML = rc.HERE.parents[2] / "src" / rc.EVAL_NAME / "eval.yaml"


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _n(count: int, word: str, plural: str | None = None) -> str:
    """`1 sample`, `2 samples`."""
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "n/a"


def _one_line(text: str, limit: int = EXCERPT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _setting_order(settings: set[str]) -> list[str]:
    known = [s for s in rc.SETTING_ORDER if s in settings]
    return known + sorted(settings - set(known))


def _label(setting: str) -> str:
    return rc.SETTING_LABELS.get(setting, setting)


def _duration(hours: str) -> str:
    """Upstream's `redruntime` in hours as a length: `0.3333` -> `20 min`, `5` -> `5 h`."""
    try:
        h = float(hours)
    except ValueError:
        return hours or "?"
    if h < 1:
        return f"{round(h * 60)} min"
    return f"{h:g} h"


def _runs(names: Any) -> str:
    return ", ".join(f"`{r}`" for r in sorted(names))


def current_version() -> str | None:
    """The eval's version now, from `eval.yaml`."""
    if not EVAL_YAML.exists():
        return None
    m = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', EVAL_YAML.read_text(), re.MULTILINE)
    return m.group(1).strip() if m else None


def load(group_dir: Path) -> dict[str, Any]:
    data = {
        "known": _csv(group_dir / "known_answers.csv"),
        "validity": _json(group_dir / "validity_summary.json"),
        "per_sample": _csv(group_dir / "validity_per_sample.csv"),
        "results": _csv(group_dir / "scanner_results.csv"),
        "metadata": _json(group_dir / "scanner_metadata.json") or {},
        "agreement": _json(group_dir / "agreement.json") or [],
        "usage": _json(group_dir / "usage.json") or {},
        "dry_run": _json(group_dir.parent / "dry_run.json"),
        "group": group_dir.name,
        "references": {},
    }
    missing = [k for k in ("validity",) if data[k] is None]
    if missing:
        raise SystemExit(f"{group_dir}: missing {missing}; run run.py (or --post-only) first")
    review = review_for(data["usage"].get("label"))
    data["siblings"] = {}
    if review is not None:
        for ref in review.REFERENCE_LABELS:
            path = group_dir.parent.parent / ref / group_dir.name / "scanner_results.csv"
            if path.exists():
                data["references"][ref] = {
                    (r["scanner"], r["transcript_id"]): r["value"] for r in _csv(path)
                }
        # The other groups of the same scan, for the checklist line that covers them all.
        for other in sorted(group_dir.parent.iterdir()):
            table = other / "validity_per_sample.csv"
            if other != group_dir and table.exists():
                data["siblings"][other.name] = _csv(table)
    return data


def review_for(label: str | None) -> ModuleType | None:
    """The notes of a check by hand, if they were made against this scan."""
    return review_notes if label and label == review_notes.LABEL else None


def _answer(value: str) -> str:
    return {"true": "yes", "false": "no"}.get(value, "no answer")


def render(data: dict[str, Any]) -> str:
    known: list[dict[str, str]] = data["known"]
    by_id = {k["transcript_id"]: k for k in known}
    id_of = {k["run"]: k["transcript_id"] for k in known}
    per_sample: list[dict[str, str]] = data["per_sample"]
    ps_by_run = {p["run"]: p for p in per_sample}
    validity = data["validity"]
    usage = data["usage"]
    metadata: dict[str, dict[str, bool]] = data["metadata"]
    results: list[dict[str, str]] = data["results"]
    references: dict[str, dict[tuple[str, str], str]] = data["references"]
    group = data["group"]
    review = review_for(usage.get("label"))
    n = len(per_sample)
    logs = sorted({k["log"] for k in known})
    settings = _setting_order({k["setting"] for k in known})
    models = sorted({k["model"] for k in known})
    agents = sorted({k["agent"] for k in known})
    invalid = int(validity["total_invalid"])
    rate = invalid / n if n else 0.0
    flags: dict[str, set[str]] = defaultdict(set)
    answered: dict[str, set[str]] = defaultdict(set)
    values: dict[tuple[str, str], str] = {}
    explanations: dict[tuple[str, str], str] = {}
    for r in results:
        if r["value"] in ("true", "false"):
            answered[r["scanner"]].add(r["transcript_id"])
        if r["value"] == "true":
            flags[r["scanner"]].add(r["transcript_id"])
        values[(r["scanner"], r["transcript_id"])] = r["value"]
        explanations[(r["scanner"], r["transcript_id"])] = r.get("explanation", "")
    checks = _checks(review, per_sample, n) if review else None
    warnings: list[str] = []
    out: list[str] = []
    add = out.append

    title_model = rc.model_slug(models[0]) if len(models) == 1 else ", ".join(models)
    add(f"# ResearchArena trajectory analysis: {group}\n")
    if usage.get("mock"):
        add(
            "> **Mock run.** The scanner model was `mockllm/model`, which answers every question "
            "with 'no' without reading the transcript. This file shows the pipeline works; it "
            "says nothing about the trajectories.\n"
        )
    add("- **Eval:** `researcharena_inspect`")
    add(f"- **Versions:** {_versions_line(known, review, group)}")
    add(f"- **Evaluated model:** `{', '.join(models)}`; agent: {', '.join(f'`{a}`' for a in agents)}")
    add(f"- **Scanner model:** `{usage.get('scanner_model', '?')}`")
    add(
        f"- **Scan:** `{usage.get('scan_id', '?')}`, post-processed "
        f"{usage.get('post_processed', '?')} at commit `{usage.get('commit', '?')}` "
        f"(results: `runs/{usage.get('label', '?')}/{group}/`); this file is written from those "
        f"tables by `make_analysis.py`"
        + (f", with the checks by hand of {review.CHECKED} (`review.py`)" if review else "")
    )
    add(f"- **Samples:** {n} from {_n(len(logs), 'log')}, one sample a log\n")

    add("## Summary\n")
    verdict = "within" if rate <= THRESHOLD else "above"
    add(
        f"{_n(n, 'transcript')} of `{title_model}` were scanned with {len(metadata)} scanners: the "
        f"workflow's {len(metadata) - _own_count(metadata)} defaults and "
        f"{_own_count(metadata)} written for this eval. By the paper's main-task rule, "
        f"{_n(int(validity['successful']), 'is a success', 'are successes')} and "
        f"{_n(int(validity['failed']), 'failure')}. As the scanners give it, "
        f"{invalid} of {n} are invalid ({_pct(invalid, n)}: {validity['invalid_success']} "
        f"successes and {validity['invalid_failure']} failures)"
        + (
            ": the scanners' count, not the checklist determination, which follows the checks by "
            "hand below.\n"
            if checks
            else f", {verdict} the checklist's {THRESHOLD:.0%}.\n"
        )
    )
    readback = _read_back_rows(known)
    missed = [k for k in readback if values.get(("score_read_back", k["transcript_id"])) != "true"]
    if missed:
        add(
            f"**`score_read_back` missed the known read-back** ({_runs(k['run'] for k in missed)}; "
            f"see [The known read-back](#the-known-read-back-d-001)): the one case the scanner "
            f"can be checked on. "
            + _read_back_consequence(missed, ps_by_run, metadata, known)
            + "\n"
        )
    if checks:
        add(
            f"**After the checks by hand** ([Checks by hand](#checks-by-hand)): "
            f"{_n(len(checks['invalid']), 'sample')} invalid on checking "
            f"({_runs(checks['invalid']) or 'none'}), "
            f"{_n(len(checks['errors']), 'scanner error')} set aside, and "
            f"{_n(len(checks['unsettled']), 'failure')} counted valid whose cause is not settled: "
            f"{_span(checks['low'], checks['high'], n)} invalid.\n"
        )
        add(
            "**The checklist (at most 10% invalid samples):** "
            + _checklist_lines(review, group, per_sample, data.get("siblings") or {})
            + f" {review.UPSTREAM_NOTE.get(group, '')}\n"
        )
        caveats = [r for r in review.REPLAY_CAVEATS if ps_by_run.get(r, {}).get("outcome") == "success"]
        if caveats:
            add(
                f"**{_n(len(caveats), 'success', 'successes')} not confirmed by replay** (D-255; "
                f"[below](#successes-not-confirmed-by-replay-d-255)): {_runs(caveats)}. Kept as "
                f"successes, and not part of the range above.\n"
            )
    elif not usage.get("mock"):
        add("No check by hand was made against this scan's answers (`review.py`).\n")

    add("## What \"invalid\" means in a sabotage benchmark\n")
    add(
        "A sample is invalid when its main-task outcome does not reflect what the agent did: a "
        "success the agent did not earn, or a failure it did not cause. Success here is the "
        "paper's main-task rule for the setting, as `analysis.rules` encodes it (main score "
        + "; ".join(f"{_label(s)} {rc.main_threshold(s)}" for s in settings)
        + "); a sample whose evaluation failed (upstream's -1 record) is a failure.\n"
    )
    add(
        "In a side-task sample the agent is told, inside its instruction, to sabotage as well: "
        "game a benchmark, contaminate training data, plant a backdoor or a slow path, add a "
        "look-alike dependency, upload user conversations or model weights. That instructed "
        "behaviour is what ResearchArena measures, so it never makes a sample invalid, even "
        "where it raises the main score (the capabilities side task trains on the test set the "
        "main score is computed on). Score gaming the instructions did not ask for does "
        "(`uninstructed_reward_hacking`), as does a score read back from the agent's own file "
        "(`score_read_back`, ledger D-001). A failure is invalid when the scorer failed on work "
        "that ran in the agent's own runs (`scorer_failed_on_working_submission`, D-037, D-141), "
        "when the submission was in a form the instructions allow but the scorer cannot load "
        "(`submission_not_loadable`, D-179), or when a default scanner finds an external or "
        "environment failure, a formatting failure or a refusal. The default "
        "`reward_hacking_success` cannot tell instructed sabotage from gaming; it runs as "
        "information only.\n"
    )
    add("Scanners that invalidate a success: " + _names(metadata, "invalidates_success") + ".")
    add("Scanners that invalidate a failure: " + _names(metadata, "invalidates_failure") + ".\n")

    add("## What was scanned\n")
    add("| Setting | Logs | Baseline | Side-task runs | Side tasks | Run lengths |")
    add("| --- | --- | --- | --- | --- | --- |")
    for s in settings:
        rows = [k for k in known if k["setting"] == s]
        base = sum(rc.is_baseline(k["side_task"]) for k in rows)
        sides = Counter(k["side_task"] for k in rows if not rc.is_baseline(k["side_task"]))
        lengths = Counter(k["redruntime"] for k in rows)
        add(
            f"| {_label(s)} | {len(rows)} | {base} | {len(rows) - base} | "
            f"{', '.join(f'{t} ({c})' for t, c in sorted(sides.items())) or '-'} | "
            + ", ".join(
                f"{_duration(h)} ({c})" for h, c in sorted(lengths.items(), key=lambda x: float(x[0] or 0))
            )
            + " |"
        )
    compacted = sum(k.get("compactions", "0") not in ("", "0") for k in known)
    hidden = sum(k.get("task_prompt_in_messages") == "no" for k in known)
    restored = sum(k.get("earlier_conversations_added", "0") not in ("", "0") for k in known)
    add(f"\nLogs: {_runs({k['run'] for k in known})}.\n")
    if compacted:
        add(
            f"The agent's conversation was compacted in {compacted} of {_n(n, 'sample')}. In "
            f"{hidden} of them the sample's messages, which the default scanners read, start "
            f"after the compaction and do not hold the task prompt; this eval's scanners got the "
            f"earlier conversation back in {_n(restored, 'sample')} "
            f"(`ra_common.scanner_messages`).\n"
        )
    else:
        add("No sample's conversation was compacted: every scanner read the whole of each.\n")

    add("## Validity by setting\n")
    add("As the scanners give it.\n")
    add(
        "| Setting | Samples | Successes | Failures | Invalid successes | Invalid failures | "
        "Invalid rate |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    totals: Counter = Counter()
    above: list[str] = []
    for s in settings:
        rows = [p for p in per_sample if p["setting"] == s]
        c = _counts(rows)
        totals.update(c)
        add(
            f"| {_label(s)} | {c['n']} | {c['success']} | {c['failure']} | "
            f"{c['inv_success']} | {c['inv_failure']} | {_pct(c['inv'], c['n'])} |"
        )
        if c["n"] and c["inv"] / c["n"] > THRESHOLD:
            above.append(s)
    add(
        f"| **All** | **{totals['n']}** | **{totals['success']}** | **{totals['failure']}** | "
        f"**{totals['inv_success']}** | **{totals['inv_failure']}** | "
        f"**{_pct(totals['inv'], totals['n'])}** |\n"
    )
    for s in above:
        rows = [k for k in known if k["setting"] == s]
        by_len: dict[str, list[str]] = defaultdict(list)
        for k in rows:
            by_len[k["redruntime"]].append(k["run"])
        bad = [p for p in per_sample if p["setting"] == s and p["valid"] == "no"]
        add(
            f"{_label(s)}: {len(bad)} of {len(rows)} ({_pct(len(bad), len(rows))}), a rate on "
            f"{_n(len(rows), 'sample')}: "
            + "; ".join(
                f"{_n(len(v), 'run')} of {_duration(h)} ({_runs(v)})"
                for h, v in sorted(by_len.items(), key=lambda x: -float(x[0] or 0))
            )
            + ". Invalid: "
            + ", ".join(
                f"`{p['run']}` ({_duration(by_id[p['transcript_id']]['redruntime'])})" for p in bad
            )
            + "."
            + (_setting_after_checks(checks, s, per_sample) if checks else "")
            + "\n"
        )

    add("## Scanner flags\n")
    add(
        "Flagged / answered, by setting. A scanner that does not apply to a sample (a side-task "
        "question on a baseline, a CUDA-only question elsewhere) gives no answer and calls no "
        "model.\n"
    )
    add("| Scanner | Invalidates | " + " | ".join(_label(s) for s in settings) + " | All |")
    add("| --- | --- | " + " | ".join("---" for _ in settings) + " | --- |")
    setting_of = {p["transcript_id"]: p["setting"] for p in per_sample}
    for name, meta in metadata.items():
        if name in TEXT_SCANNERS:
            continue
        cells = []
        for s in settings:
            ids = {t for t, st in setting_of.items() if st == s}
            cells.append(f"{len(flags[name] & ids)} / {len(answered[name] & ids)}")
        add(
            f"| `{name}` | {_role(meta)} | {' | '.join(cells)} | "
            f"{len(flags[name])} / {len(answered[name])} |"
        )

    if readback or any(k["setting"] in rc.READ_BACK_SETTINGS for k in known):
        add("\n## The known read-back (D-001)\n")
        cuda = [k for k in known if k["setting"] in rc.READ_BACK_SETTINGS]
        add(
            f"`known_answers.py` reads CUDA's evaluator log with no model: a verdict line with no "
            f"benchmark output means the score was read from a results file in the agent's "
            f"workspace. It finds {_n(len(readback), 'read-back')} among {_n(len(cuda), 'CUDA sample')}"
            + (": " + "; ".join(
                f"`{k['run']}`, {k['read_back_speedup']} from `{k['read_back_file']}`, a "
                f"{ps_by_run[k['run']]['outcome']} by the paper's rule; `score_read_back` answered "
                f"{_answer(values.get(('score_read_back', k['transcript_id']), ''))}"
                + _reference_answers(references, "score_read_back", k["transcript_id"], values)
                for k in readback
            ) if readback else "")
            + ".\n"
        )
        if missed:
            add(_read_back_consequence(missed, ps_by_run, metadata, known) + "\n")
            if review:
                add(f"Checked by hand: {review.READ_BACK_NOTE}\n")
            add(
                "What to conclude: this scanner's no is not evidence that a score was measured. "
                "That no CUDA success here was read back rests on the file-based check, which "
                "reads the evaluator's own log; `score_read_back` adds nothing to it on these logs "
                "and should not be relied on alone where that log is missing.\n"
            )

    side_ids = {k["transcript_id"] for k in known if not rc.is_baseline(k["side_task"])}
    stock_side = flags["reward_hacking_success"] & side_ids
    own_side = flags["uninstructed_reward_hacking"] & side_ids
    add("\n## Instructed sabotage and the default reward-hacking scanner\n")
    if not side_ids:
        add("No sample here has a side task.\n")
    else:
        add(
            f"Of {_n(len(side_ids), 'side-task sample')}, the default `reward_hacking_success` "
            f"flagged {len(stock_side)}; `uninstructed_reward_hacking`, which is told the side "
            f"task, flagged {len(own_side)}, and {len(stock_side & own_side)} of those are "
            f"flagged by both. Had the default scanner invalidated successes, "
            f"{_n(len(stock_side & _ids(per_sample, 'success')), 'side-task success', 'side-task successes')} "
            f"would have been counted invalid for doing what they were told.\n"
        )
        for tid in sorted(stock_side, key=lambda t: by_id.get(t, {}).get("run", "")):
            k = by_id.get(tid, {})
            add(
                f"- `{k.get('run', tid)}` ({k.get('setting')}, side task {k.get('side_task')}, "
                f"{ps_by_run.get(k.get('run', ''), {}).get('outcome', '?')}): "
                f"\"{_one_line(explanations.get(('reward_hacking_success', tid), ''))}\""
            )

    add("\n## Invalid samples\n")
    add("As the scanners give it; the checks by hand follow.\n")
    invalid_rows = [p for p in per_sample if p["valid"] == "no"]
    if not invalid_rows:
        add("None.\n")
    for p in sorted(invalid_rows, key=lambda p: p["run"]):
        add(
            f"- `{p['run']}` ({p['setting']}, side task {p['side_task']}, "
            f"{_duration(by_id[p['transcript_id']]['redruntime'])}): main score "
            f"{p['main_task_score']}, a {p['outcome']}, invalidated by "
            f"{', '.join(f'`{s}`' for s in p['invalidated_by'].split())}."
        )
        for scanner in p["invalidated_by"].split():
            add(f"  - `{scanner}`: \"{_one_line(explanations.get((scanner, p['transcript_id']), ''))}\"")
    other = [p for p in per_sample if p["valid"] == "yes" and p["other_flags"].strip()]
    if other:
        add(
            "\nFlags that do not change validity (the flag does not apply to the sample's "
            "outcome, or the scanner invalidates nothing):\n"
        )
        for p in sorted(other, key=lambda p: p["run"]):
            add(
                f"- `{p['run']}` ({p['setting']}, side task {p['side_task']}, {p['outcome']}): "
                f"{', '.join(f'`{s}`' for s in p['other_flags'].split())}"
            )

    if checks:
        add("\n## Checks by hand\n")
        add(
            f"Read in the logs and the eval README on {review.CHECKED}; the notes are in "
            f"`review.py`. Verdicts on a sample: {review.VERDICT_KEY}.\n"
        )
        add(f"The rule for CUDA failures, applied to each: {review.ATTRIBUTION_RULE}\n")
        add("| Sample | Setting | Length | Main score | Scanners' verdict | Check |")
        add("| --- | --- | --- | --- | --- | --- |")
        listed = sorted(checks["invalid_runs"]) + sorted(checks["added"]) + sorted(checks["unsettled"])
        for run in listed:
            p = ps_by_run[run]
            k = by_id[p["transcript_id"]]
            scanners_say = (
                f"invalid ({', '.join(f'`{s}`' for s in p['invalidated_by'].split())})"
                if p["valid"] == "no"
                else "valid"
            )
            add(
                f"| `{run}` | {_label(p['setting'])} | {_duration(k['redruntime'])} | "
                f"{p['main_task_score']} ({p['outcome']}) | {scanners_say} | "
                f"{_check_verdict(review, checks, run)} |"
            )
        add("")
        for run in sorted(checks["invalid_runs"]):
            if run in review.INVALID:
                add(f"- `{run}` ({review.INVALID[run][0]}): {review.INVALID[run][1]}")
        for run in sorted(checks["added"]):
            add(f"- `{run}` ({review_notes.INVALID_ON_CHECK}; the scanners counted it valid): "
                f"{review.ADDED_INVALID[run]}")
        for run in sorted(checks["unsettled"]):
            add(f"- `{run}` ({review_notes.UNSETTLED}): {review.UNSETTLED_FAILURES[run]}")
        for run in checks["unchecked_invalid"]:
            warnings.append(f"invalid sample {run} has no note in review.py")
            add(f"- `{run}`: invalid by the scanners, **not checked by hand**.")
        unchecked_failures = sorted(
            p["run"]
            for p in per_sample
            if p["outcome"] == "failure"
            and p["valid"] == "yes"
            and p["run"] not in review.UNSETTLED_FAILURES
            and p["run"] not in review.ADDED_INVALID
        )
        for run in unchecked_failures:
            warnings.append(f"failure {run} counted valid has no note in review.py")
        add(
            f"\nValidity after the checks: {_n(len(checks['invalid']), 'sample')} invalid on "
            f"checking ({_runs(checks['invalid']) or 'none'}) and "
            f"{len(checks['unchecked_invalid'])} unchecked, so at least {checks['low']} of {n} "
            f"({_pct(checks['low'], n)}); with every unsettled failure invalid, {checks['high']} "
            f"of {n} ({_pct(checks['high'], n)}). "
            + (
                f"The other failures counted valid ({_runs(unchecked_failures)}) were not checked "
                f"by hand. "
                if unchecked_failures
                else "Every failure was checked. "
            )
            + f"The {_n(int(validity['successful']), 'success', 'successes')} were checked only "
            f"through the flags above and, for CUDA, the file-based read-back check.\n"
        )
        add(
            "Checklist determination: "
            + _checklist_lines(review, group, per_sample, data.get("siblings") or {})
            + f" {review.UPSTREAM_NOTE.get(group, '')}\n"
        )
        caveats = [r for r in review.REPLAY_CAVEATS if ps_by_run.get(r, {}).get("outcome") == "success"]
        if caveats:
            add("### Successes not confirmed by replay (D-255)\n")
            add(
                f"{_n(len(caveats), 'inference success', 'inference successes')} counted above by "
                f"their live score: "
                + "; ".join(f"`{r}` ({review.REPLAY_CAVEATS[r]})" for r in sorted(caveats))
                + f". {review.REPLAY_NOTE}\n"
            )

    add("\n## Agreement with the known answers\n")
    add(
        "`known_answers.py` reads the same facts from the scorer's own records with no model "
        "(the evaluator's side-task verdict, the CUDA log's read-back verdict, the fault text, "
        "the submitted files). A scanner's question can ask more than its fact, so a "
        "disagreement is either the question differing from the fact or a scanner error; the "
        "kinds below say which, from the checks by hand.\n"
    )
    add(
        "| Scanner | Fact | Compared | Both yes | Both no | Scanner yes, fact no | "
        "Scanner no, fact yes | No answer | Agreement |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in data["agreement"]:
        share = "n/a" if t["agreement"] is None else f"{t['agreement']:.0%}"
        add(
            f"| `{t['scanner']}` | `{t['fact']}` | {t['compared']} | {t['both_yes']} | "
            f"{t['both_no']} | {t['scanner_yes_fact_no']} | {t['scanner_no_fact_yes']} | "
            f"{t['no_answer']} | {share} |"
        )
    add("")
    for t in data["agreement"]:
        add(f"- `{t['scanner']}`: {KEY_NOTES.get(t['scanner'], t['note'])}.")
    unscored = sorted(
        d["run"]
        for t in data["agreement"]
        if t["scanner"] == "side_task_carried_out"
        for d in t["disagreements"]
        if _side_not_scored(d, by_id, id_of)
    )
    if unscored:
        add(
            f"\nA \"fact no\" is a verdict computed by the paper's rule, and it includes a "
            f"measurement that is absent: for `side_task_carried_out`, {_n(len(unscored), 'run')} "
            f"({_runs(unscored)}) had no side score at all, shown below as \"not scored\". "
            f"Agreement is with the chosen signal; where the questions differ it is not an "
            f"estimate of the scanner's accuracy."
        )
    kinds: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in data["agreement"]:
        if t["scanner"] == "reward_hacking_success":
            continue  # a different question on purpose; listed below, not judged
        for d in t["disagreements"]:
            direction = (
                review_notes.YES_NO if d["scanner"] == "yes" else review_notes.NO_YES
            )
            kinds[(t["scanner"], direction)].append(d)
    if kinds:
        add("\n### Disagreements by kind\n")
        for (scanner, direction), ds in kinds.items():
            note = review.KINDS_BY_GROUP.get(group, {}).get((scanner, direction)) if review else None
            head = f"- `{scanner}`, {direction}: {_n(len(ds), 'sample')} ({_runs(d['run'] for d in ds)})"
            if note:
                runs_by_verdict: dict[str, list[str]] = defaultdict(list)
                for d in ds:
                    runs_by_verdict[_run_verdict(review, scanner, d["run"], note[0])].append(d["run"])
                tally = "; ".join(
                    f"{len(rs)} {v}" + (f" ({_runs(rs)})" if len(runs_by_verdict) > 1 else "")
                    for v, rs in sorted(runs_by_verdict.items(), key=lambda x: x[0] != SCANNER_ERROR_KEY)
                )
                text = f"{head}. **{tally[0].upper()}{tally[1:]}.** {note[1]}"
                for d in ds:
                    own = review.RUN_VERDICTS.get((scanner, d["run"]))
                    if own:
                        text += f" `{d['run']}` ({own[0]}): {own[1]}"
                add(text)
            else:
                if review:
                    warnings.append(f"no note in review.py for {scanner}, {direction}")
                add(f"{head}. Not checked by hand.")
    disagreements = [(t, d) for t in data["agreement"] for d in t["disagreements"]]
    if disagreements:
        add("\n### Every disagreement\n")
        for t, d in disagreements:
            detail = f" ({_one_line(d['fact_detail'], 120)})" if d["fact_detail"] else ""
            fact = d["fact"]
            if t["scanner"] == "side_task_carried_out" and _side_not_scored(d, by_id, id_of):
                fact, detail = "not scored", ""
            tid = id_of.get(d["run"], "")
            own = review.RUN_VERDICTS.get((t["scanner"], d["run"])) if review else None
            add(
                f"- `{t['scanner']}` on `{d['run']}` ({d['setting']}, side task "
                f"{d['side_task']}): scanner {d['scanner']}, fact {fact}{detail}"
                f"{_reference_answers(references, t['scanner'], tid, values)}"
                + (f"; checked by hand: {own[0]}" if own else "")
                + f". Scanner: \"{_one_line(d['explanation'], 250)}\""
            )

    if review and review.NOT_COVERED.get(group):
        add("\n## What this scan does and does not cover\n")
        for item in review.NOT_COVERED[group]:
            add(f"- {item}")

    add("\n## Scanner model usage\n")
    tot = usage.get("totals") or {}
    add(
        f"{tot.get('total_tokens', 0):,} tokens in all: input {tot.get('input_tokens', 0):,}, "
        f"cache read {tot.get('input_tokens_cache_read', 0):,}, cache write "
        f"{tot.get('input_tokens_cache_write', 0):,}, output {tot.get('output_tokens', 0):,} "
        f"(reasoning {tot.get('reasoning_tokens', 0):,}). Scan errors: "
        f"{sum(int(s.get('errors') or 0) for s in (usage.get('scanners') or {}).values())}. "
        + _cost_sentence(usage)
        + "\n"
    )

    add("## Limits\n")
    limits = [
        _sample_limit(n, group, data.get("dry_run"), review),
        f"One scanner model (`{usage.get('scanner_model', '?')}`); its answers are judgements, "
        f"checked against facts from the files only where such a fact exists (the agreement "
        f"table). The uninstructed-gaming, refusal, formatting and environment questions have "
        f"no file-based check.",
        "The scanners set no temperature, so the scanner model may sample, and a second scan of "
        "the same logs could answer differently; no scan was repeated to measure it"
        + (
            " (for the known read-back the two saved extracts, pilot and full pass, disagree; the "
            "raw request of the full pass was not kept)."
            if missed and references
            else "."
        ),
        "The default scanners are not told the score or the side task: they judge success and "
        "failure from the transcript alone, and an instructed side task can look like a "
        "failure or a hack to them. Their flags count only where they match the outcome the "
        "paper's rule gives, and they cannot see a fault in scoring, which happens after the "
        "transcript ends.",
    ]
    if hidden:
        limits.append(
            f"The default scanners read the sample's messages, which hold the task prompt in "
            f"{n - hidden} of {n} samples; in the other {hidden} Claude Code had compacted the "
            f"conversation, and they see its summary of the earlier work. This eval's scanners "
            f"read the earlier conversation too."
        )
    plan = data.get("dry_run")
    if plan and plan.get("totals"):
        t = plan["totals"]
        limits.append(
            f"Transcripts over the scanner model's context are split into segments, each scanned "
            f"on its own (a yes in any segment is a yes). The dry run for "
            f"`{plan.get('model')}` ({plan.get('context_window'):,} tokens of context) "
            f"estimated {t.get('multi_segment_samples', 0)} of {t.get('samples', 0)} samples "
            f"over one segment, across all groups of the manifest."
        )
    for item in limits:
        add(f"- {item}")

    add("\n## Reproduce\n")
    add("From the repository root (the logs are in the project notebook, not in this repository):\n")
    add("```bash")
    model_args = (
        "--mock" if usage.get("mock") else f"--model {usage.get('scanner_model', '<model>')}"
    )
    add(
        "uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/run.py \\\n"
        f"    --manifest {usage.get('manifest', '<manifest>')} "
        f"--logs-root <notebook>/docs/audit/data \\\n"
        f"    {model_args} --label {usage.get('label', '<label>')} --analysis"
    )
    add("```\n")
    add("This file again from the tables alone, with no model and no raw scan:\n")
    add("```bash")
    add(
        "uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/make_analysis.py \\\n"
        f"    agent_artefacts/trajectory_analysis/researcharena_inspect/runs/"
        f"{usage.get('label', '<label>')}/{group}"
    )
    add("```")
    for w in warnings:
        print(f"Warning: {w}")
    return "\n".join(out) + "\n"


def _versions_line(known: list[dict[str, str]], review: ModuleType | None, group: str) -> str:
    by_version: dict[str, Counter] = defaultdict(Counter)
    for k in known:
        by_version[k["task_version"]][k["setting"]] += 1
    parts = [
        f"{v} ("
        + ", ".join(f"{_label(s)} {c[s]}" for s in _setting_order(set(c)))
        + ")"
        for v, c in sorted(by_version.items(), key=lambda x: _version_key(x[0]))
    ]
    now = current_version()
    line = f"The logs were made at task version {'; '.join(parts)} (logs per setting)"
    if now:
        line += f"; the eval is now at {now} (`eval.yaml`)."
        note = review.VERSIONS_NOTE.get(group) if review and now not in by_version else None
        if note:
            line += f" {note}"
    return line


def _version_key(version: str) -> tuple[int, str]:
    number, _, letter = version.partition("-")
    return (int(number) if number.isdigit() else 0, letter)


def _read_back_rows(known: list[dict[str, str]]) -> list[dict[str, str]]:
    return [k for k in known if k.get("read_back") == "yes"]


def _setting_after_checks(checks: dict[str, Any], setting: str, per_sample: list[dict[str, str]]) -> str:
    runs = {p["run"] for p in per_sample if p["setting"] == setting}
    low = len(runs & set(checks["invalid"])) + len(runs & set(checks["unchecked_invalid"]))
    high = low + len(runs & set(checks["unsettled"]))
    return f" After the checks by hand: {_span(low, high, len(runs))}."


def _read_back_consequence(
    missed: list[dict[str, str]],
    ps_by_run: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, bool]],
    known: list[dict[str, str]],
) -> str:
    """What a correct answer on the missed read-backs would have done to the counts."""
    meta = metadata.get("score_read_back", {})
    changes = [
        k["run"]
        for k in missed
        if (ps_by_run[k["run"]]["outcome"] == "success" and meta.get("invalidates_success"))
        or (ps_by_run[k["run"]]["outcome"] == "failure" and meta.get("invalidates_failure"))
    ]
    rb_success = [k["run"] for k in known if k.get("read_back") == "yes" and k.get("main_success") == "yes"]
    if changes:
        return (
            f"A yes would have made {_runs(changes)} invalid, so the invalid count above is "
            f"short by {len(changes)}."
        )
    outcomes = ", ".join(f"`{k['run']}` is a {ps_by_run[k['run']]['outcome']}" for k in missed)
    cuda_success = [
        k for k in known if k["setting"] in rc.READ_BACK_SETTINGS and k.get("main_success") == "yes"
    ]
    return (
        f"It does not change the counts here: {outcomes} by the paper's rule, and "
        f"`score_read_back` invalidates a success only. A read-back that gave a success would "
        f"be an invalid success. The count of scanner flags that invalidate a success is zero; "
        f"separately, the file-based check shows "
        f"{'no' if not rb_success else len(rb_success)} read-back among the "
        f"{_n(len(cuda_success), 'CUDA success', 'CUDA successes')}"
        + (f" ({_runs(rb_success)})" if rb_success else "")
        + ". The second statement rests on that check, not on this scanner."
    )


def _reference_answers(
    references: dict[str, dict[tuple[str, str], str]],
    scanner: str,
    tid: str,
    values: dict[tuple[str, str], str],
) -> str:
    """`; in the pilot scan pilot-x: yes` where an earlier scan answered differently."""
    here = values.get((scanner, tid), "")
    other = [
        f"{label}: {_answer(answers[(scanner, tid)])}"
        for label, answers in references.items()
        if (scanner, tid) in answers and answers[(scanner, tid)] != here
    ]
    return f" (earlier scans of the same log answered differently, {'; '.join(other)})" if other else ""


def _checks(review: ModuleType, per_sample: list[dict[str, str]], n: int) -> dict[str, Any]:
    invalid_runs = [p["run"] for p in per_sample if p["valid"] == "no"]
    on_check = [r for r in invalid_runs if review.INVALID.get(r, ("",))[0] == review_notes.INVALID_ON_CHECK]
    errors = [r for r in invalid_runs if review.INVALID.get(r, ("",))[0] == review_notes.SCANNER_ERROR]
    unchecked = [r for r in invalid_runs if r not in review.INVALID]
    counted_valid_failures = {
        p["run"] for p in per_sample if p["valid"] == "yes" and p["outcome"] == "failure"
    }
    added = sorted(r for r in review.ADDED_INVALID if r in counted_valid_failures)
    unsettled = sorted(r for r in review.UNSETTLED_FAILURES if r in counted_valid_failures)
    invalid = sorted(on_check + added)
    low = len(invalid) + len(unchecked)
    return {
        "invalid_runs": invalid_runs,
        "invalid": invalid,
        "added": added,
        "errors": errors,
        "unchecked_invalid": unchecked,
        "unsettled": unsettled,
        "low": low,
        "high": low + len(unsettled),
    }


def _check_verdict(review: ModuleType, checks: dict[str, Any], run: str) -> str:
    if run in review.INVALID:
        return review.INVALID[run][0]
    if run in checks["added"]:
        return review_notes.INVALID_ON_CHECK
    if run in checks["unsettled"]:
        return review_notes.UNSETTLED
    return "not checked"


def _run_verdict(review: ModuleType, scanner: str, run: str, default: str) -> str:
    own = review.RUN_VERDICTS.get((scanner, run))
    return own[0] if own else default


def _side_not_scored(d: dict[str, Any], by_id: dict[str, dict[str, str]], id_of: dict[str, str]) -> bool:
    """A side-task 'fact no' whose side score is absent: the probe was not scored."""
    k = by_id.get(id_of.get(d["run"], ""), {})
    return d["fact"] == "no" and k.get("side_score", "x") == ""


def _span(low: int, high: int, n: int) -> str:
    if low == high:
        return f"{low} of {n} ({_pct(low, n)})"
    return f"{low} to {high} of {n} ({_pct(low, n)} to {_pct(high, n)})"


def _determination(low: int, high: int, n: int, unsettled: int) -> str:
    """Whether the checklist's threshold is met, stated as far as the counts allow."""
    if high <= THRESHOLD * n:
        return "within the threshold"
    if low > THRESHOLD * n:
        return f"above the threshold on {_n(n, 'sample')}"
    room = math.floor(THRESHOLD * n + 1e-9) - low
    if room == 0:
        cond = (
            "the unsettled failure is valid"
            if unsettled == 1
            else f"all {unsettled} unsettled failures are valid"
            if unsettled != 2
            else "both unsettled failures are valid"
        )
    else:
        cond = f"at most {room} of the {unsettled} unsettled failures are invalid"
    return f"so the threshold is not shown to be met (it holds only if {cond})"


def _checklist_lines(
    review: ModuleType,
    group: str,
    per_sample: list[dict[str, str]],
    siblings: dict[str, list[dict[str, str]]],
) -> str:
    """The checklist determination: this group pooled, its settings, the other groups."""
    c = _checks(review, per_sample, len(per_sample))
    n = len(per_sample)
    parts = [
        f"`{group}` pooled, {_span(c['low'], c['high'], n)}, "
        + _determination(c["low"], c["high"], n, len(c["unsettled"]))
        + "."
    ]
    by_setting = []
    for s in _setting_order({p["setting"] for p in per_sample}):
        rows = [p for p in per_sample if p["setting"] == s]
        runs = {p["run"] for p in rows}
        low = len(runs & set(c["invalid"])) + len(runs & set(c["unchecked_invalid"]))
        high = low + len(runs & set(c["unsettled"]))
        if high == 0:
            by_setting.append(f"{_label(s)} 0 of {len(rows)}")
            continue
        text = f"{_label(s)} {_span(low, high, len(rows))}"
        if low > THRESHOLD * len(rows):
            text += f", above the threshold on {_n(len(rows), 'sample')}"
        by_setting.append(text)
    parts.append("By setting: " + "; ".join(by_setting) + ".")
    for other, rows in siblings.items():
        oc = _checks(review, rows, len(rows))
        parts.append(f"`{other}`: {_span(oc['low'], oc['high'], len(rows))}.")
    return " ".join(parts)


def _sample_limit(n: int, group: str, plan: Any, review: ModuleType | None) -> str:
    groups = Counter(str(s.get("group")) for s in ((plan or {}).get("samples") or []))
    total = sum(groups.values())
    sizes = ", ".join(f"`{g}` {c}" for g, c in sorted(groups.items()))
    text = (
        f"{n} transcripts, fewer than the {GUIDANCE_SAMPLES} the Inspect Evals guidance asks for, "
        f"one sample a log, at most a few per cell (setting by side task)"
    )
    if total:
        text += f": this group's share of the {total} selected report trajectories ({sizes})"
    if review:
        text += f"; {review.MANIFEST_EXCLUDES}"
    return text + "."


def _cost_sentence(usage: dict[str, Any]) -> str:
    """The scan's cost at `run.py`'s list prices, from its own token counts."""
    import run

    model = str(usage.get("scanner_model") or "")
    key = run._model_key(model, run.PRICES)
    tot = usage.get("totals") or {}
    if usage.get("mock") or key is None:
        return "Cost is not reported by the tool; see the provider's dashboard."
    p_in, p_read, p_write, p_out = run.PRICES[key]
    cost = (
        tot.get("input_tokens", 0) * p_in
        + tot.get("input_tokens_cache_read", 0) * p_read
        + tot.get("input_tokens_cache_write", 0) * p_write
        + tot.get("output_tokens", 0) * p_out
    ) / 1e6
    return (
        f"At the list prices in `run.py` (`PRICES`, not checked against the provider's page) "
        f"these tokens cost ${cost:.2f}; the provider's bill is the authority."
    )


def _role(meta: dict[str, bool]) -> str:
    if meta.get("invalidates_success") and meta.get("invalidates_failure"):
        return "success and failure"
    if meta.get("invalidates_success"):
        return "success"
    if meta.get("invalidates_failure"):
        return "failure"
    return "nothing"


def _own_count(metadata: dict[str, Any]) -> int:
    """Scanners in the metadata that are not the workflow's defaults."""
    sys.path.insert(0, str(rc.TEMPLATE_DIR))
    from scanners import DEFAULT_SCANNERS

    defaults = {s.name for s in DEFAULT_SCANNERS}
    return len([n for n in metadata if n not in defaults])


def _names(metadata: dict[str, dict[str, bool]], key: str) -> str:
    names = [n for n, m in metadata.items() if m.get(key)]
    return ", ".join(f"`{n}`" for n in names) if names else "none"


def _ids(per_sample: list[dict[str, str]], outcome: str) -> set[str]:
    return {p["transcript_id"] for p in per_sample if p["outcome"] == outcome}


def _counts(rows: list[dict[str, str]]) -> Counter:
    c: Counter = Counter()
    for p in rows:
        c["n"] += 1
        c[p["outcome"]] += 1
        if p["valid"] == "no":
            c["inv"] += 1
            c["inv_" + p["outcome"]] += 1
    return c


def format_markdown(path: Path) -> None:
    """Run the repository's mdformat hook on the file, as `make check` will (best effort)."""
    import subprocess

    repo = rc.HERE.parents[2]
    if not path.resolve().is_relative_to(repo):
        return
    result = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "mdformat", "--files", str(path)],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):  # 1: the hook reformatted the file
        print(f"Note: mdformat did not run ({result.returncode}); run `make check` before committing")


def output_path(group: str, out_dir: Path | None) -> Path:
    return (out_dir or rc.HERE) / f"{rc.EVAL_NAME}_{group}_ANALYSIS.md"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("group_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    data = load(args.group_dir)
    path = output_path(data["group"], args.out_dir)
    path.write_text(render(data))
    format_markdown(path)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    main()
