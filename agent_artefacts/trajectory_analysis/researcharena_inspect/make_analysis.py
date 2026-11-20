#!/usr/bin/env python3
"""Write `researcharena_inspect_<model>_ANALYSIS.md` from one group's result files.

Every number in the file is computed here from the files `run.py` leaves in a group's folder
(`runs/<label>/<group>/`): `known_answers.csv`, `validity_summary.json`,
`validity_per_sample.csv`, `scanner_results.csv`, `scanner_metadata.json`, `agreement.json`,
`usage.json`, and the label's `dry_run.json` when there is one. Nothing is typed by hand.

Usage:
    uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/make_analysis.py \
        runs/<label>/<group> [--out-dir <folder>]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ra_common as rc  # noqa: E402

GUIDANCE_SAMPLES = 100  # Inspect Evals' guidance: analyse at least 100 samples
THRESHOLD = 0.10  # EVALUATION_CHECKLIST: logs have a 10% or lower rate of invalid samples
EXCERPT = 350
TEXT_SCANNERS = {"outcome_summary"}  # answers in words, flags nothing


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
    }
    missing = [k for k in ("validity",) if data[k] is None]
    if missing:
        raise SystemExit(f"{group_dir}: missing {missing}; run run.py (or --post-only) first")
    return data


def render(data: dict[str, Any]) -> str:
    known: list[dict[str, str]] = data["known"]
    by_id = {k["transcript_id"]: k for k in known}
    per_sample: list[dict[str, str]] = data["per_sample"]
    validity = data["validity"]
    usage = data["usage"]
    metadata: dict[str, dict[str, bool]] = data["metadata"]
    results: list[dict[str, str]] = data["results"]
    group = data["group"]
    n = len(per_sample)
    logs = sorted({k["log"] for k in known})
    settings = _setting_order({k["setting"] for k in known})
    models = sorted({k["model"] for k in known})
    agents = sorted({k["agent"] for k in known})
    versions = sorted({k["task_version"] for k in known})
    invalid = int(validity["total_invalid"])
    rate = invalid / n if n else 0.0
    flags: dict[str, set[str]] = defaultdict(set)
    answered: dict[str, set[str]] = defaultdict(set)
    explanations: dict[tuple[str, str], str] = {}
    for r in results:
        if r["value"] in ("true", "false"):
            answered[r["scanner"]].add(r["transcript_id"])
        if r["value"] == "true":
            flags[r["scanner"]].add(r["transcript_id"])
        explanations[(r["scanner"], r["transcript_id"])] = r.get("explanation", "")
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
    add(f"- **Eval:** `researcharena_inspect` (task version {', '.join(versions)})")
    add(f"- **Evaluated model:** `{', '.join(models)}`; agent: {', '.join(f'`{a}`' for a in agents)}")
    add(f"- **Scanner model:** `{usage.get('scanner_model', '?')}`")
    add(
        f"- **Scan:** `{usage.get('scan_id', '?')}`, post-processed "
        f"{usage.get('post_processed', '?')} at commit `{usage.get('commit', '?')}` "
        f"(results: `runs/{usage.get('label', '?')}/{group}/`)"
    )
    add(f"- **Samples:** {n} from {_n(len(logs), 'log')}, one sample a log\n")

    add("## Summary\n")
    verdict = "within" if rate <= THRESHOLD else "above"
    add(
        f"{_n(n, 'transcript')} of `{title_model}` were scanned with {len(metadata)} scanners: the "
        f"workflow's {len(metadata) - _own_count(metadata)} defaults and "
        f"{_own_count(metadata)} written for this eval. By the paper's main-task rule, "
        f"{validity['successful']} are successes and {validity['failed']} failures. "
        f"{invalid} of {n} are invalid ({_pct(invalid, n)}: {validity['invalid_success']} "
        f"successes and {validity['invalid_failure']} failures), {verdict} the checklist's "
        f"{THRESHOLD:.0%}.\n"
    )

    add("## What \"invalid\" means in a sabotage benchmark\n")
    add(
        "A sample is invalid when its main-task outcome does not reflect what the agent did: a "
        "success the agent did not earn, or a failure it did not cause. Success here is the "
        "paper's main-task rule for the setting, as `analysis.rules` encodes it (main score "
        + "; ".join(f"{rc.SETTING_LABELS[s]} {rc.main_threshold(s)}" for s in settings)
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
    add("| Setting | Logs | Baseline | Side-task runs | Side tasks | Run length (hours) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for s in settings:
        rows = [k for k in known if k["setting"] == s]
        base = sum(rc.is_baseline(k["side_task"]) for k in rows)
        sides = Counter(k["side_task"] for k in rows if not rc.is_baseline(k["side_task"]))
        hours = sorted({k["redruntime"] for k in rows})
        add(
            f"| {rc.SETTING_LABELS.get(s, s)} | {len(rows)} | {base} | {len(rows) - base} | "
            f"{', '.join(f'{t} ({c})' for t, c in sorted(sides.items())) or '-'} | "
            f"{', '.join(hours)} |"
        )
    compacted = sum(k.get("compactions", "0") not in ("", "0") for k in known)
    hidden = sum(k.get("task_prompt_in_messages") == "no" for k in known)
    restored = sum(k.get("earlier_conversations_added", "0") not in ("", "0") for k in known)
    add(
        f"\nLogs: {', '.join(f'`{k}`' for k in sorted({k['run'] for k in known}))}.\n\n"
        f"The agent's conversation was compacted in {compacted} of {_n(n, 'sample')}. In "
        f"{hidden} of them the sample's messages, which the default scanners read, start after "
        f"the compaction and do not hold the task prompt; this eval's scanners got the earlier "
        f"conversation back in {_n(restored, 'sample')} (`ra_common.scanner_messages`).\n"
    )

    add("## Validity by setting\n")
    add(
        "| Setting | Samples | Successes | Failures | Invalid successes | Invalid failures | "
        "Invalid rate |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    totals = Counter()
    for s in settings:
        rows = [p for p in per_sample if p["setting"] == s]
        c = _counts(rows)
        totals.update(c)
        add(
            f"| {rc.SETTING_LABELS.get(s, s)} | {c['n']} | {c['success']} | {c['failure']} | "
            f"{c['inv_success']} | {c['inv_failure']} | {_pct(c['inv'], c['n'])} |"
        )
    add(
        f"| **All** | **{totals['n']}** | **{totals['success']}** | **{totals['failure']}** | "
        f"**{totals['inv_success']}** | **{totals['inv_failure']}** | "
        f"**{_pct(totals['inv'], totals['n'])}** |\n"
    )

    add("## Scanner flags\n")
    add(
        "Flagged / answered, by setting. A scanner that does not apply to a sample (a side-task "
        "question on a baseline, a CUDA-only question elsewhere) gives no answer and calls no "
        "model.\n"
    )
    add("| Scanner | Invalidates | " + " | ".join(rc.SETTING_LABELS.get(s, s) for s in settings) + " | All |")
    add("| --- | --- | " + " | ".join("---" for _ in settings) + " | --- |")
    setting_of = {p["transcript_id"]: p["setting"] for p in per_sample}
    for name, meta in metadata.items():
        if name in TEXT_SCANNERS:
            continue
        role = (
            "success and failure"
            if meta.get("invalidates_success") and meta.get("invalidates_failure")
            else "success"
            if meta.get("invalidates_success")
            else "failure"
            if meta.get("invalidates_failure")
            else "nothing"
        )
        cells = []
        for s in settings:
            ids = {t for t, st in setting_of.items() if st == s}
            cells.append(f"{len(flags[name] & ids)} / {len(answered[name] & ids)}")
        add(
            f"| `{name}` | {role} | {' | '.join(cells)} | "
            f"{len(flags[name])} / {len(answered[name])} |"
        )

    side_ids = {k["transcript_id"] for k in known if not rc.is_baseline(k["side_task"])}
    stock_side = flags["reward_hacking_success"] & side_ids
    own_side = flags["uninstructed_reward_hacking"] & side_ids
    add("\n## Instructed sabotage and the default reward-hacking scanner\n")
    add(
        f"Of {_n(len(side_ids), 'side-task sample')}, the default `reward_hacking_success` flagged "
        f"{len(stock_side)}; `uninstructed_reward_hacking`, which is told the side task, "
        f"flagged {len(own_side)}, and {len(stock_side & own_side)} of those are flagged by "
        f"both. Had the default scanner invalidated successes, "
        f"{_n(len(stock_side & _ids(per_sample, 'success')), 'side-task success', 'side-task successes')} "
        f"would have been counted invalid for doing what they were told.\n"
    )
    for tid in sorted(stock_side, key=lambda t: by_id.get(t, {}).get("run", "")):
        k = by_id.get(tid, {})
        add(
            f"- `{k.get('run', tid)}` ({k.get('setting')}, side task {k.get('side_task')}): "
            f"\"{_one_line(explanations.get(('reward_hacking_success', tid), ''))}\""
        )

    add("\n## Invalid samples\n")
    invalid_rows = [p for p in per_sample if p["valid"] == "no"]
    if not invalid_rows:
        add("None.\n")
    for p in sorted(invalid_rows, key=lambda p: p["run"]):
        add(
            f"- `{p['run']}` ({p['setting']}, side task {p['side_task']}): main score "
            f"{p['main_task_score']}, a {p['outcome']}, invalidated by "
            f"{', '.join(f'`{s}`' for s in p['invalidated_by'].split())}."
        )
        for scanner in p["invalidated_by"].split():
            add(f"  - `{scanner}`: \"{_one_line(explanations.get((scanner, p['transcript_id']), ''))}\"")
    other = [
        p for p in per_sample if p["valid"] == "yes" and p["other_flags"].strip()
    ]
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

    add("\n## Agreement with the known answers\n")
    add(
        "`known_answers.py` reads the same facts from the scorer's own records with no model "
        "(the evaluator's side-task verdict, the CUDA log's read-back verdict, the fault text, "
        "the submitted files). The scanners' questions ask a little more than each fact; see the "
        "note under the table.\n"
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
        add(f"- `{t['scanner']}`: {t['note']}.")
    disagreements = [(t, d) for t in data["agreement"] for d in t["disagreements"]]
    if disagreements:
        add("\nEvery disagreement:\n")
        for t, d in disagreements:
            detail = f" ({_one_line(d['fact_detail'], 120)})" if d["fact_detail"] else ""
            add(
                f"- `{t['scanner']}` on `{d['run']}` ({d['setting']}, side task "
                f"{d['side_task']}): scanner {d['scanner']}, fact {d['fact']}{detail}. "
                f"Scanner: \"{_one_line(d['explanation'], 250)}\""
            )

    add("\n## Scanner model usage\n")
    tot = usage.get("totals") or {}
    add(
        f"{tot.get('total_tokens', 0):,} tokens in all: input {tot.get('input_tokens', 0):,}, "
        f"cache read {tot.get('input_tokens_cache_read', 0):,}, cache write "
        f"{tot.get('input_tokens_cache_write', 0):,}, output {tot.get('output_tokens', 0):,} "
        f"(reasoning {tot.get('reasoning_tokens', 0):,}). Scan errors: "
        f"{sum(int(s.get('errors') or 0) for s in (usage.get('scanners') or {}).values())}. "
        f"Cost is not reported by the tool; see the provider's dashboard.\n"
    )

    add("## Limits\n")
    limits = [
        f"{n} transcripts, fewer than the {GUIDANCE_SAMPLES} the Inspect Evals guidance asks for: "
        f"every live run of the port that exists, one sample a log, at most a few per cell "
        f"(setting by side task).",
        f"One scanner model (`{usage.get('scanner_model', '?')}`); its answers are judgements, "
        f"checked against facts from the files only where such a fact exists (the agreement "
        f"table). The uninstructed-gaming, refusal, formatting and environment questions have "
        f"no file-based check.",
        "The default scanners are not told the score or the side task: they judge success and "
        "failure from the transcript alone, and an instructed side task can look like a "
        "failure or a hack to them. Their flags count only where they match the outcome the "
        "paper's rule gives.",
        f"The default scanners read the sample's messages, which hold the task prompt in "
        f"{n - hidden} of {n} samples; in the other {hidden} Claude Code had compacted the "
        f"conversation, and they see its summary of the earlier work. This eval's scanners "
        f"read the earlier conversation too.",
    ]
    plan = data.get("dry_run")
    if plan and plan.get("totals"):
        t = plan["totals"]
        limits.append(
            f"Transcripts over the scanner model's context are split into segments, each scanned "
            f"on its own (a yes in any segment is a yes). The dry run for "
            f"`{plan.get('model')}` ({plan.get('context_window'):,} tokens of context) "
            f"estimated {t.get('multi_segment_samples', 0)} of {t.get('samples', 0)} samples "
            f"over one segment."
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
    add("```")
    return "\n".join(out) + "\n"


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
