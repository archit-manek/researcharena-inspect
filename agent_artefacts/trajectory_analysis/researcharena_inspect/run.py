#!/usr/bin/env python3
"""Run the check-trajectories workflow on ResearchArena's live logs, in one command.

The template's `run_all_scanners.py` takes one log file (it counts samples with
`read_eval_log(<path>)`, which fails on a folder) and writes the scanners' validity flags as
they are declared. This runner keeps everything else of the workflow and adds what this eval
needs:

1. Logs come from a manifest of globs (`manifests/pilot.txt`, `manifests/full.txt`) under
   `--logs-root` (the notebook's `docs/audit/data`), and are grouped by evaluated model and
   agent (`claude-sonnet-4-6`, `claude-sonnet-4-6-react`, ...): one scan, one results folder
   and one `..._ANALYSIS.md` per group, as the workflow names them.
2. Scanners are the template's `collect_scanners(...)` (the eight defaults, then
   `eval_scanners.py`), with `eval_scanners.VALIDITY_OVERRIDES` applied before
   `scanner_metadata.json` is written.
3. `--dry-run` scans nothing and calls no model: per sample it renders the messages exactly as
   an `llm_scanner` does, estimates tokens and segments for the scanner model, counts the model
   calls and prices them (`dry_run.json`).
4. After a scan: the template's `extract_results.py`; `validity.py` (the template's
   `analyze_validity.py` with the paper's success rule); `known_answers.py` on the same logs;
   `compare.py`; `usage.json`; with `--analysis`, `make_analysis.py`.

Output: `runs/<label>/<group>/` holds the small tables (committed); the raw scan is
`runs/<label>/<group>/scan_id=*/` (not committed: it holds every prompt, transcripts included).

Usage (from the repository root):
    uv run python agent_artefacts/trajectory_analysis/researcharena_inspect/run.py \
        --manifest pilot --logs-root <notebook>/docs/audit/data \
        --model openai/gpt-5-nano --label pilot-gpt-5-nano [--dry-run] [--analysis]
    ... --mock  (a mock scanner model that answers "no" to everything: no spend)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "inspect_scout"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TEMPLATE))

import ra_common as rc  # noqa: E402

RUNS = HERE / "runs"
SKIP_AGENTS = {"null", "replay"}  # no trajectory of their own

# Default scanners that call the scanner model once per segment. filtered_refusal calls none;
# grader_score_mismatch calls one only when a model call follows the agent's last state event,
# which the red-team scorer never makes.
DEFAULT_LLM_SCANNERS = (
    "outcome_summary",
    "external_failure",
    "formatting_failure",
    "reward_hacking_success",
    "ethical_refusal",
    "broken_env",
)

# Characters of rendered transcript per input token, measured on 18 Nov 2026 on
# full-independent-1 (244,693 characters): gpt-5-nano 60,548 input tokens a call, Haiku 4.5
# 74,957 (docs/audit/data/2026-11-18-scout-trial/*/scan_id=*/_summary.json).
CHARS_PER_TOKEN = {"openai/gpt-5-nano": 4.04, "anthropic/claude-haiku-4-5": 3.26}
DEFAULT_CHARS_PER_TOKEN = 3.26  # the lower (more tokens) of the two
# Output tokens a call, same source: gpt-5-nano 26,161 over 8 calls (reasoning included),
# Haiku 3,555 over 6.
OUTPUT_TOKENS_PER_CALL = {"openai/gpt-5-nano": 3300, "anthropic/claude-haiku-4-5": 600}
DEFAULT_OUTPUT_TOKENS = 3300
# Question, answer format and (own scanners) the evaluator's record, in tokens, at most.
TEMPLATE_TOKENS = {"default": 600, "own": 2500}
# USD per million tokens (list prices; check before relying on them): input, cached input
# read, cache write, output.
PRICES = {
    "openai/gpt-5-nano": (0.05, 0.005, 0.05, 0.40),
    "anthropic/claude-haiku-4-5": (1.00, 0.10, 1.25, 5.00),
}


def _model_key(model: str, table: dict[str, Any]) -> str | None:
    for key in table:
        if model == key or model.startswith(key + "-") or model.startswith(key + "@"):
            return key
    return None


# --- logs and groups -------------------------------------------------------------------------


def log_groups(entries: list[rc.LogEntry]) -> dict[str, list[rc.LogEntry]]:
    """Logs by evaluated model and agent; null and replay runs left out."""
    from inspect_ai.log import read_eval_log

    groups: dict[str, list[rc.LogEntry]] = defaultdict(list)
    for entry in entries:
        header = read_eval_log(str(entry.path), header_only=True)
        agent = (header.eval.task_args or {}).get("agent")
        if agent in SKIP_AGENTS:
            print(f"Skipping {entry.rel}: agent {agent!r} has no trajectory of its own")
            continue
        groups[rc.group_name(str(header.eval.model), agent)].append(entry)
    return dict(sorted(groups.items()))


# --- scanners --------------------------------------------------------------------------------


def load_scanners() -> list[Any]:
    """The template's scanner collection for this eval, with VALIDITY_OVERRIDES applied."""
    from run_all_scanners import collect_scanners

    scanners = collect_scanners(rc.EVAL_NAME, [])
    module = sys.modules.get(f"eval_scanners_{rc.EVAL_NAME}")
    overrides: dict[str, dict[str, bool]] = getattr(module, "VALIDITY_OVERRIDES", {})
    unknown = set(overrides) - {s.name for s in scanners}
    if unknown:
        raise SystemExit(f"VALIDITY_OVERRIDES names unknown scanners: {sorted(unknown)}")
    return [dataclasses.replace(s, **overrides.get(s.name, {})) for s in scanners]


def own_scanner_applies(name: str, setting: str, side_task: str) -> bool:
    if name == "side_task_carried_out":
        return not rc.is_baseline(side_task)
    if name == "score_read_back":
        return setting in rc.READ_BACK_SETTINGS
    if name == "submission_not_loadable":
        return setting in rc.POST_TRAINING
    return True


# --- dry run ---------------------------------------------------------------------------------


async def _rendered_chars(messages: list[Any]) -> int:
    from inspect_scout._scanner.extract import message_numbering

    messages_as_str, _ = message_numbering()
    return len(await messages_as_str(messages))


def context_window(model: str) -> int:
    from inspect_ai.model import get_model_info
    from inspect_scout._transcript.messages import DEFAULT_CONTEXT_WINDOW

    info = get_model_info(model)
    return info.input_tokens if info and info.input_tokens else DEFAULT_CONTEXT_WINDOW


def dry_run(groups: dict[str, list[rc.LogEntry]], scanners: list[Any], model: str) -> dict:
    """What a scan would do, with no model call: per sample, tokens, segments and calls."""
    from inspect_ai.log import read_eval_log
    from inspect_scout._transcript.messages import _TOKENIZER_SAFETY_MARGIN

    names = [s.name for s in scanners]
    own = [n for n in names if n not in DEFAULT_LLM_SCANNERS and n not in rc_default_rule_based()]
    ratio = CHARS_PER_TOKEN.get(_model_key(model, CHARS_PER_TOKEN) or "", DEFAULT_CHARS_PER_TOKEN)
    window = context_window(model)
    budgets = {
        kind: int(window * (1 - _TOKENIZER_SAFETY_MARGIN)) - tokens
        for kind, tokens in TEMPLATE_TOKENS.items()
    }
    samples = []
    for group, entries in groups.items():
        for entry in entries:
            log = read_eval_log(str(entry.path))
            for s in log.samples or []:
                chars = asyncio.run(_rendered_chars(s.messages))
                own_messages, added = rc.scanner_messages(s, str(log.eval.model))
                own_chars = asyncio.run(_rendered_chars(own_messages)) if added else chars
                tokens = math.ceil(chars / ratio)
                own_tokens = math.ceil(own_chars / ratio)
                setting, side = s.metadata.get("setting", ""), s.metadata.get("side_task", "")
                seg_default = max(1, math.ceil(tokens / budgets["default"]))
                seg_own = max(1, math.ceil(own_tokens / budgets["own"]))
                default_calls = len([n for n in DEFAULT_LLM_SCANNERS if n in names]) * seg_default
                default_calls += 1 if seg_default > 1 else 0  # outcome_summary's reducer call
                own_used = [n for n in own if own_scanner_applies(n, setting, side)]
                own_calls = len(own_used) * seg_own
                input_tokens = default_calls * (tokens / seg_default + TEMPLATE_TOKENS["default"])
                input_tokens += own_calls * (own_tokens / seg_own + TEMPLATE_TOKENS["own"])
                # A prompt prefix is cached after its first call: the default view once, the own
                # view once more when it differs, and each own scanner's question.
                uncached = tokens + (own_tokens if added else 0)
                uncached += TEMPLATE_TOKENS["own"] * len(own_used) * seg_own
                samples.append(
                    {
                        "group": group,
                        "run": entry.run,
                        "log": entry.rel,
                        "setting": setting,
                        "side_task": side,
                        "messages": len(s.messages),
                        "rendered_chars": chars,
                        "est_tokens": tokens,
                        "segments": seg_default,
                        "earlier_conversations_added": added,
                        "own_view_chars": own_chars,
                        "own_view_tokens": own_tokens,
                        "own_segments": seg_own,
                        "own_scanners": own_used,
                        "llm_calls": default_calls + own_calls,
                        "est_input_tokens": round(input_tokens),
                        "uncached_input_tokens": round(uncached),
                    }
                )
    calls = sum(s["llm_calls"] for s in samples)
    total_in = sum(s["est_input_tokens"] for s in samples)
    uncached = sum(s["uncached_input_tokens"] for s in samples)
    out_per_call = OUTPUT_TOKENS_PER_CALL.get(
        _model_key(model, OUTPUT_TOKENS_PER_CALL) or "", DEFAULT_OUTPUT_TOKENS
    )
    output = calls * out_per_call
    cost = None
    key = _model_key(model, PRICES)
    if key:
        p_in, p_read, p_write, p_out = PRICES[key]
        cached = total_in - uncached
        low = (uncached * p_write + cached * p_read + output * p_out) / 1e6
        high = (total_in * p_in + output * p_out) / 1e6
        cost = {"low_usd": round(low, 2), "high_usd": round(high, 2)}
    return {
        "model": model,
        "context_window": window,
        "chars_per_token": ratio,
        "scanners": names,
        "samples": samples,
        "totals": {
            "logs": len({s["log"] for s in samples}),
            "samples": len(samples),
            "llm_calls": calls,
            "est_input_tokens": total_in,
            "est_uncached_input_tokens": uncached,
            "est_output_tokens": output,
            "multi_segment_samples": sum(
                s["segments"] > 1 or s["own_segments"] > 1 for s in samples
            ),
            "samples_with_earlier_conversations": sum(
                s["earlier_conversations_added"] > 0 for s in samples
            ),
            "est_cost": cost,
        },
    }


def rc_default_rule_based() -> set[str]:
    return {"filtered_refusal", "grader_score_mismatch"}


def print_dry_run(plan: dict) -> None:
    print(f"\nScanner model {plan['model']}: context {plan['context_window']:,} tokens, "
          f"{plan['chars_per_token']} characters a token (measured, see CHARS_PER_TOKEN)")
    print(f"{'run':<24}{'setting':<24}{'side':<16}{'msgs':>5}{'tokens':>9}{'own view':>10}"
          f"{'segs':>6}{'calls':>6}{'input tok':>12}")
    for s in plan["samples"]:
        print(f"{s['run']:<24}{s['setting']:<24}{s['side_task']:<16}{s['messages']:>5}"
              f"{s['est_tokens']:>9,}{s['own_view_tokens']:>10,}"
              f"{str(s['segments']) + '/' + str(s['own_segments']):>6}"
              f"{s['llm_calls']:>6}{s['est_input_tokens']:>12,}")
    t = plan["totals"]
    print(f"\n{t['logs']} logs, {t['samples']} samples, {t['llm_calls']} model calls; about "
          f"{t['est_input_tokens']:,} input tokens ({t['est_uncached_input_tokens']:,} not "
          f"cacheable), {t['est_output_tokens']:,} output; {t['multi_segment_samples']} samples "
          f"need more than one segment; {t['samples_with_earlier_conversations']} samples get "
          f"their pre-compaction conversation back for this eval's scanners ('own view').")
    if t["est_cost"]:
        print(f"Estimated cost: ${t['est_cost']['low_usd']} with prompt caching, "
              f"${t['est_cost']['high_usd']} without (list prices in PRICES).")
    print("Dry run: nothing scanned, no model called.")


# --- scan and post-processing ----------------------------------------------------------------


def mock_model() -> Any:
    """A scanner model that answers every question with 'no', for a run with no spend."""
    from inspect_ai.model import ModelOutput, get_model

    def judge(input: Any, tools: Any, tool_choice: Any, config: Any) -> ModelOutput:
        return ModelOutput.from_content(
            model="mockllm/model",
            content="Mock scanner model: no transcript was read.\n\nANSWER: no",
        )

    return get_model("mockllm/model", custom_outputs=judge)


def run_scan(group_dir: Path, entries: list[rc.LogEntry], scanners: list[Any], model: Any,
             mock: bool) -> None:
    from inspect_scout import scan, transcripts_from
    from run_all_scanners import save_scanner_metadata

    if list(group_dir.glob("scan_id=*")):
        raise SystemExit(f"{group_dir} already holds a scan; use a new --label, or --post-only")
    save_scanner_metadata(scanners, group_dir)
    status = scan(
        scanners={s.name: s.create_scanner() for s in scanners},
        transcripts=transcripts_from([str(e.path) for e in entries]),
        model=model,
        scans=str(group_dir),
        max_processes=1 if mock else None,
        display="none" if mock else None,
    )
    print(f"Scan of {len(entries)} logs complete: {status.complete}")


def run_template_script(name: str, results_dir: Path) -> None:
    """One of the template's scripts, unchanged, on `results_dir`."""
    subprocess.run(
        [sys.executable, str(TEMPLATE / name), str(results_dir)], check=True, cwd=TEMPLATE
    )


def usage(results_dir: Path) -> dict:
    from utils import find_scan_dir

    scan_dir = find_scan_dir(results_dir)
    summary = json.loads((scan_dir / "_summary.json").read_text())
    per_scanner: dict[str, dict[str, Any]] = {}
    totals: dict[str, int] = defaultdict(int)
    for name, info in (summary.get("scanners") or {}).items():
        merged: dict[str, int] = defaultdict(int)
        for usage_row in (info.get("model_usage") or {}).values():
            for k, v in usage_row.items():
                if isinstance(v, int):
                    merged[k] += v
                    totals[k] += v
        per_scanner[name] = {
            "scans": info.get("scans"),
            "results": info.get("results"),
            "errors": info.get("errors"),
            "tokens": info.get("tokens"),
            **merged,
        }
    errors_file = scan_dir / "_errors.jsonl"
    errors = errors_file.read_text().count("\n") if errors_file.exists() else 0
    return {
        "scan_id": scan_dir.name.removeprefix("scan_id="),
        "complete": summary.get("complete"),
        "error_lines": errors,
        "totals": dict(totals),
        "scanners": per_scanner,
    }


def post_process(group_dir: Path, entries: list[rc.LogEntry], model_name: str, label: str,
                 analysis: bool, mock: bool, manifest: str) -> None:
    import compare
    import known_answers
    import validity

    run_template_script("extract_results.py", group_dir)
    validity.main([str(group_dir)])
    known_csv = group_dir / "known_answers.csv"
    known_answers.write_csv(known_answers.rows_for_logs(entries), known_csv)
    compare.main([str(group_dir), str(known_csv)])
    info = usage(group_dir)
    info.update(
        {
            "label": label,
            "manifest": manifest,
            "scanner_model": model_name,
            "mock": mock,
            "logs": [e.rel for e in entries],
            "post_processed": datetime.now(UTC).isoformat(timespec="seconds"),
            "commit": _git_commit(),
        }
    )
    (group_dir / "usage.json").write_text(json.dumps(info, indent=2) + "\n")
    _relative_paths(group_dir)
    if analysis:
        import make_analysis

        make_analysis.main([str(group_dir)] + (["--out-dir", str(group_dir)] if mock else []))


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _relative_paths(group_dir: Path) -> None:
    """Drop this machine's absolute paths from the text outputs that are committed."""
    root = str(HERE.parents[2]) + "/"
    for name in ("results_summary.txt", "outcome_summaries.txt", "validity_analysis.txt"):
        path = group_dir / name
        if path.exists():
            path.write_text(path.read_text().replace(root, ""))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--manifest", default="pilot", help="pilot, full, or a manifest path")
    parser.add_argument("--logs-root", required=True, type=Path,
                        help="the folder the manifest's globs are relative to (docs/audit/data)")
    parser.add_argument("--model", default="openai/gpt-5-nano", help="scanner model")
    parser.add_argument("--label", help="results folder under runs/ (default: <manifest>-<model>)")
    parser.add_argument("--group", action="append", help="only these groups (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="count and price, scan nothing")
    parser.add_argument("--mock", action="store_true", help="mock scanner model, no spend")
    parser.add_argument("--post-only", action="store_true", help="redo the steps after the scan")
    parser.add_argument("--analysis", action="store_true", help="write the ..._ANALYSIS.md files")
    parser.add_argument("--runs-dir", type=Path, default=RUNS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    manifest = rc.manifest_path(args.manifest)
    entries, empty = rc.resolve_logs(rc.read_manifest(manifest), args.logs_root)
    for pattern in empty:
        print(f"Note: no log matches {pattern}")
    groups = log_groups(entries)
    if args.group:
        groups = {g: e for g, e in groups.items() if g in args.group}
    if not groups:
        raise SystemExit("no logs to scan")
    model_name = "mockllm/model" if args.mock else args.model
    manifest_name = Path(args.manifest).stem
    label = args.label or (
        f"mock-{manifest_name}" if args.mock else f"{manifest_name}-{rc.model_slug(model_name)}"
    )
    if args.mock and not label.startswith("mock"):
        label = f"mock-{label}"
    for group, group_entries in groups.items():
        print(f"Group {group}: {len(group_entries)} logs")

    scanners = load_scanners()
    print(f"Scanners ({len(scanners)}): {[s.name for s in scanners]}")
    label_dir = args.runs_dir / label
    if args.dry_run:
        plan = dry_run(groups, scanners, args.model)
        print_dry_run(plan)
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / "dry_run.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Plan written to {label_dir / 'dry_run.json'}")
        return

    model = mock_model() if args.mock else args.model
    for group, group_entries in groups.items():
        group_dir = label_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        if not args.post_only:
            run_scan(group_dir, group_entries, scanners, model, args.mock)
        post_process(group_dir, group_entries, model_name, label, args.analysis, args.mock,
                     manifest_name)
        print(f"Results for {group}: {group_dir}")


if __name__ == "__main__":
    main()
