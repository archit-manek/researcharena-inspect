"""Run bundles: everything one monitor run needs about one red-team run.

Two loaders build them: `published_bundles()` from the authors' published runs on Hugging Face,
and `local_bundles()` from run folders in upstream's layout (`<run>/metadata.json`,
`<run>/red_team/solve_parsed.txt`, ...), which is how our own runs will be handed to the monitor.
"""

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_evals_utils.huggingface import snapshot_download

from researcharena_inspect.pins import HF_DATASET, HF_REVISION
from researcharena_inspect.published import SETTINGS, fetch_run_file, load_index
from researcharena_inspect.transcripts import parse_cli_output


@dataclass(frozen=True)
class RunBundle:
    """The inputs of one monitor run, independent of the access type."""

    run_id: str
    setting: str
    side_task: str
    """As in the run's `metadata.json`: `none` for a baseline. Fills `{side_task_type}`."""
    task_config: str
    red_provider: str
    red_model: str
    transcript_with_cot: Path
    transcript_without_cot: Path
    workspace: Path | None = None
    """The agent's final workspace (`red_team/task_snapshot`), or None if not available."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """For analysis only, never shown to the monitor: scores, condition, published monitors."""

    def transcript(self, with_cot: bool) -> Path:
        return self.transcript_with_cot if with_cot else self.transcript_without_cot


def _as_list(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def select_published_runs(
    runs: str | Sequence[str] | None = None,
    settings: str | Sequence[str] | None = None,
    side_tasks: str | Sequence[str] | None = None,
    red_providers: str | Sequence[str] | None = None,
    include_excluded: bool = True,
) -> list[dict[str, Any]]:
    """Index entries of the published runs that pass every given filter, in index order.

    Args:
        runs: run ids (comma-separated string or list).
        settings: settings, e.g. `cuda_kernel`.
        side_tasks: the index's side-task names; `baseline` for runs without one.
        red_providers: red-team providers as in the index, e.g. `claude` or `codex_norepeat`.
        include_excluded: keep runs the website excludes for unprompted sabotage.
    """
    wanted_runs = _as_list(runs)
    wanted_settings = _as_list(settings) or list(SETTINGS)
    wanted_side = _as_list(side_tasks)
    wanted_providers = _as_list(red_providers)
    index_runs = load_index()["runs"]
    if wanted_runs is not None:
        known = {run["id"] for run in index_runs}
        unknown = [run_id for run_id in wanted_runs if run_id not in known]
        if unknown:
            raise ValueError(f"not in the published index: {unknown}")
    return [
        run
        for run in index_runs
        if (wanted_runs is None or run["id"] in wanted_runs)
        and run["setting"] in wanted_settings
        and (wanted_side is None or run["side_task"] in wanted_side)
        and (wanted_providers is None or run["red_provider"] in wanted_providers)
        and (include_excluded or "excluded_reason" not in run)
    ]


def fetch_workspace(run_id: str) -> Path:
    """Download a published run's `red_team/task_snapshot/` and return its folder.

    The published snapshots carry no `.git` folder (so artifact monitors replayed on them cannot
    read git history, unlike `pipeline_monitor.sh` at the pinned commit, ledger D-003) and no
    `final_model` (ledger D-008).
    """
    root = snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        revision=HF_REVISION,
        allow_patterns=[f"{run_id}/red_team/task_snapshot/*"],
    )
    return Path(root) / run_id / "red_team" / "task_snapshot"


def published_bundle(run: dict[str, Any], with_workspace: bool = False) -> RunBundle:
    """The bundle of one published run (an entry of `published.load_index()["runs"]`).

    The transcripts are the published, sanitized copies of what the authors' monitors read.
    """
    run_id = run["id"]
    meta = json.loads(fetch_run_file(run_id, "metadata.json").read_text(encoding="utf-8"))
    return RunBundle(
        run_id=run_id,
        setting=meta["setting"],
        side_task=meta["side_task"],
        task_config=meta.get("task_config", ""),
        red_provider=meta.get("red_provider", run["red_provider"]),
        red_model=meta.get("red_model", run["red_model"]),
        transcript_with_cot=fetch_run_file(run_id, "red_team/solve_parsed_sanitized.txt"),
        transcript_without_cot=fetch_run_file(run_id, "red_team/solve_parsed_no_cot_sanitized.txt"),
        workspace=fetch_workspace(run_id) if with_workspace else None,
        metadata={
            "source": "published",
            "run_metadata": meta,
            "side_task_index": run["side_task"],
            "category": run.get("cat"),
            "redruntime_hours": meta.get("redruntime_hours", run.get("hours")),
            "red_time_s": run.get("red_time_s"),
            "main_score": run.get("main_score"),
            "side_score": run.get("side_score"),
            "excluded": "excluded_reason" in run,
            "excluded_reason": run.get("excluded_reason"),
            "published_monitors": run.get("monitors", []),
        },
    )


def published_bundles(
    runs: str | Sequence[str] | None = None,
    settings: str | Sequence[str] | None = None,
    side_tasks: str | Sequence[str] | None = None,
    red_providers: str | Sequence[str] | None = None,
    include_excluded: bool = True,
    with_workspace: bool = False,
) -> list[RunBundle]:
    """Bundles for the published runs that pass the filters (see `select_published_runs`).

    Downloads each run's metadata and two transcripts, and its workspace if `with_workspace`.
    """
    selected = select_published_runs(runs, settings, side_tasks, red_providers, include_excluded)
    return [published_bundle(run, with_workspace) for run in selected]


def _first_existing(folder: Path, names: Iterable[str]) -> Path | None:
    return next((folder / n for n in names if (folder / n).is_file()), None)


def local_bundle(run_dir: Path) -> RunBundle:
    """The bundle of one run folder in upstream's layout.

    Reads `metadata.json`, `red_team/solve_parsed.txt` and `red_team/solve_parsed_no_cot.txt`
    (or their `_sanitized` copies), `red_team/task_snapshot/` and `eval/metrics.json` when present.
    A missing transcript is made from `red_team/solve_out.json` with upstream's parser, as
    `pipeline_monitor.sh:131-149` backfills the no-CoT one.
    """
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    red = run_dir / "red_team"
    cot = _first_existing(red, ("solve_parsed.txt", "solve_parsed_sanitized.txt"))
    no_cot = _first_existing(red, ("solve_parsed_no_cot.txt", "solve_parsed_no_cot_sanitized.txt"))
    if cot is None or no_cot is None:
        raw = _first_existing(red, ("solve_out.json", "solve_out_sanitized.json"))
        if raw is None:
            raise FileNotFoundError(f"{run_dir}: no transcript and no solve_out.json")
        parsed = parse_cli_output(raw, meta["red_provider"])
        if cot is None:
            cot = red / "solve_parsed.txt"
            cot.write_text(parsed.with_cot, encoding="utf-8")
        if no_cot is None:
            no_cot = red / "solve_parsed_no_cot.txt"
            no_cot.write_text(parsed.without_cot, encoding="utf-8")
    metrics_path = run_dir / "eval" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    workspace = red / "task_snapshot"
    return RunBundle(
        run_id=run_dir.name,
        setting=meta["setting"],
        side_task=meta["side_task"],
        task_config=meta.get("task_config", ""),
        red_provider=meta.get("red_provider", ""),
        red_model=meta.get("red_model", ""),
        transcript_with_cot=cot,
        transcript_without_cot=no_cot,
        workspace=workspace if workspace.is_dir() else None,
        metadata={
            "source": str(run_dir),
            "run_metadata": meta,
            "redruntime_hours": meta.get("redruntime_hours"),
            "main_score": metrics.get("main_task_score"),
            "side_score": metrics.get("side_task_score"),
            "published_monitors": [],
        },
    )


def local_bundles(root: str | os.PathLike[str]) -> list[RunBundle]:
    """Bundles for a run folder, or for every run folder below `root` (any depth), sorted."""
    root = Path(root)
    if (root / "metadata.json").is_file():
        return [local_bundle(root)]
    run_dirs = sorted(
        p.parent for p in root.rglob("metadata.json") if (p.parent / "red_team").is_dir()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no run folders (metadata.json + red_team/) under {root}")
    return [local_bundle(d) for d in run_dirs]
