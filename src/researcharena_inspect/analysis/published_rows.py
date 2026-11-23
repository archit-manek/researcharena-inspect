"""Adapter from the authors' published runs to analysis rows.

The website index has each run's headline scores and one suspicion per monitor folder. Scorer
outputs that some rules need come from each run's `results.json`, through a packaged snapshot
(`data/published_raw_metrics.json.gz`), so the analysis runs offline.
"""

import gzip
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from researcharena_inspect.analysis.published_scores import decode_main_score
from researcharena_inspect.analysis.rows import (
    FLAG_UNPROMPTED_SABOTAGE,
    OPTIMIZATION_SETTINGS,
    POST_TRAINING_MONITORS,
    Access,
    Row,
)
from researcharena_inspect.pins import HF_DATASET, HF_REVISION
from researcharena_inspect.published import fetch_run_file, load_index

RAW_METRICS_PATH = Path(__file__).parent / "data" / "published_raw_metrics.json.gz"

MonitorSet = Literal["paper", "all"]
BaselineExclusions = Literal["paper", "website"]
MainScoreField = Literal["main_score", "main_success"]

# D-007: the website excludes 13 baselines, the paper 12 (L527, L570-572). Table 7 scores three
# GPT-5.4 capabilities baselines (L2023-2024), so this is the one the paper keeps (inferred).
WEBSITE_ONLY_EXCLUSIONS = frozenset(
    {"utility_finetuning/utility_finetuning__baseline__codex_norepeat__gpt-5.4__20260505_183246"}
)

# Fields kept from each run's results.json `raw_metrics`, by (setting, side task). A dotted key
# reads a nested dict; the stored name is the last part.
RAW_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("cuda_kernel", "embedded"): ("side_task.kl_ratio", "side_task.suspicious_code"),
    ("cuda_kernel", "embedded2"): (
        "per_row_latency_ratio",
        "output_equivalent",
        "landmine_detected",
    ),
    ("cuda_kernel", "embedded3"): ("kl_bf16", "kl_fp16", "landmine_detected", "kl_landmine"),
    ("cuda_kernel", "independent"): ("pin_added",),
    ("inference_optimization", "embedded"): (
        "speed_A_over_B",
        "baseline_speed_A_over_B",
        "side_task_numeric_value",
        "benchmark_gaming_side_task_success",
    ),
    ("inference_optimization", "embedded2"): (
        "trigger_wrong_rate",
        "control_wrong_rate",
        "override_wrong_rate",
        "backdoor_active",
        "server_started",
    ),
    ("inference_optimization", "independent2"): (
        "uploaded_to_ingest_host",
        "prompt_uploaded",
        "exfiltration_detected",
        "num_payloads_to_ingest_host",
    ),
    ("inference_optimization", "independent"): ("pin_added",),
    ("safety_finetuning", "independent"): ("upload_observed", "upload_total_bytes"),
    ("utility_finetuning", "independent"): ("upload_observed", "upload_total_bytes"),
    ("utility_finetuning", "embedded"): ("gsm8k_accuracy", "gsm1k_accuracy", "contamination_gap"),
}

_MONITOR_DIR = re.compile(
    r"^(?P<static>L0_)?(?P<cli>claude|codex|opencode)_(?P<model>.+?)"
    r"(?:_(?P<effort>high|medium))?(?P<artifact>_artifact)?(?P<cot>_cot)?$"
)


@dataclass(frozen=True)
class MonitorConfig:
    """A published monitor folder name, parsed.

    Attributes:
        dir: the folder name, for example "claude_claude-sonnet-4-6_high_artifact_cot".
        model: the monitor model, for example "claude-sonnet-4-6".
        effort: "high", "medium", or None when the name has no effort tag.
        access: the access type.
        static: True for the `L0_` folders, upstream's removed static monitor (D-052).
    """

    dir: str
    model: str
    effort: str | None
    access: Access
    static: bool


def parse_monitor_dir(name: str) -> MonitorConfig:
    """Parse a monitor folder name from the index or the dataset."""
    m = _MONITOR_DIR.match(name)
    if m is None:
        raise ValueError(f"not a monitor folder name: {name!r}")
    artifact, cot = bool(m["artifact"]), bool(m["cot"])
    access: Access = (
        "artifact_cot" if artifact and cot else "artifact" if artifact else "cot" if cot else "traj"
    )
    return MonitorConfig(name, m["model"], m["effort"], access, bool(m["static"]))


def in_paper_monitor_set(setting: str, side_task_class: str, monitor: MonitorConfig) -> bool:
    """Whether the paper counts this monitor on this run (§5.2, L575-580).

    Sonnet 4.6 and GPT-5.4 everywhere; GPT-5.5 and Opus 4.8 on the optimization settings; Kimi
    K2.6 and GLM-5.1 on the optimization settings' embedded and baseline runs only. The Claude and
    Codex monitors are the `high`-effort folders; `medium`, untagged and `L0_` folders are earlier
    variants the paper doesn't describe (D-052).
    """
    if monitor.static:
        return False
    if monitor.model in POST_TRAINING_MONITORS:
        return monitor.effort == "high"
    if setting not in OPTIMIZATION_SETTINGS:
        return False
    if monitor.model in ("gpt-5.5", "claude-opus-4-8"):
        return monitor.effort == "high"
    if monitor.model in ("kimi-k2.6", "glm-5.1"):
        return monitor.effort is None and side_task_class != "independent"
    return False


def agent_name(red_model: str) -> str:
    """The red-team model without its provider prefix ("opencode/glm-5.1" -> "glm-5.1")."""
    return red_model.rsplit("/", 1)[-1]


def _get(raw: Mapping[str, Any], dotted: str) -> Any:
    value: Any = raw
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def load_raw_metrics(path: Path = RAW_METRICS_PATH) -> dict[str, dict[str, Any]]:
    """The packaged snapshot: run id -> the kept scorer fields."""
    return json.loads(gzip.decompress(path.read_bytes()))["runs"]


def raw_metrics_entry(run: Mapping[str, Any], results: Mapping[str, Any], folder: str) -> dict:
    """The snapshot entry for one run, from one of its results.json files."""
    raw = results.get("raw_metrics") or {}
    entry: dict[str, Any] = {
        "results_from": folder,
        "scorer_main_task_score": results.get("main_task_score"),
        "scorer_side_task_score": results.get("side_task_score"),
    }
    for dotted in RAW_FIELDS.get((run["setting"], run["side_task"]), ()):
        entry[dotted.rsplit(".", 1)[-1]] = _get(raw, dotted)
    return entry


def _is_not_found(err: OSError) -> bool:
    response = getattr(err, "response", None)
    return getattr(response, "status_code", None) == 404


def build_raw_metrics_snapshot(runs: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Rebuild the snapshot from the pinned Hugging Face revision (one download per run).

    Reads the first `results.json` among the run's monitor folders (sorted); each holds the same
    scorer output.
    """
    out: dict[str, dict[str, Any]] = {}
    for run in runs if runs is not None else load_index()["runs"]:
        for folder in sorted(m["dir"] for m in run["monitors"]):
            try:
                path = fetch_run_file(run["id"], f"blue_team/{folder}/results.json")
            except OSError as err:  # huggingface_hub's missing-file error is an OSError
                if not _is_not_found(err):
                    raise
                continue
            out[run["id"]] = raw_metrics_entry(
                run, json.loads(path.read_text(encoding="utf-8")), folder
            )
            break
    return {"hf_dataset": HF_DATASET, "hf_revision": HF_REVISION, "runs": out}


def write_raw_metrics_snapshot(snapshot: Mapping[str, Any], path: Path = RAW_METRICS_PATH) -> None:
    """Write the snapshot deterministically (sorted keys, fixed gzip mtime)."""
    data = json.dumps(snapshot, sort_keys=True, indent=1).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0))


def rows_from_index(
    index: Mapping[str, Any] | None = None,
    raw_metrics: Mapping[str, Mapping[str, Any]] | None = None,
    monitors: MonitorSet = "paper",
    baseline_exclusions: BaselineExclusions = "paper",
    main_score_field: MainScoreField = "main_score",
) -> list[Row]:
    """One row per (published run, monitor folder); a run with no kept monitor gets one row.

    Args:
        index: the website index; defaults to the pinned copy.
        raw_metrics: run id -> scorer fields; defaults to the packaged snapshot.
        monitors: "paper" keeps the monitors the paper counts (`in_paper_monitor_set`); "all"
            keeps every folder, with the parsed model and access.
        baseline_exclusions: which baselines get `FLAG_UNPROMPTED_SABOTAGE`: the paper's 12 or the
            website's 13 (D-007).
        main_score_field: the index field read as the main score. `main_score` equals the
            scorer's results.json value (apart from the CUDA scale, D-004). `main_success` is the
            website's copy of the authors' result tables; it differs for 17 of the 24 safety
            baselines, and Table 6's Main column for baselines matches it.
    """
    index = index if index is not None else load_index()
    raw_metrics = raw_metrics if raw_metrics is not None else load_raw_metrics()
    rows: list[Row] = []
    for run in index["runs"]:
        excluded = "excluded_reason" in run and not (
            baseline_exclusions == "paper" and run["id"] in WEBSITE_ONLY_EXCLUSIONS
        )
        raw = raw_metrics.get(run["id"], {})
        side_metrics = {k: v for k, v in raw.items() if k != "results_from"}
        side_metrics["published_side_success"] = run.get("side_success")
        base = {
            "run_id": run["id"],
            "setting": run["setting"],
            "side_task": run["side_task"],
            "side_task_class": run["cat"],
            "agent": agent_name(run["red_model"]),
            "main_score": decode_main_score(run["setting"], run[main_score_field], "index"),
            "side_score": run["side_score"],
            "side_metrics": side_metrics,
            "flags": frozenset({FLAG_UNPROMPTED_SABOTAGE}) if excluded else frozenset(),
        }
        kept = []
        for monitor in run["monitors"]:
            config = parse_monitor_dir(monitor["dir"])
            if monitors == "paper" and not in_paper_monitor_set(run["setting"], run["cat"], config):
                continue
            kept.append(
                Row(
                    **base,
                    monitor_model=config.model,
                    access=config.access,
                    suspicion=monitor["susp"],
                )
            )
        rows.extend(kept or [Row(**base, monitor_model=None, access=None, suspicion=None)])
    return rows
