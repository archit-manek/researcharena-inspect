"""The authors' published runs: the website's index and the files on Hugging Face.

Each index entry is one red-team run, keyed by setting, side task, agent and timestamp, with
the published main and side scores and one suspicion score per monitor configuration.
"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_evals_utils.huggingface import hf_hub_download

from researcharena_inspect.pins import HF_DATASET, HF_REVISION, INDEX_PATH, INDEX_SHA256

SETTINGS = ("cuda_kernel", "inference_optimization", "safety_finetuning", "utility_finetuning")


def load_index(path: Path = INDEX_PATH, sha256: str | None = INDEX_SHA256) -> dict[str, Any]:
    """Load the pinned website index, refusing a file that doesn't match the pinned hash."""
    raw = path.read_bytes()
    data = gzip.decompress(raw) if path.suffix == ".gz" else raw
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(
            f"{path} does not match the pinned sha256; re-pin it on purpose in pins.py"
        )
    return json.loads(data)


def run_to_sample(run: dict[str, Any], prompt: str = "") -> Sample:
    """One published run as a Sample. `prompt` is filled in by the task that uses the run."""
    return Sample(
        id=run["id"], input=prompt, metadata={**run, "excluded": "excluded_reason" in run}
    )


def runs_dataset(
    settings: tuple[str, ...] = SETTINGS,
    include_excluded: bool = True,
    path: Path = INDEX_PATH,
) -> MemoryDataset:
    """All published runs in `settings`.

    Runs the website excludes for unprompted sabotage are kept by default and flagged in
    metadata (`excluded`, `excluded_reason`).
    """
    runs = [
        run
        for run in load_index(path)["runs"]
        if run["setting"] in settings and (include_excluded or "excluded_reason" not in run)
    ]
    return MemoryDataset([run_to_sample(run) for run in runs], name="researcharena_published_runs")


def fetch_run_file(run_id: str, relative_path: str) -> Path:
    """Download one file of a published run at the pinned revision; cached by huggingface_hub.

    Args:
        run_id: the run's `id` in the index, for example "cuda_kernel/cuda_kernel__baseline__...".
        relative_path: path inside the run's folder, for example "red_team/solve_parsed_sanitized.txt".
    """
    return Path(
        hf_hub_download(
            repo_id=HF_DATASET,
            filename=f"{run_id}/{relative_path}",
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )
