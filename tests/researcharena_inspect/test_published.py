from collections import Counter
from pathlib import Path

import pytest

from researcharena_inspect.published import SETTINGS, fetch_run_file, load_index, runs_dataset

# The exclusions are the website's own flags: 9 safety, 3 capabilities, 1 inference.
EXCLUDED_BY_SETTING = {"safety_finetuning": 9, "utility_finetuning": 3, "inference_optimization": 1}


def test_index_matches_pin() -> None:
    assert set(load_index()["settings"]) == set(SETTINGS)


def test_index_hash_is_enforced(tmp_path: Path) -> None:
    bad = tmp_path / "index.json"
    bad.write_text('{"runs": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="pinned sha256"):
        load_index(bad)


def test_all_runs_become_samples_with_unique_ids() -> None:
    samples = list(runs_dataset())
    ids = [sample.id for sample in samples]
    assert len(ids) == 386
    assert len(set(ids)) == len(ids)
    assert {sample.metadata["setting"] for sample in samples if sample.metadata} == set(SETTINGS)


def test_excluded_runs_are_flagged_not_dropped() -> None:
    samples = list(runs_dataset())
    excluded = Counter(
        s.metadata["setting"] for s in samples if s.metadata and s.metadata["excluded"]
    )
    assert excluded == EXCLUDED_BY_SETTING
    assert len(runs_dataset(include_excluded=False)) == 386 - sum(EXCLUDED_BY_SETTING.values())


def test_every_run_has_published_monitor_scores() -> None:
    for sample in runs_dataset():
        assert sample.metadata is not None
        assert sample.metadata["monitors"], sample.id
        assert all("dir" in m and "susp" in m for m in sample.metadata["monitors"]), sample.id


@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_fetch_run_file_reads_a_published_transcript() -> None:
    run_id = "cuda_kernel/cuda_kernel__baseline__codex__gpt-5.5__20260605_114453_c17313477"
    transcript = fetch_run_file(run_id, "red_team/solve_parsed_sanitized.txt")
    assert transcript.stat().st_size == 539198
