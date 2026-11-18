"""Pass rules 2 and 3 on the published runs.

Rule 2: the scorer's `read_metrics` must give the main and side-task values the wave 1 analysis
reads for the same run, for all 113 published CUDA runs, except the two listed in
`published_metrics.KNOWN_DIFFERENCES` (each explained there).

Rule 3: upstream's side-task evaluators on the published snapshots, where no GPU is needed
(`published_side_tasks`).
"""

from pathlib import Path

import pytest

from tests.researcharena_inspect.published_metrics import (
    KNOWN_DIFFERENCES,
    KNOWN_DIFFERENCES_INFERENCE,
    compare_all,
)
from tests.researcharena_inspect.published_side_tasks import attack_runs, compare, replay_runs
from tests.researcharena_inspect.upstream_redteam_pipeline import docker_available


@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_reading_published_metrics_matches_the_analysis() -> None:
    comparisons = compare_all()
    assert len(comparisons) == 113
    differing = {c.run_id: c.differences for c in comparisons if c.differences}
    assert set(differing) == set(KNOWN_DIFFERENCES)
    # Only the side score differs in those two; main scores and side metrics all agree.
    for run_id, differences in differing.items():
        assert [d.split()[0] for d in differences] == ["side_score"], run_id
    for c in comparisons:
        assert c.ours.side_task == c.theirs.side_task
        assert c.ours.side_task_class == c.theirs.side_task_class
        assert c.ours.agent == c.theirs.agent


@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_reading_published_inference_metrics_matches_the_analysis() -> None:
    # The same for the 105 inference runs: only the side score differs, in the runs listed.
    comparisons = compare_all("inference_optimization")
    assert len(comparisons) == 105
    differing = {c.run_id: c.differences for c in comparisons if c.differences}
    assert set(differing) == set(KNOWN_DIFFERENCES_INFERENCE)
    for run_id, differences in differing.items():
        assert [d.split()[0] for d in differences] == ["side_score"], run_id
    for c in comparisons:
        assert c.ours.side_task == c.theirs.side_task
        assert c.ours.side_task_class == c.theirs.side_task_class
        assert c.ours.agent == c.theirs.agent


@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.huggingface
@pytest.mark.dataset_download
@pytest.mark.skipif(not docker_available(), reason="needs a running Docker daemon")
def test_side_task_evaluators_on_published_snapshots_match(tmp_path: Path) -> None:
    # Pass rule 3: every published attack run, replayed as published on a CPU, gives the
    # published values for the fields that need no GPU.
    runs = attack_runs()
    assert len(runs) == 89
    checks = compare(runs, replay_runs(runs, str(tmp_path)))
    assert [c.run_id for c in checks if c.sample_error] == []
    assert {c.run_id: c.differences for c in checks if c.differences} == {}
    assert {c.our_kl_ratio for c in checks if c.side_task == "embedded"} == {None}
