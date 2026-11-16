"""Pass rule 2: the scorer's reader on every published CUDA run's `metrics.json`.

The scorer's `read_metrics` must give the main and side-task values the wave 1 analysis reads
for the same run, for all 113 published CUDA runs, except the two listed in
`published_metrics.KNOWN_DIFFERENCES` (each explained there).
"""

import pytest

from tests.researcharena_inspect.published_metrics import KNOWN_DIFFERENCES, compare_all


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
