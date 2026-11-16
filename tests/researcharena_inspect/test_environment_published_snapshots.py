"""The port's staged `task/` against the published runs' final `task_snapshot/` (pass rule 3).

One Claude, one Codex and one OpenCode run per CUDA category (baseline and the four side
tasks). Differences are expected: the snapshot is the agent's folder at the end of the run.
The test checks that each one falls into a class we can explain. No staged file may be
missing, and a staged file may differ only where the agent works (the kernels and
`bindings.cpp`), in `timer.sh`'s `CREATION_DATE`, in the independent side task's
`requirements.txt` with lines added at the end, or as an earlier public upstream version of
a harness file. `published_snapshots.py` prints the full table for the report.
"""

import pytest

from tests.researcharena_inspect.published_snapshots import (
    CATEGORIES,
    compare_runs,
    select_runs,
)

ALLOWED_FOR_STAGED = {
    "edited by the agent (a file it is told to edit)",
    "timer.sh: only CREATION_DATE differs",
    "side task's file, lines appended",
    "harness file: an earlier public upstream version",
}


@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_published_snapshots_differ_only_in_explained_ways() -> None:
    runs = select_runs()
    assert sorted({r["side_task"] for r in runs}) == sorted(CATEGORIES)
    assert len(runs) == 15
    for run, diffs, identical in compare_runs(runs):
        assert identical >= 15, run["id"]
        for d in diffs:
            if d.kind.startswith("made during the run"):
                continue
            assert d.kind in ALLOWED_FOR_STAGED, (run["id"], d)
            if d.kind == "harness file: an earlier public upstream version":
                # The published config.py reads ARENA_SEED; upstream's 2f6523f renamed it.
                assert d.path == "config.py", (run["id"], d)
