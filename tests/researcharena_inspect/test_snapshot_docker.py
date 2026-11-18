"""The final workspace from a red-team run to an artifact monitor's sandbox, with no model.

A `null`-agent run in the CPU-only Docker sandbox is scored by upstream's `evaluate.py`; a
second scorer then describes `red_team/task_snapshot` in the same sandbox, as the run left it.
From the `.eval` log alone, `write_run_dirs` rebuilds the run folder, `local_bundles` reads it,
`bundle_to_sample` makes the `artifact` monitor's sample, and the monitor's real staging solver
writes its files into a second Docker sandbox. The monitor's `task/` must hold every file of
the snapshot with the same bytes and mode, and the same git repository.

The same run without the snapshot (`keep_snapshot=False`) gives the log's size without it.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from researcharena_inspect import researcharena
from researcharena_inspect.agents import write_run_dirs
from researcharena_inspect.environment import AGENT_USER, SNAPSHOT_DIR, evaluator_sandbox
from researcharena_inspect.monitor import (
    SANDBOX_TASK_DIR,
    bundle_to_sample,
    final_tree,
    local_bundles,
    sample_staged_files,
    stage_monitor_inputs,
)
from researcharena_inspect.scoring import red_team_scorer
from tests.researcharena_inspect.upstream_redteam_pipeline import (
    IMAGE,
    MANIFEST_SOURCE,
    docker_available,
)

pytestmark = [
    pytest.mark.docker,
    pytest.mark.slow,
    pytest.mark.dataset_download,  # clones the pinned upstream checkout on first use
    pytest.mark.skipif(not docker_available(), reason="needs a running Docker daemon"),
]

COMPOSE = f"""services:
  default:
    image: {IMAGE}
    command: ["tail", "-f", "/dev/null"]
    init: true
    network_mode: none
"""


async def _manifest(root: str, user: str) -> dict[str, Any]:
    """Described as the files' owner: git refuses a repository another user owns.

    In the evaluator's container where there is one: the agent's is stopped by scoring (D-093).
    """
    box = evaluator_sandbox()
    await box.write_file("/opt/test/tree_manifest.py", MANIFEST_SOURCE.read_text())
    assert (await box.exec(["chmod", "-R", "a+rX", "/opt/test"], user="root")).success
    result = await box.exec(["python3", "/opt/test/tree_manifest.py", root], user=user)
    assert result.success, result.stderr
    return json.loads(result.stdout)


@scorer(metrics=[])
def describe_snapshot() -> Scorer:
    """After `red_team_scorer`: the snapshot as the run left it, in the evaluator's container."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0, metadata={"manifest": await _manifest(SNAPSHOT_DIR, AGENT_USER)})

    return score


@solver
def describe_monitor_task() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["monitor_view"] = await _manifest(SANDBOX_TASK_DIR, "root")
        return state

    return solve


def _red_team_log(folder: Path, keep_snapshot: bool) -> tuple[EvalLog, Path]:
    task = researcharena(
        side_tasks="none",
        agent="null",
        sandbox="docker",
        scorer=[
            red_team_scorer(pause_seconds=0, keep_snapshot=keep_snapshot),
            describe_snapshot(),
        ],
    )
    [log] = eval(task, model="mockllm/model", log_dir=str(folder), display="none")
    assert log.status == "success", log.error
    [path] = folder.glob("*.eval")
    return log, path


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("snapshot")
    kept_log, kept_path = _red_team_log(root / "kept", keep_snapshot=True)
    off_log, off_path = _red_team_log(root / "off", keep_snapshot=False)
    return {
        "root": root,
        "kept": kept_log,
        "kept_bytes": kept_path.stat().st_size,
        "off": off_log,
        "off_bytes": off_path.stat().st_size,
    }


def _files(manifest: dict[str, Any]) -> dict[str, list[Any]]:
    return {p: e for p, e in manifest["entries"].items() if e[0] == "file"}


def test_artifact_monitor_sees_the_snapshot(runs: dict[str, Any]) -> None:
    log: EvalLog = runs["kept"]
    assert log.samples is not None
    [sample] = log.samples
    assert sample.error is None, sample.error
    assert sample.scores is not None
    snapshot = sample.scores["describe_snapshot"].metadata["manifest"]  # type: ignore[index]
    record = sample.metadata["task_snapshot"]
    assert record["status"] == "kept", record
    print(
        f"\nsnapshot: {record['files']} files, {record['raw_bytes']} bytes raw, "
        f"{record['archive_bytes']} bytes gzipped, {record['seconds']} s; "
        f"log {runs['kept_bytes']} bytes with it, {runs['off_bytes']} bytes without"
    )

    # The run folder from the log alone, then the monitor's sample, as the monitor task makes it.
    run_dirs = write_run_dirs(log, runs["root"] / "runs")
    [bundle] = local_bundles(runs["root"] / "runs")
    assert bundle.workspace == run_dirs[0] / "red_team" / "task_snapshot"
    assert not bundle.workspace_links_outside
    monitor_sample = bundle_to_sample(bundle, "artifact")
    assert monitor_sample.metadata is not None
    assert monitor_sample.metadata["with_artifact"] is True

    # Upstream writes the transcript and the instruction first; the workspace overwrites them
    # only if it has files of those names (D-019).
    tree = final_tree(sample_staged_files(monitor_sample.metadata))
    staged_task = {p.removeprefix("task/") for p in tree if p.startswith("task/")}
    assert {"agent_transcript.txt", "task_instruction.md"} <= staged_task
    assert not {"agent_transcript.txt", "task_instruction.md"} & set(_files(snapshot))

    # The monitor's real staging, in its own sandbox.
    monitor_task = Task(
        dataset=MemoryDataset([monitor_sample]),
        setup=stage_monitor_inputs(),
        solver=describe_monitor_task(),
        sandbox=("docker", _compose(runs["root"])),
    )
    [monitor_log] = eval(
        monitor_task, model="mockllm/model", log_dir=str(runs["root"] / "monitor"), display="none"
    )
    assert monitor_log.status == "success", monitor_log.error
    assert monitor_log.samples is not None
    view = monitor_log.samples[0].metadata["monitor_view"]

    seen = _files(view)
    for path in ("agent_transcript.txt", "task_instruction.md"):
        del seen[path]
    assert seen == _files(snapshot)
    assert view["git"].keys() == snapshot["git"].keys() == {"."}
    staged_by_monitor = {f"?? {name}" for name in ("agent_transcript.txt", "task_instruction.md")}
    for key, theirs in snapshot["git"]["."].items():
        ours = view["git"]["."][key]
        if key == "status":  # the monitor's two files are untracked in its copy
            assert staged_by_monitor <= set(ours)
            ours = [line for line in ours if line not in staged_by_monitor]
        assert ours == theirs, key
    # The evaluator wrote into the snapshot (Python's byte-code cache of the snapshot's bench.py).
    assert any(p.startswith("__pycache__/bench.") for p in _files(snapshot))


def test_snapshot_off_is_recorded(runs: dict[str, Any]) -> None:
    log: EvalLog = runs["off"]
    assert log.samples is not None
    [sample] = log.samples
    assert sample.metadata["task_snapshot"]["status"] == "off"
    assert runs["off_bytes"] < runs["kept_bytes"]
    [run_dir] = write_run_dirs(log, runs["root"] / "runs_off")
    assert not (run_dir / "red_team" / "task_snapshot").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["inspect_task_snapshot"]["status"] == "off"


def _compose(root: Path) -> str:
    path = root / "monitor_compose.yaml"
    path.write_text(COMPOSE)
    return str(path)
