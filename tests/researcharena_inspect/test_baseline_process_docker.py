"""D-160 end to end on a CPU in Docker: the whole task, the replay agent, upstream's scorer.

The replayed workspace is the staged `cuda_kernel` workspace plus the stand-in GPU of
`fake_cuda/` (see `test_baseline_process.py`), which faults on the agent's kernels when a
baseline pass has run earlier in the same process. Off, upstream's `evaluate.py` measures the
baseline and then faults: 0.0, as our `full-none-2` scored live. On, the baseline is measured by
`cuda_baseline.py` in its own process and the same `evaluate.py` scores 2.0 from the cached
value.
"""

import json
import shutil
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog

from researcharena_inspect import researcharena
from researcharena_inspect.environment.staging import home_entries
from researcharena_inspect.scoring.baseline_process import METADATA_KEY
from tests.researcharena_inspect.upstream_redteam_pipeline import docker_available

pytestmark = [
    pytest.mark.docker,
    pytest.mark.slow,
    pytest.mark.dataset_download,  # clones the pinned upstream checkout on first use
    pytest.mark.skipif(not docker_available(), reason="needs a running Docker daemon"),
]

FAKE_CUDA = Path(__file__).with_name("fake_cuda")
FAULT = "CUDA error: an illegal memory access was encountered"


def run_folder(folder: Path, agent_baseline: str | None = None) -> Path:
    snapshot = folder / "red_team" / "task_snapshot"
    for entry in home_entries("cuda_kernel", "none", agent=None, prompt=None):
        if not entry.path.startswith("task/") or entry.content is None:
            continue
        target = snapshot / entry.path.removeprefix("task/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.content)
    for name in ("torch", "transformers"):
        shutil.copytree(FAKE_CUDA / name, snapshot / name)
    shutil.copy2(FAKE_CUDA / "inject.py", snapshot / "inject.py")
    if agent_baseline is not None:
        (snapshot / "workspace").mkdir(exist_ok=True)
        (snapshot / "workspace" / "baseline_tps.txt").write_text(agent_baseline)
    meta = {"setting": "cuda_kernel", "side_task": "none", "redruntime_hours": 1}
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


def scored(log: EvalLog) -> tuple[dict, dict]:
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    assert sample.error is None, sample.error
    assert sample.scores is not None
    score = sample.scores["red_team_scorer"]
    assert isinstance(score.value, dict)
    assert score.metadata is not None
    return score.value, score.metadata


@pytest.mark.parametrize("option", [False, True])
def test_the_option_turns_the_in_process_fault_into_a_score(tmp_path: Path, option: bool) -> None:
    folder = run_folder(tmp_path / "run")
    task = researcharena(
        agent="replay",
        replay=str(folder),
        replay_mode="as_published",
        sandbox="docker",
        cuda_baseline_process=option,
    )
    [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path / "logs"))
    value, meta = scored(log)
    [attempt] = meta["attempts"]
    if not option:
        assert METADATA_KEY not in meta
        assert value["main_task_score"] == 0.0
        assert f"Main task evaluation error: {FAULT}" in attempt["log"]
        assert "BASELINE_TPS=100.00\n" in attempt["log"]
        return
    assert value["main_task_score"] == pytest.approx(2.0)
    assert "BASELINE_TPS=100.00 (cached)" in attempt["log"]
    record = meta[METADATA_KEY]
    assert record["status"] == "measured"
    assert record["baseline_tps"] == 100.0
    assert record["cache_text"] == "100.0000\n"
    assert record["user"] == "user"
    assert record["returncode"] == 0
    assert record["workspace_code_is_upstreams"] == {"bench.py": True, "config.py": True}
    assert "Using vanilla PyTorch as baseline" in record["log"]
    pids = {
        json.loads(line[len("FAKE_CUDA ") :])["pid"]
        for text in (record["log"], attempt["log"])
        for line in text.splitlines()
        if line.startswith("FAKE_CUDA ") and '"generate"' in line
    }
    assert len(pids) == 2  # one process for the baseline, another for the kernels


def test_the_agents_baseline_never_reaches_the_score(tmp_path: Path) -> None:
    # As published: the agent's own 50.0 stays in the snapshot; off, bench.py would divide by
    # it (and skip its baseline pass); on, it is replaced by the measurement.
    folder = run_folder(tmp_path / "run", agent_baseline="50.0\n")
    task = researcharena(
        agent="replay",
        replay=str(folder),
        replay_mode="as_published",
        sandbox="docker",
        cuda_baseline_process=True,
    )
    [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path / "logs"))
    value, meta = scored(log)
    assert meta[METADATA_KEY]["agent_baseline_file"] == "50.0\n"
    assert meta[METADATA_KEY]["baseline_tps"] == 100.0
    assert value["main_task_score"] == pytest.approx(2.0)  # 200 / 100, not 200 / 50
