"""The GPU environment check's solver and scorer, without a GPU.

The Docker test runs the whole check (staging, `pre_agent_setup.sh`, the evaluator's
preparation and upstream's real `evaluate.py`) in a CPU-only Debian container. With no
PyTorch there, upstream's `evaluate.py` takes its own error path for the main task and still
writes `metrics.json`; the GPU parts record their failure, which is what they should do.
"""

import json
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelName
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from researcharena_inspect.environment.check import (
    CHECK_METADATA_KEY,
    evaluation_finished,
    run_environment_check,
)
from tests.researcharena_inspect.upstream_redteam_pipeline import IMAGE, docker_available


def _state(metadata: dict) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s",
        epoch=1,
        input="check",
        messages=[],
        metadata=metadata,
    )


async def test_evaluation_finished_scores_on_metrics_json_alone() -> None:
    score = evaluation_finished()
    done = await score(_state({CHECK_METADATA_KEY: {"metrics_json": '{"x": 1}'}}), Target(""))
    assert done is not None
    assert done.value == CORRECT
    assert done.answer == '{"x": 1}'
    missing = await score(_state({CHECK_METADATA_KEY: {"metrics_json": None}}), Target(""))
    assert missing is not None
    assert missing.value == INCORRECT
    nothing = await score(_state({}), Target(""))
    assert nothing is not None
    assert nothing.value == INCORRECT


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
@pytest.mark.skipif(not docker_available(), reason="needs a running Docker daemon")
def test_environment_check_runs_upstreams_evaluator_on_a_cpu(tmp_path: Path) -> None:
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        f'services:\n  default:\n    image: {IMAGE}\n    command: ["tail", "-f", "/dev/null"]\n'
        "    init: true\n    network_mode: none\n"
    )
    task = Task(
        dataset=MemoryDataset([Sample(id="cpu", input="check")]),
        solver=run_environment_check(build=False),
        scorer=evaluation_finished(),
        sandbox=("docker", str(compose)),
    )
    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path / "logs"), display="none")
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    assert sample.error is None, sample.error
    check = sample.metadata[CHECK_METADATA_KEY]
    steps = {step["step"]: step for step in check["steps"]}

    assert steps["nvidia-smi"]["returncode"] != 0  # no GPU, recorded rather than raised
    assert "task/timer.sh" in steps["staged home"]["stdout"]
    assert "Initial starter kernels" in steps["staged home"]["stdout"]
    assert "WARNING: CUDA check failed" in steps["pre_agent_setup.sh"]["stdout"]
    assert [h["hook"] for h in check["staging"]["hooks"]] == [
        "settings/cuda_kernel/hooks/pre_agent.sh"
    ]
    assert check["evaluate_command"][:2] == ["python3", "evaluate.py"]
    assert steps["evaluate.py attempt 1"]["returncode"] == 0
    # Upstream's main-task fallback cannot import torch here and scores 0.
    assert "Main task evaluation error" in check["evaluate_log"]
    assert json.loads(check["metrics_json"]) == {"main_task_score": 0.0}
    assert sample.scores is not None
    assert sample.scores["evaluation_finished"].value == CORRECT
