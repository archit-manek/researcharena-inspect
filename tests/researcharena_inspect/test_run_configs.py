"""The run configs in `run_configs/` name this package's tasks and only their parameters.

They are read by `inspect eval --run-config`, which refuses unknown top-level fields but passes
task arguments through, so a renamed task parameter would only be dropped with a warning.
"""

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

import researcharena_inspect
from researcharena_inspect import researcharena

RUN_CONFIGS = Path(researcharena_inspect.__file__).parent / "run_configs"

# The top-level fields of Inspect's run config file (`inspect eval --run-config`).
RUN_CONFIG_FIELDS = {
    "task",
    "model",
    "model_roles",
    "generate_config",
    "eval_config",
    "solver",
    "tags",
    "metadata",
    "sandbox",
}


def load(name: str) -> dict[str, Any]:
    return yaml.safe_load((RUN_CONFIGS / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(p.name for p in RUN_CONFIGS.glob("*.yaml")))
def test_run_config_names_a_task_and_its_parameters(name: str) -> None:
    config = load(name)
    assert set(config) <= RUN_CONFIG_FIELDS
    namespace, _, task_name = config["task"]["task"].partition("/")
    assert namespace == "researcharena_inspect"
    parameters = inspect.signature(getattr(researcharena_inspect, task_name)).parameters
    assert set(config["task"]["args"]) <= set(parameters)


@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
def test_original_is_the_cuda_baseline_and_four_side_tasks() -> None:
    config = load("original.yaml")
    task = researcharena(**config["task"]["args"])
    assert [s.id for s in task.dataset] == [
        "cuda_kernel__baseline",
        "cuda_kernel__embedded",
        "cuda_kernel__embedded2",
        "cuda_kernel__embedded3",
        "cuda_kernel__independent",
    ]
    assert all(s.metadata and s.metadata["redruntime"] == "1" for s in task.dataset)


@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
def test_original_inference_is_the_inference_baseline_and_four_side_tasks() -> None:
    config = load("original_inference.yaml")
    args = config["task"]["args"]
    task = researcharena(**args, hf_seed="/srv/hf_seed")
    assert [s.id for s in task.dataset] == [
        "inference_optimization__baseline",
        "inference_optimization__embedded",
        "inference_optimization__embedded2",
        "inference_optimization__embedded3",
        "inference_optimization__independent",
    ]
    assert all(s.metadata and s.metadata["redruntime"] == "1" for s in task.dataset)
    assert all(s.metadata and s.metadata["resume_agent"] is True for s in task.dataset)
    assert args["task_config"] == "qwen2_5_3b_gpu"
    assert config["model"] == "anthropic/claude-sonnet-4-6"


@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
def test_original_capabilities_is_the_baseline_and_two_side_tasks_for_five_hours() -> None:
    config = load("original_capabilities.yaml")
    args = config["task"]["args"]
    task = researcharena(**args, hf_seed="/srv/hf_seed")
    assert [s.id for s in task.dataset] == [
        "utility_finetuning__baseline",
        "utility_finetuning__embedded",
        "utility_finetuning__independent",
    ]
    assert all(s.metadata and s.metadata["redruntime"] == "5" for s in task.dataset)
    assert all(s.metadata and s.metadata["resume_agent"] is False for s in task.dataset)
    assert args["task_config"] == "qwen2.5_0.5b"
    assert args["time_limit"] is None
    assert config["model"] == "anthropic/claude-sonnet-4-6"


@pytest.mark.parametrize(
    "name", ["original.yaml", "original_inference.yaml", "original_capabilities.yaml"]
)
def test_red_team_configs_run_one_sample_at_a_time(name: str) -> None:
    # The scores are timings on the machine's one GPU, which every sandbox asks for.
    from inspect_ai._cli.eval import parse_run_config

    config = load(name)
    assert config["eval_config"] == {"max_samples": 1}
    params = parse_run_config(str(RUN_CONFIGS / name))
    assert params["max_samples"] == 1
