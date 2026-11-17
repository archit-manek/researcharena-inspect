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
