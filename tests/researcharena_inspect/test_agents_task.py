"""The red-team task and its agents, without Docker or a model."""

import importlib
from collections.abc import AsyncIterator

import pytest
from inspect_ai.util import (
    ComposeConfig,
    ExecCompleted,
    ExecStderr,
    ExecStdout,
    SandboxEnvironmentSpec,
)

from researcharena_inspect import researcharena
from researcharena_inspect.agents import AGENTS, red_team_agent, resolve_agent_effort
from researcharena_inspect.agents.claude_code import (
    capture_cli_output,
    claude_code_env,
    red_team_claude_code,
)
from researcharena_inspect.agents.react import red_team_react
from researcharena_inspect.environment.layout import CONTAINER_ENV, TASK_DIR
from researcharena_inspect.environment.sandbox import sandbox_spec
from researcharena_inspect.prompts import (
    agent_instruction_file,
    agent_prompt,
    list_side_tasks,
    resume_agent,
)
from researcharena_inspect.task import (
    CPU_IMAGE,
    red_team_dataset,
    red_team_sample,
    red_team_sandbox,
)


@pytest.mark.parametrize("side_task", list_side_tasks("cuda_kernel"))
def test_sample_input_is_the_agent_prompt(side_task: str) -> None:
    sample = red_team_sample("cuda_kernel", side_task, "1")
    expected = agent_prompt("cuda_kernel", side_task, "claude", "1")
    assert isinstance(sample.input, str)
    assert sample.input.encode("utf-8") == expected.encode("utf-8")
    assert sample.metadata is not None
    assert sample.metadata["instruction_file"] == agent_instruction_file(
        "cuda_kernel", side_task, "claude", "1"
    )
    assert sample.metadata["redruntime"] == "1"
    assert sample.metadata["resume_agent"] is True
    assert sample.metadata["task_config"] == "tinyllama"
    assert sample.metadata["baseline"] == (side_task == "none")


def test_budget_is_kept_as_typed_in_the_prompt() -> None:
    sample = red_team_sample("cuda_kernel", "none", "0.1667")
    assert isinstance(sample.input, str)
    assert sample.input.endswith("You have approximately 0.1667 hours of wall-clock time.")
    assert sample.metadata is not None
    assert sample.metadata["redruntime"] == "0.1667"


def test_dataset_covers_side_tasks_and_baseline() -> None:
    ids = [s.id for s in red_team_dataset("cuda_kernel", None, "1")]
    assert ids == [
        "cuda_kernel__embedded",
        "cuda_kernel__embedded2",
        "cuda_kernel__embedded3",
        "cuda_kernel__independent",
        "cuda_kernel__baseline",
    ]
    assert [s.id for s in red_team_dataset("cuda_kernel", "independent,none", "1")] == [
        "cuda_kernel__independent",
        "cuda_kernel__baseline",
    ]
    with pytest.raises(ValueError, match="unknown side tasks"):
        red_team_dataset("cuda_kernel", "exfiltrate", "1")


def test_post_training_settings_are_not_resumed() -> None:
    assert resume_agent("safety_finetuning") is False
    sample = red_team_sample("safety_finetuning", "none", "5")
    assert sample.metadata is not None
    assert sample.metadata["resume_agent"] is False


def test_docker_without_a_config_is_the_cpu_image_in_the_agents_folder() -> None:
    # The CPU image for tests; it has git, which staging's hooks need. The task's default is
    # the environment's GPU sandbox.
    spec = red_team_sandbox("docker", None)
    assert isinstance(spec.config, ComposeConfig)
    service = spec.config.services["default"]
    assert service.image == CPU_IMAGE
    # Staging needs an empty /home/user; each agent command sets the workspace as its cwd.
    assert service.working_dir is None
    assert TASK_DIR == "/home/user/task"
    assert service.environment == CONTAINER_ENV
    assert CONTAINER_ENV["HOME"] == "/home/user"
    assert CONTAINER_ENV["PATH"].startswith("/home/user/task:")


def test_sandbox_can_be_injected() -> None:
    assert red_team_sandbox("docker", "compose.yaml") == SandboxEnvironmentSpec(
        "docker", "compose.yaml"
    )
    assert red_team_sandbox("other", None) == SandboxEnvironmentSpec("other", None)
    spec = SandboxEnvironmentSpec("docker", "arena.yaml")
    assert red_team_sandbox(spec, None) is spec


@pytest.mark.parametrize("agent", [a for a in AGENTS if a != "replay"])
def test_task_builds_for_each_agent(agent: str) -> None:
    # Setup and scorer default to the environment's staging and upstream's scorer, and the
    # sandbox to the environment's.
    task = researcharena(agent=agent, side_tasks="none", redruntime="0.1667")
    assert len(task.dataset) == 1
    assert task.setup is not None
    assert task.scorer is not None
    assert len(task.scorer) == 1
    assert task.sandbox == sandbox_spec(setting="cuda_kernel")  # with the evaluator


def test_task_rejects_unknown_agents() -> None:
    with pytest.raises(ValueError, match="agent must be one of"):
        researcharena(agent="codex")
    with pytest.raises(ValueError, match="agent must be one of"):
        red_team_agent(agent="codex")


def test_solver_is_registered() -> None:
    assert callable(red_team_agent(agent="react"))
    assert callable(red_team_agent(agent="claude_code", claude_code_version="2.1.162"))


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-opus-4-8", "model"])
def test_effort_comes_from_effort_json(model: str) -> None:
    # effort.json at the pinned commit: red_team "claude:*" -> xhigh, for any model.
    assert resolve_agent_effort("cuda_kernel", model) == "xhigh"


def test_claude_code_env_is_what_upstream_passes_the_agent() -> None:
    env = claude_code_env("claude-sonnet-4-6", "xhigh", "1", True)
    assert env == {
        "BASH_MAX_TIMEOUT_MS": "36000000",
        "BASH_DEFAULT_TIMEOUT_MS": "3600000",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "0",
        "AGENT_CONFIG": "claude-sonnet-4-6",
        "AGENT_EFFORT": "xhigh",
        "TIME_BUDGET_HOURS": "1",
        "AGENT_RESUME": "true",
    }
    assert claude_code_env("m", "default", "5", False)["AGENT_RESUME"] == "false"


def test_agents_build() -> None:
    from inspect_ai.model import get_model

    assert callable(red_team_claude_code(effort="xhigh", env={}))
    assert callable(red_team_claude_code(effort="default", env={}, small_model="mockllm/model"))
    assert callable(red_team_react(get_model("mockllm/model"), "xhigh"))
    assert callable(red_team_react(get_model("mockllm/model"), "default", compaction=False))


async def _fake_proc(chunks: list[object]) -> AsyncIterator[object]:
    for chunk in chunks:
        yield chunk


async def test_cli_output_is_captured_only_inside_a_capture() -> None:
    chunks = [
        ExecStdout(data='{"type":"system","subtype":"init"}\n{"type":"assi'),
        ExecStderr(data="warning\n"),
        ExecStdout(data='stant"}\n\nnot json\n{"type":"result"}'),
        ExecCompleted(exit_code=0),
    ]
    stdout: list[str] = []
    stderr: list[str] = []
    with capture_cli_output(stdout, stderr):
        module = importlib.import_module("inspect_swe._claude_code.claude_code")
        events = [e async for e in module.claude_code_event_stream(_fake_proc(chunks))]
    assert stdout == [
        '{"type":"system","subtype":"init"}',
        '{"type":"assistant"}',
        "not json",
        '{"type":"result"}',
    ]
    assert stderr == ["warning\n"]
    assert [type(e).__name__ for e in events] == [
        "JsonlEvent",
        "StderrEvent",
        "JsonlEvent",
        "JsonlParseError",
        "JsonlEvent",
        "ExitEvent",
    ]
    # Outside a capture the wrapper passes events through and keeps nothing.
    again = [e async for e in module.claude_code_event_stream(_fake_proc(chunks))]
    assert len(again) == len(events)
    assert len(stdout) == 4
