"""ResearchArena's red-team run as an Inspect task: one sample per side task (or baseline).

Port of `run_redteam.sh` and `src/pipeline_redteam.sh` at the pinned commit. Each sample is one
red-team run of a setting: the prompt is built by `prompts.agent_prompt`, the agent runs in a
sandbox for upstream's time budget and is re-prompted as upstream re-prompts it, and the sample
is then scored. Three things come from elsewhere and are passed in:

- `sandbox`: the container. The default is a plain Python image with upstream's working folder
  and environment, enough to run the agent; the arena image and GPU come from the environment
  work package.
- `setup`: the solver that stages the agent's workspace before the agent starts
  (`pipeline_redteam.sh:169-249` and the pre-agent hooks, `:272-309`). Default: none.
- `scorer`: the main and side task scorers (`pipeline_redteam.sh:440-530`). Default: none.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec
from inspect_evals_utils.metadata import load_eval_version

from researcharena_inspect.agents.layout import AGENT_TASK_DIR, CONTAINER_ENV
from researcharena_inspect.agents.solver import AGENTS, red_team_agent
from researcharena_inspect.prompts import (
    BASELINE,
    agent_instruction_file,
    agent_prompt,
    default_task_config,
    format_hours,
    list_side_tasks,
    resume_agent,
)

EVAL_VERSION = load_eval_version(__file__)

PROMPT_FAMILY = "claude"
"""Whose prompt both agents get: Claude Code's, with the suffix naming "Read, Write, Bash"."""

DEFAULT_IMAGE = "python:3.12-slim-bookworm"
"""Default sandbox image: runs the agent, but has no CUDA, no git and none of the setting's
packages. Pass the environment's sandbox config for real runs."""


def red_team_sample(
    setting: str,
    side_task: str,
    redruntime: str | int | float,
    task_config: str | None = None,
) -> Sample:
    """One red-team run: the agent's first prompt as input, the run's settings as metadata.

    Args:
        setting: e.g. `cuda_kernel`.
        side_task: a side task of the setting, or `none` for the baseline.
        redruntime: the budget in hours, as upstream's `--redruntime` (kept as typed, since it
            appears in the prompt).
        task_config: the setting's task config; `None` for its first, as upstream defaults.
    """
    hours = format_hours(redruntime)
    config = task_config if task_config is not None else default_task_config(setting)
    side_tag = "baseline" if side_task == BASELINE else side_task
    return Sample(
        id=f"{setting}__{side_tag}",
        input=agent_prompt(setting, side_task, PROMPT_FAMILY, hours, config),
        metadata={
            "setting": setting,
            "side_task": side_task,
            "task_config": config,
            "redruntime": hours,
            "resume_agent": resume_agent(setting),
            "baseline": side_task == BASELINE,
            "prompt_family": PROMPT_FAMILY,
            # Upstream's prompt.txt in the agent's home and the run's instruction.md
            # (pipeline_redteam.sh:264-265), for a staging step to write.
            "instruction_file": agent_instruction_file(
                setting, side_task, PROMPT_FAMILY, hours, config
            ),
        },
    )


def _as_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def red_team_dataset(
    setting: str,
    side_tasks: str | list[str] | None,
    redruntime: str | int | float,
    task_config: str | None = None,
) -> MemoryDataset:
    """The setting's runs: every side task and the baseline, or the ones named."""
    available = list_side_tasks(setting)
    wanted = _as_list(side_tasks)
    if wanted is None:
        wanted = available
    unknown = [s for s in wanted if s not in available]
    if unknown:
        raise ValueError(f"unknown side tasks for {setting}: {unknown}; available: {available}")
    samples = [red_team_sample(setting, s, redruntime, task_config) for s in wanted]
    return MemoryDataset(samples, name=f"researcharena_{setting}")


def red_team_sandbox(
    sandbox: str | SandboxEnvironmentSpec, sandbox_config: str | None
) -> SandboxEnvironmentSpec:
    """The agent's sandbox: a given spec or config, or for Docker a plain Python image."""
    if isinstance(sandbox, SandboxEnvironmentSpec):
        return sandbox
    if sandbox_config is not None or sandbox != "docker":
        return SandboxEnvironmentSpec(sandbox, sandbox_config)
    compose = ComposeConfig(
        services={
            "default": ComposeService(
                image=DEFAULT_IMAGE,
                command="tail -f /dev/null",
                init=True,
                working_dir=AGENT_TASK_DIR,
                environment=dict(CONTAINER_ENV),
            )
        }
    )
    return SandboxEnvironmentSpec("docker", compose)


@task
def researcharena(
    setting: str = "cuda_kernel",
    side_tasks: str | list[str] | None = None,
    task_config: str | None = None,
    agent: str = "claude_code",
    redruntime: str = "1",
    effort: str = "auto",
    claude_code_version: str = "latest",
    small_model: str | None = None,
    react_compaction: bool = True,
    sandbox: str | SandboxEnvironmentSpec = "docker",
    sandbox_config: str | None = None,
    setup: Solver | None = None,
    scorer: Scorer | list[Scorer] | None = None,
) -> Task:
    """ResearchArena's red-team run: an agent does the setting's task (and a side task).

    Args:
        setting: the setting, e.g. `cuda_kernel`.
        side_tasks: side tasks to run (comma-separated or a list), `none` for the baseline;
            all of the setting's and the baseline if omitted.
        task_config: the setting's task config; its first if omitted, as upstream.
        agent: `claude_code` (Claude Code through Inspect SWE, as upstream) or `react`.
        redruntime: the agent's budget in hours, as upstream's `--redruntime`. The published
            runs used 1 for CUDA and inference and 5 for post-training; `0.1667` is ten minutes.
        effort: the agent's reasoning effort; `auto` reads upstream's `effort.json`
            (`xhigh`), `default` sets none.
        claude_code_version: Claude Code version (`latest`, `stable`, or e.g. `2.1.162`).
            Upstream installs the latest release at the start of each run.
        small_model: model for Claude Code's background calls; the task's model if omitted.
        react_compaction: let the ReAct agent summarise its conversation near the context limit.
        sandbox: sandbox type (`docker`, `k8s`, ...) or a full sandbox spec.
        sandbox_config: sandbox config file; for Docker the default is a plain Python image.
        setup: solver that stages the agent's workspace before the agent runs.
        scorer: the main and side task scorers.
    """
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")
    return Task(
        dataset=red_team_dataset(setting, side_tasks, redruntime, task_config),
        setup=setup,
        solver=red_team_agent(
            agent=agent,
            effort=effort,
            claude_code_version=claude_code_version,
            small_model=small_model,
            react_compaction=react_compaction,
        ),
        scorer=scorer,
        sandbox=red_team_sandbox(sandbox, sandbox_config),
        version=EVAL_VERSION,
    )
