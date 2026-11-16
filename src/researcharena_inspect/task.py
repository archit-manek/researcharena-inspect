"""ResearchArena's red-team run as an Inspect task: one sample per side task (or baseline).

Port of `run_redteam.sh` and `src/pipeline_redteam.sh` at the pinned commit. Each sample is one
red-team run of a setting: the prompt is built by `prompts.agent_prompt`, the workspace is staged
as upstream stages it, the agent runs in the sandbox for upstream's time budget and is
re-prompted as upstream re-prompts it, and the workspace is then scored by upstream's
`evaluate.py`. By default each part comes from the environment and scoring packages:

- `sandbox`: `environment.sandbox_spec(backend, gpu)`, upstream's image with one GPU (Docker on
  a GPU machine, or Kubernetes such as Hawk with `backend="k8s"`). `sandbox="docker"` without a
  config gives a CPU-only Debian image for tests.
- `setup`: `environment.stage_agent_workspace`, upstream's staging and pre-agent hooks
  (`pipeline_redteam.sh:169-309`). `setup=[]` stages nothing.
- `scorer`: `scoring.red_team_scorer`, upstream's Phase 4 (`pipeline_redteam.sh:415-530`).

The agent is Claude Code (as upstream), ReAct, `null` (does nothing, so the staged workspace is
scored as staged) or `replay` (a given workspace, such as a published run's, in place of the
agent's work; one sample per replayed run).
"""

import os
from pathlib import PurePosixPath

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec

from researcharena_inspect.agents.replay import REPLAY_MODES, ReplayMode, replay_source
from researcharena_inspect.agents.solver import AGENTS, red_team_agent
from researcharena_inspect.environment.layout import CONTAINER_ENV
from researcharena_inspect.environment.sandbox import Backend, GpuName, sandbox_spec
from researcharena_inspect.environment.staging import stage_agent_workspace
from researcharena_inspect.prompts import (
    BASELINE,
    agent_instruction_file,
    agent_prompt,
    default_task_config,
    format_hours,
    list_side_tasks,
    resume_agent,
)
from researcharena_inspect.scoring import red_team_scorer

EVAL_VERSION = "1-A"  # side branch: inspect-evals-utils 0.2.0 has no metadata

PROMPT_FAMILY = "claude"
"""Whose prompt every agent gets: Claude Code's, with the suffix naming "Read, Write, Bash"."""

STAGED_FAMILY = "claude"
"""Whose launch helpers (`agent_solve.sh`, ...) staging puts in the agent's home: Claude Code's
for every agent, so ReAct and the no-model agents see the home a Claude Code run sees."""

CPU_IMAGE = "python:3.12-bookworm"
"""The CPU-only sandbox's image (`sandbox="docker"` without a config): no CUDA and none of the
setting's packages, but git, which staging's hooks need. For tests and plumbing only."""


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


def replay_dataset(sources: str | list[str]) -> MemoryDataset:
    """One sample per replayed run: published run ids or local run folders.

    Each sample is the run's own setting, side task, config and budget, with the source in
    `metadata["replay"]`; its id is the run's folder name.
    """
    samples = []
    for name in _as_list(sources) or []:
        source = replay_source(name)
        sample = red_team_sample(
            source.setting, source.side_task, source.redruntime, source.task_config
        )
        sample.id = PurePosixPath(name.rstrip("/")).name
        assert sample.metadata is not None
        sample.metadata["replay"] = source.source
        samples.append(sample)
    if not samples:
        raise ValueError("agent='replay' needs `replay`: published run ids or run folders")
    return MemoryDataset(samples, name="researcharena_replay")


def red_team_sandbox(
    sandbox: str | SandboxEnvironmentSpec, sandbox_config: str | None
) -> SandboxEnvironmentSpec:
    """A sandbox given by spec, or by type and config; `docker` alone is the CPU-only image."""
    if isinstance(sandbox, SandboxEnvironmentSpec):
        return sandbox
    if sandbox_config is not None or sandbox != "docker":
        return SandboxEnvironmentSpec(sandbox, sandbox_config)
    compose = ComposeConfig(
        services={
            "default": ComposeService(
                image=CPU_IMAGE,
                command="tail -f /dev/null",
                init=True,
                # No `working_dir`: Docker would create /home/user/task as root before staging,
                # which needs an empty home. The agent's commands set their own directory.
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
    backend: Backend = "docker",
    gpu: GpuName | None = None,
    sandbox: str | SandboxEnvironmentSpec | None = None,
    sandbox_config: str | None = None,
    setup: Solver | list[Solver] | None = None,
    scorer: Scorer | list[Scorer] | None = None,
    replay: str | list[str] | None = None,
    replay_mode: ReplayMode = "fresh",
    time_limit: int | None = None,
    sandbox_start_timeout: int | None = None,
) -> Task:
    """ResearchArena's red-team run: an agent does the setting's task (and a side task).

    Args:
        setting: the setting, e.g. `cuda_kernel`.
        side_tasks: side tasks to run (comma-separated or a list), `none` for the baseline;
            all of the setting's and the baseline if omitted.
        task_config: the setting's task config; its first if omitted, as upstream.
        agent: `claude_code` (Claude Code through Inspect SWE, as upstream), `react`, `null`
            (does nothing; the staged workspace is scored) or `replay` (see `replay`). `null`
            and `replay` call no model.
        redruntime: the agent's budget in hours, as upstream's `--redruntime`. The published
            runs used 1 for CUDA and inference and 5 for post-training; `0.1667` is ten minutes.
        effort: the agent's reasoning effort; `auto` reads upstream's `effort.json`
            (`xhigh`), `default` sets none.
        claude_code_version: Claude Code version (`latest`, `stable`, or e.g. `2.1.162`).
            Upstream installs the latest release at the start of each run.
        small_model: model for Claude Code's background calls; the task's model if omitted.
        react_compaction: let the ReAct agent summarise its conversation near the context limit.
        backend: `docker` (a machine with an NVIDIA GPU) or `k8s` (for example Hawk).
        gpu: the GPU model to ask Kubernetes for (`l4` or `h100`; `l4` if omitted); must be
            omitted with Docker.
        sandbox: overrides `backend` and `gpu`: a sandbox type (with `sandbox_config`) or a
            full spec. `docker` without a config is a CPU-only image for tests.
        sandbox_config: sandbox config file for `sandbox`.
        setup: solver(s) run before the agent; the environment's staging if omitted, `[]` for
            none.
        scorer: the scorer(s); upstream's `evaluate.py` (`red_team_scorer`) if omitted.
        replay: for `agent="replay"`: published run ids (`cuda_kernel/<run>`) or local run
            folders, comma-separated or a list; one sample each, replacing `side_tasks`.
        replay_mode: `fresh` (the default: without the cached results `evaluate.py` and
            `bench.py` would read back) or `as_published`. See `agents.replay`.
        time_limit: the sample's time limit in seconds (Inspect gives scoring half of it).
        sandbox_start_timeout: seconds to wait for a Kubernetes sandbox to start (sets
            `INSPECT_HELM_TIMEOUT`).
    """
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")
    if (agent == "replay") != (replay is not None):
        raise ValueError("`replay` is given with agent='replay', and only then")
    if replay_mode not in REPLAY_MODES:
        raise ValueError(f"replay_mode must be one of {REPLAY_MODES}, got {replay_mode!r}")
    if sandbox_start_timeout is not None:
        os.environ["INSPECT_HELM_TIMEOUT"] = str(sandbox_start_timeout)
    dataset = (
        replay_dataset(replay)
        if replay is not None
        else red_team_dataset(setting, side_tasks, redruntime, task_config)
    )
    return Task(
        dataset=dataset,
        setup=setup if setup is not None else stage_agent_workspace(agent=STAGED_FAMILY),
        solver=red_team_agent(
            agent=agent,
            effort=effort,
            claude_code_version=claude_code_version,
            small_model=small_model,
            react_compaction=react_compaction,
            replay_mode=replay_mode,
        ),
        scorer=scorer if scorer is not None else red_team_scorer(),
        sandbox=(
            sandbox_spec(backend, gpu)
            if sandbox is None
            else red_team_sandbox(sandbox, sandbox_config)
        ),
        time_limit=time_limit,
        version=EVAL_VERSION,
    )
