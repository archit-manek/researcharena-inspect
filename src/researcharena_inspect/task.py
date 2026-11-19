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
  (`pipeline_redteam.sh:169-309`), for `inference_optimization` and the two post-training
  settings the model (and their datasets, and `safety_finetuning`'s judges) in the agent's
  Hugging Face cache
  (`environment.model_cache`), and the `pipeline_pre_agent.sh` hooks, such as the post-training
  `independent` side task's upload capture, in a service beside the agent
  (`environment.services`). `setup=[]` stages nothing.
- `scorer`: `scoring.red_team_scorer`, upstream's Phase 4 (`pipeline_redteam.sh:415-530`).

The agent is Claude Code (as upstream), ReAct, `null` (does nothing, so the staged workspace is
scored as staged) or `replay` (a given workspace, such as a published run's, in place of the
agent's work; one sample per replayed run).
"""

# Adapted from https://github.com/aisa-group/ResearchArena at b922f4b (Copyright (c) 2026 AI Safety and Alignment Group, MIT licence)

import os
from pathlib import PurePosixPath

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec
from inspect_evals_utils.metadata import load_eval_version

from researcharena_inspect.agents.replay import REPLAY_MODES, ReplayMode, replay_source
from researcharena_inspect.agents.solver import AGENTS, red_team_agent
from researcharena_inspect.environment.layout import CONTAINER_ENV
from researcharena_inspect.environment.model_cache import (
    MODEL_CACHE_MODES,
    PROVISIONED_SETTINGS,
    ModelCacheMode,
)
from researcharena_inspect.environment.sandbox import (
    Backend,
    GpuName,
    sandbox_spec,
    with_evaluator,
)
from researcharena_inspect.environment.services import pipeline_hooks
from researcharena_inspect.environment.staging import stage_agent_workspace
from researcharena_inspect.pins import CLAUDE_CODE_SMALL_MODEL
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
from researcharena_inspect.scoring.baseline_process import (
    OPTION_SETTINGS as BASELINE_PROCESS_SETTINGS,
)
from researcharena_inspect.scoring.epochs import scored_epochs_mean

EVAL_VERSION = load_eval_version(__file__)

PROMPT_FAMILY = "claude"
"""Whose prompt every agent gets: Claude Code's, with the suffix naming "Read, Write, Bash"."""

STAGED_FAMILY = "claude"
"""Whose launch helpers (`agent_solve.sh`, ...) staging puts in the agent's home: Claude Code's
for every agent, so ReAct and the no-model agents see the home a Claude Code run sees."""

CPU_IMAGE = (
    "python:3.12-bookworm@sha256:4209fae09a3e20afd863e528bfddc235da5fd5977983e1996cd54cc9a72e6267"
)
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
    """A sandbox given by spec, or by type and config.

    `docker` alone is the CPU-only image, with the evaluator's service as every Docker sandbox
    of the task has (D-093).
    """
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
    return SandboxEnvironmentSpec("docker", with_evaluator(compose))


def dataset_setting(dataset: MemoryDataset) -> str | None:
    """The one setting of the dataset's samples; None for samples of several settings.

    Raises:
        ValueError: samples of several settings, one of which runs a model server (its
            sandbox differs, `environment.sandbox_spec`).
    """
    settings = sorted({str((s.metadata or {}).get("setting")) for s in dataset})
    if len(settings) == 1:
        return settings[0]
    if PROVISIONED_SETTINGS & set(settings):
        raise ValueError(f"run {sorted(PROVISIONED_SETTINGS)} on their own, not with {settings}")
    return None


def check_hook_backend(dataset: MemoryDataset, backend: Backend, sandbox: object) -> None:
    """Refuse Kubernetes for a run that needs the hook service (`environment.services`).

    The service must share the agent container's network namespace; Inspect's Kubernetes
    sandbox gives each service a pod of its own.

    Raises:
        ValueError: a sample sources a `pipeline_pre_agent.sh` hook and `backend` is `k8s`.
    """
    if backend != "k8s" or sandbox is not None:
        return
    needing = sorted(
        str(s.id)
        for s in dataset
        if pipeline_hooks(str((s.metadata or {})["setting"]), str((s.metadata or {})["side_task"]))
    )
    if needing:
        raise ValueError(
            f"{needing} source upstream's pipeline_pre_agent.sh (the upload capture), whose "
            "service must share the agent's network namespace: use backend='docker'"
        )


def check_offline_run(
    agent: str,
    model_cache: str,
    backend: str,
    sandbox: object = None,
    settings: frozenset[str] = frozenset(),
) -> None:
    """Refuse `check_offline` where it would not be a scorer check with every input staged.

    Raises:
        ValueError: a model agent (it needs its provider), a cache that may download, a backend
            other than Docker, a `sandbox` given (it would replace the network-off sandbox), a
            host Hugging Face client that is not offline, or a replayed `safety_finetuning`
            model (its judges load only while the Hub answers, D-254).
    """
    from huggingface_hub import constants as hub_constants

    if agent not in ("null", "replay"):
        raise ValueError("check_offline is for scorer checks: agent must be 'null' or 'replay'")
    if model_cache not in ("seed", "none"):
        raise ValueError("check_offline needs model_cache='seed' (or 'none'): nothing may download")
    if backend != "docker":
        raise ValueError("check_offline needs backend='docker'")
    if sandbox is not None:
        raise ValueError("check_offline builds its own network-off sandbox: do not give `sandbox`")
    if not hub_constants.HF_HUB_OFFLINE:
        raise ValueError(
            "check_offline needs HF_HUB_OFFLINE=1 in the environment before Inspect starts, so "
            "that replay reads only what was staged"
        )
    if agent == "replay" and "safety_finetuning" in settings:
        raise ValueError(
            "check_offline cannot score a safety_finetuning model: the scorer's judges load "
            "only while the Hub answers (D-254); run the check without it"
        )


def check_baseline_process(dataset: MemoryDataset, scorer: object) -> None:
    """Refuse `cuda_baseline_process` where it would not do what it says (D-160).

    Raises:
        ValueError: a sample of a setting without CUDA's in-process baseline pass, or a
            custom scorer (the option belongs to `red_team_scorer`).
    """
    others = sorted(
        {str((s.metadata or {}).get("setting")) for s in dataset} - set(BASELINE_PROCESS_SETTINGS)
    )
    if others:
        raise ValueError(
            f"cuda_baseline_process applies to {list(BASELINE_PROCESS_SETTINGS)} only; "
            f"this task has samples of {others}"
        )
    if scorer is not None:
        raise ValueError("cuda_baseline_process is an option of the default scorer; pass one")


@task
def researcharena(
    setting: str = "cuda_kernel",
    side_tasks: str | list[str] | None = None,
    task_config: str | None = None,
    agent: str = "claude_code",
    redruntime: str = "1",
    effort: str = "auto",
    claude_code_version: str = "latest",
    small_model: str | None = CLAUDE_CODE_SMALL_MODEL,
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
    model_cache: ModelCacheMode = "auto",
    hf_seed: str | None = None,
    fail_on_no_model_output: bool = True,
    keep_snapshot: bool = True,
    snapshot_max_mb: float = 20,
    cuda_baseline_process: bool = False,
    model_store: str = "auto",
    model_store_max_gb: float = 20,
    check_offline: bool = False,
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
        small_model: model for Claude Code's own background calls: Haiku 4.5, as in the
            published runs; `None` sends them to the task's model.
        react_compaction: let the ReAct agent summarise its conversation near the context limit.
        backend: `docker` (a machine with an NVIDIA GPU) or `k8s` (for example Hawk).
        gpu: the GPU model to ask Kubernetes for (`l4` or `h100`; `l4` if omitted); must be
            omitted with Docker.
        sandbox: overrides `backend` and `gpu`: a sandbox type (with `sandbox_config`) or a
            full spec. `docker` without a config is a CPU-only image for tests. A Docker
            sandbox given here needs the evaluator's service (`sandbox.with_evaluator`, D-093)
            for scoring in a container of its own; without it, scoring runs in the agent's
            container, as on Kubernetes.
        sandbox_config: sandbox config file for `sandbox`.
        setup: solver(s) run before the agent; the environment's staging if omitted, `[]` for
            none.
        scorer: the scorer(s); upstream's `evaluate.py` (`red_team_scorer`) if omitted.
        replay: for `agent="replay"`: published run ids (`cuda_kernel/<run>`) or local run
            folders, comma-separated or a list; one sample each, replacing `side_tasks`.
        replay_mode: `fresh` (the default: without the cached results `evaluate.py` and
            `bench.py` would read back) or `as_published`. See `agents.replay`.
        time_limit: the sample's time limit in seconds; Inspect gives scoring half of it, which
            can cut upstream's evaluation attempts short (D-099). None for no limit.
        sandbox_start_timeout: seconds to wait for a Kubernetes sandbox to start (sets
            `INSPECT_HELM_TIMEOUT`).
        model_cache: `inference_optimization`, `utility_finetuning` and `safety_finetuning`:
            how the model (and the post-training settings' datasets, and `safety_finetuning`'s
            two judges) reaches the agent's Hugging Face cache before the agent starts
            (upstream's cluster cache already held them; D-130, D-170, D-210). `auto` copies each
            from `hf_seed` if that holds it, else downloads it in the sandbox; `seed` and
            `download` force one; `none` leaves the cache empty. `safety_finetuning`'s judges
            include two gated repositories, which only a seed filled with a token can hold.
        hf_seed: with Docker, for those settings: absolute path on the Docker host of a
            Hugging Face cache holding the pinned repositories, mounted read-only.
        fail_on_no_model_output: end the sample with an error, unscored, when `claude_code` or
            `react` got no response from any model call (an outage, a wrong model name, an
            exhausted key), so that it is not counted as a score and Inspect's retries can rerun
            it. A refusal is a response. `False` scores the untouched workspace, as upstream
            does (D-108).
        keep_snapshot: keep the agent's final workspace (upstream's `red_team/task_snapshot/`,
            as it is after evaluation) in the log, gzipped in the sample's store, so that
            `agents.record.write_run_dir` can rebuild it for artifact monitors.
        snapshot_max_mb: largest gzipped workspace kept, in MiB. A larger one is not kept; the
            sample's `task_snapshot` metadata then says so, with its size and largest files.
            The default scorer only; ignored when `scorer` is given.
        cuda_baseline_process: `cuda_kernel` only, off by default (D-160, not upstream's
            behaviour). The scorer measures the plain-PyTorch baseline with the workspace's
            own `bench.py` in a process of its own before `evaluate.py`, and leaves it where
            `bench.py` reads a cached baseline, so the agent's kernels are benchmarked in a
            process where no baseline pass has run. A `baseline_tps.txt` the agent left is
            replaced by the measurement; results `evaluate.py` reads back (D-001) are not
            touched. Refused for other settings and with a custom `scorer`.
        model_store: post-training settings (`utility_finetuning`, `safety_finetuning`): where the submitted
            workspace, `final_model/` first, is copied after scoring so that it can be re-scored
            (`agent=replay`) and monitored (D-172). `auto`: a folder beside the log
            (`<log>.submissions/<sample>__e<epoch>__<uuid8>/`); `none`: not kept; or a folder. The log
            keeps the workspace without `final_model/` and the store's manifest. The default
            scorer only.
        model_store_max_gb: cap on one stored run folder, in GiB (`final_model/` goes first).
        check_offline: for scorer checks with no agent only (D-185; off by default, when the
            sandbox's network is upstream's open one). The Docker sandbox has no network, and
            the task refuses to build unless the agent calls no model (`null`, `replay`), the
            cache comes from `hf_seed` or is not filled (`model_cache` `seed` or `none`), no
            `sandbox` override is given, and the host's Hugging Face client is offline (`HF_HUB_OFFLINE=1` in the environment
            before Inspect starts), so that nothing calls the Hub during the run: every input
            must be staged first. Refused for a replayed `safety_finetuning` model: its judges
            load only while the Hub answers (D-254).
    """
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}, got {agent!r}")
    if (agent == "replay") != (replay is not None):
        raise ValueError("`replay` is given with agent='replay', and only then")
    if replay_mode not in REPLAY_MODES:
        raise ValueError(f"replay_mode must be one of {REPLAY_MODES}, got {replay_mode!r}")
    if model_cache not in MODEL_CACHE_MODES:
        raise ValueError(f"model_cache must be one of {MODEL_CACHE_MODES}, got {model_cache!r}")
    if sandbox_start_timeout is not None:
        os.environ["INSPECT_HELM_TIMEOUT"] = str(sandbox_start_timeout)
    dataset = (
        replay_dataset(replay)
        if replay is not None
        else red_team_dataset(setting, side_tasks, redruntime, task_config)
    )
    if cuda_baseline_process:  # D-160
        check_baseline_process(dataset, scorer)
    check_hook_backend(dataset, backend, sandbox)
    if check_offline:  # D-185
        settings = frozenset((s.metadata or {}).get("setting", "") for s in dataset)
        check_offline_run(agent, model_cache, backend, sandbox, settings)
    return Task(
        dataset=dataset,
        setup=(
            setup
            if setup is not None
            else stage_agent_workspace(agent=STAGED_FAMILY, model_cache=model_cache)
        ),
        solver=red_team_agent(
            agent=agent,
            effort=effort,
            claude_code_version=claude_code_version,
            small_model=small_model,
            react_compaction=react_compaction,
            replay_mode=replay_mode,
            fail_on_no_model_output=fail_on_no_model_output,
        ),
        scorer=(
            scorer
            if scorer is not None
            else red_team_scorer(
                keep_snapshot=keep_snapshot,
                snapshot_max_bytes=int(snapshot_max_mb * 1024 * 1024),
                cuda_baseline_process=cuda_baseline_process,
                model_store=model_store,
                model_store_max_bytes=int(model_store_max_gb * 1024**3),
            )
        ),
        sandbox=(
            sandbox_spec(
                backend,
                gpu,
                setting=dataset_setting(dataset),
                hf_seed=hf_seed,
                offline=check_offline,
            )
            if sandbox is None
            else red_team_sandbox(sandbox, sandbox_config)
        ),
        time_limit=time_limit,
        # D-123: with --epochs, upstream's -1 ("no score") is left out of each sample's mean.
        epochs=Epochs(1, scored_epochs_mean()),
        version=EVAL_VERSION,
    )
