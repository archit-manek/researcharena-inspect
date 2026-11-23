"""The sandbox the agent runs in: Docker, with one GPU.

Upstream asks its cluster for one "NVIDIA H100 80GB HBM3", 8 CPUs, 64 GB (`run.sub`) or 256 GB
(`run_redteam.sub`) of memory and 100-200 GB of disk, and runs the agent with `apptainer exec
--nv` (all GPUs of the job, made visible as device 0 by `CUDA_VISIBLE_DEVICES=0`) and no
network limit (ledger D-029). `compose.yaml` keeps the single GPU and the open network: a
single-GPU Docker machine with the NVIDIA Container Toolkit; upstream's 8 CPUs and 64 GB.

For a setting whose servers run vLLM (`inference_optimization`, and the two post-training
settings, whose scorers serve the agent's model with vLLM and whose agents train with PyTorch
data loaders; `model_cache.PROVISIONED_SETTINGS`), the
Docker sandbox is `compose.yaml` with two additions (`vllm_compose`): a larger `/dev/shm` (D-132)
and, if given, a host Hugging Face cache mounted read-only at `model_cache.SEED_MOUNT` (D-130).

Every sandbox this module builds for the red-team task also gets the evaluator service
(`with_evaluator`, D-093): a second container of the same image with the same GPU, CPUs, memory and `/dev/shm`,
which shares two named volumes with the agent's container and nothing else: the Hugging Face
cache at `HF_HOME` (upstream's per-run overlay) and the run folder at `EVAL_DIR`. Upstream's
evaluator runs there after the agent's container has been stopped (`environment.evaluation`).
A custom `sandbox` with no `evaluator` service scores in the agent's container instead.

A setting with a `pipeline_pre_agent.sh` hook (`utility_finetuning`, `safety_finetuning`) also
gets the hook service (`with_hook_service`, `environment.services`): a second container of the
same image that joins the agent container's network namespace and nothing else.
"""

from pathlib import Path
from typing import Final

import yaml
from inspect_ai.util import (
    ComposeConfig,
    ComposeService,
    ComposeVolumeMount,
    SandboxEnvironmentSpec,
)

from researcharena_inspect.environment.layout import EVAL_DIR, EVALUATOR_SERVICE, HF_HOME
from researcharena_inspect.environment.model_cache import PROVISIONED_SETTINGS, SEED_MOUNT
from researcharena_inspect.environment.services import SERVICE_NAME, setting_has_hooks

_HERE = Path(__file__).parent
COMPOSE_FILE: Final = _HERE / "compose.yaml"

VLLM_SHM_SIZE: Final = "16gb"
"""`/dev/shm` for a sandbox that runs vLLM (D-132). Docker's default is 64 MB; vLLM's Docker
guide asks for `--ipc=host` or a larger `--shm-size`. Upstream's Apptainer containers have no
Docker default; the size they had is not known."""


def _load_compose() -> ComposeConfig:
    return ComposeConfig.model_validate(yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8")))


def vllm_compose(hf_seed: str | None = None) -> ComposeConfig:
    """`compose.yaml` with a larger `/dev/shm` and, if given, the host cache mounted read-only.

    Args:
        hf_seed: absolute path on the Docker host of a Hugging Face cache (`HF_HOME` layout,
            `<path>/hub/models--...`, or hub layout) that holds the pinned model; staging copies
            the model from it (`model_cache`).
    """
    config = _load_compose()
    service = config.services["default"]
    service.shm_size = VLLM_SHM_SIZE  # D-132
    if hf_seed is not None:  # D-130: the host's cache, copied at staging
        if not Path(hf_seed).is_absolute():
            raise ValueError(
                f"hf_seed must be an absolute path on the Docker host, got {hf_seed!r}"
            )
        service.volumes = [*(service.volumes or []), f"{hf_seed}:{SEED_MOUNT}:ro"]
    return config


SHARED_VOLUMES: Final = {"hf_cache": HF_HOME, "eval_dir": EVAL_DIR}
"""Named volume -> mount point, in the agent's and the evaluator's containers: what upstream's
two containers share (`pipeline_redteam.sh:373`, `:486-487`). Compose removes them with the
sample (`down --volumes`), as upstream's overlay and run folder are per run."""


def with_evaluator(config: ComposeConfig) -> ComposeConfig:
    """`config` plus the evaluator's service, sharing `SHARED_VOLUMES` with the default one.

    The service is the default one's image, platform, GPU request, CPUs, memory, `/dev/shm` and
    network, idle until scoring; it has none of the default one's other mounts (the host seed of
    D-130 is for staging only). The default service needs `init: true`: scoring stops it with
    SIGTERM to its init (`evaluation.stop_agent_container`).
    """
    default = config.services["default"]
    # D-250: the run folder is mounted in the agent's container too (empty and root's until
    # scoring), so that the workspace reaches the evaluator's without a copy through the host.
    mounts: list[str | ComposeVolumeMount] = [f"{n}:{p}" for n, p in SHARED_VOLUMES.items()]
    default.volumes = [*(default.volumes or []), *mounts]
    config.services[EVALUATOR_SERVICE] = ComposeService(
        image=default.image,
        platform=default.platform,
        command=["tail", "-f", "/dev/null"],
        init=True,
        cpus=default.cpus,
        mem_limit=default.mem_limit,
        shm_size=default.shm_size,
        deploy=default.deploy,
        network_mode=default.network_mode,
        volumes=list(mounts),
    )
    config.volumes = {**(config.volumes or {}), **{name: {} for name in SHARED_VOLUMES}}
    return config


def with_hook_service(config: ComposeConfig) -> ComposeConfig:
    """`config` plus the hook service: the default service's image, sharing its network.

    `network_mode: service:default` puts the service in the agent container's network namespace
    (its loopback, as Apptainer's agent shares the cluster host's), with its own processes and
    files, which the agent cannot see. It asks for no GPU and idles until staging starts the
    hooks in it.
    """
    default = config.services["default"]
    config.services[SERVICE_NAME] = ComposeService(
        image=default.image,
        platform=default.platform,
        command=["tail", "-f", "/dev/null"],
        init=True,
        network_mode="service:default",
    )
    return config


def without_network(config: ComposeConfig) -> ComposeConfig:
    """`config` with the agent's and the evaluator's services on no network, D-185.

    For a scorer check with no agent (`check_offline`): the hook service, if there is one,
    shares the agent's namespace and has none either. The setting's default stays upstream's
    open network (D-029).
    """
    for name in ("default", EVALUATOR_SERVICE):
        if name in config.services:
            config.services[name].network_mode = "none"
            config.services[name].networks = None
    return config


def sandbox_spec(
    *,
    setting: str | None = None,
    hf_seed: str | None = None,
    offline: bool = False,
    evaluator: bool = True,
) -> SandboxEnvironmentSpec:
    """The Inspect sandbox for the agent: Docker on the local machine's GPU.

    Args:
        setting: the setting the sandbox is for; a setting whose servers run vLLM gets
            `vllm_compose`, plus the hook service if the setting has `pipeline_pre_agent.sh`
            hooks; every other one gets `compose.yaml`.
        hf_seed: a host Hugging Face cache to mount read-only (see `vllm_compose`).
        offline: for scorer checks with no agent: the sandbox has no network
            (`without_network`, D-185). Off by default.
        evaluator: add the evaluator's service (`with_evaluator`, D-093), where the red-team
            scorer runs upstream's evaluator. The diagnostic tasks, which run `evaluate.py` on
            a staged workspace with no agent, keep one container (False).
    """
    hooks = setting is not None and setting_has_hooks(setting)
    if setting in PROVISIONED_SETTINGS or hf_seed is not None:
        config = vllm_compose(hf_seed)
    elif hooks or offline or evaluator:
        config = _load_compose()
    else:
        return SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
    if evaluator:
        config = with_evaluator(config)  # D-093
    if hooks:
        config = with_hook_service(config)
    return SandboxEnvironmentSpec("docker", without_network(config) if offline else config)
