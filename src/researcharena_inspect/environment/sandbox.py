"""The Docker sandbox the agent runs in: `compose.yaml` and what each setting adds to it.

Upstream asks its cluster for one H100, 8 CPUs and 64 GB (`run.sub`) or 256 GB
(`run_redteam.sub`), and runs the agent with `apptainer exec --nv` and an open network (D-029).
`compose.yaml` keeps one GPU, 8 CPUs, 64 GB and the open network (D-261). `PROVISIONED_SETTINGS`,
whose servers or scorers run vLLM and whose agents may train with PyTorch data loaders, get
`vllm_compose` (D-130, D-132). A custom `sandbox` with no `evaluator` service scores in the
agent's container instead (D-093).
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
        hf_seed: absolute path on the Docker host of a Hugging Face cache (`HF_HOME` or hub
            layout) holding the pinned models; staging copies from it (D-130).
    """
    config = _load_compose()
    service = config.services["default"]
    service.shm_size = VLLM_SHM_SIZE  # D-132
    if hf_seed is not None:  # D-130
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
    """`config` plus the evaluator's service (D-093), sharing `SHARED_VOLUMES` with the default.

    It copies the default service's resources and network but none of its other mounts (the
    host seed is for staging only). The default service needs `init: true`: scoring stops it
    with SIGTERM to its init (`evaluation.stop_agent_container`).
    """
    default = config.services["default"]
    # D-250: the run folder is in the agent's container too, empty and root's until scoring.
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
    """`config` plus the hook service: the default service's image, in its network namespace.

    The agent shares its loopback, as Apptainer's agent shares the host's, but cannot see its
    processes or files (`environment.services`, D-174).
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
    """`config` with no network for the agent's and evaluator's services (D-185).

    The hook service, if any, shares the agent's namespace and so has none either.
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
        setting: `PROVISIONED_SETTINGS` get `vllm_compose`; a setting with
            `pipeline_pre_agent.sh` hooks also gets the hook service.
        hf_seed: a host Hugging Face cache to mount read-only (see `vllm_compose`).
        offline: no network, for scorer checks with no agent (D-185).
        evaluator: add the evaluator's service (D-093). False for the diagnostic tasks, which
            run `evaluate.py` with no agent in one container.
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
