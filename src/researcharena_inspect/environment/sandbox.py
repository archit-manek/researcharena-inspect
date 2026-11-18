"""The sandbox the agent runs in, for Docker or Kubernetes, with one GPU.

Upstream asks its cluster for one "NVIDIA H100 80GB HBM3", 8 CPUs, 64 GB (`run.sub`) or 256 GB
(`run_redteam.sub`) of memory and 100-200 GB of disk, and runs the agent with `apptainer exec
--nv` (all GPUs of the job, made visible as device 0 by `CUDA_VISIBLE_DEVICES=0`) and no
network limit (ledger D-029). The configs here keep the single GPU and the open network:

- `k8s_values_l4.yaml`: Hawk's L4 pool (AWS g6). The node we got has 8 CPUs and about 32 GB, so
  the requests are 6 CPUs and 24 GiB (D-094); not the paper's GPU.
- `k8s_values_h100.yaml`: Hawk's H100 pool (p5.4xlarge, 16 vCPUs); upstream's 8 CPUs and 64 GiB.
- `compose.yaml`: a single-GPU Docker machine with the NVIDIA Container Toolkit; upstream's
  8 CPUs and 64 GB.

Hawk accepts only the values its validator has reviewed (`hawk/runner/sandbox_values.py` in
METR/hawk v4.1.0): these use `services.default.{image, nodeSelector, resources}` and
`allowEntities`, all on its allowlist. Hawk adds the GPU pool's toleration itself and runs a
GPU service on the node's default runtime (gVisor cannot pass a GPU through).

For a setting whose servers run vLLM (`inference_optimization`, and `utility_finetuning`, whose
scorer serves the agent's model with vLLM and whose agent trains with PyTorch data loaders), the
Docker sandbox is `compose.yaml` with two additions (`vllm_compose`): a larger `/dev/shm` (D-132)
and, if given, a host Hugging Face cache mounted read-only at `model_cache.SEED_MOUNT` (D-130).
Kubernetes gets the same values as every other setting; the model is then downloaded at staging.

A setting with a `pipeline_pre_agent.sh` hook (`utility_finetuning`, `safety_finetuning`) also
gets the hook service (`with_hook_service`, `environment.services`): a second container of the
same image that joins the agent container's network namespace and nothing else. Kubernetes
cannot do that, and a run that needs the service is refused there (`task.py`).
"""

from pathlib import Path
from typing import Final, Literal

import yaml
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec

from researcharena_inspect.environment.model_cache import PROVISIONED_SETTINGS, SEED_MOUNT
from researcharena_inspect.environment.services import SERVICE_NAME, setting_has_hooks

Backend = Literal["docker", "k8s"]
GpuName = Literal["l4", "h100"]

_HERE = Path(__file__).parent
COMPOSE_FILE: Final = _HERE / "compose.yaml"
K8S_VALUES: Final[dict[str, Path]] = {
    "l4": _HERE / "k8s_values_l4.yaml",
    "h100": _HERE / "k8s_values_h100.yaml",
}

# Karpenter's label for the GPU model of an AWS node; Hawk's validator allows `nodeSelector`.
GPU_NAME_LABEL: Final = "karpenter.k8s.aws/instance-gpu-name"

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


def sandbox_spec(
    backend: Backend = "k8s",
    gpu: GpuName | None = None,
    setting: str | None = None,
    hf_seed: str | None = None,
) -> SandboxEnvironmentSpec:
    """The Inspect sandbox for the agent.

    Args:
        backend: "k8s" for a Kubernetes cluster such as Hawk, "docker" for one Docker machine.
        gpu: the GPU model to ask Kubernetes for ("l4" if None). With Docker the machine's GPU
            is used, so a model cannot be chosen and must be None.
        setting: the setting the sandbox is for; with Docker, a setting whose servers run vLLM
            gets `vllm_compose`, plus the hook service if the setting has
            `pipeline_pre_agent.sh` hooks; every other one gets `compose.yaml` unchanged.
        hf_seed: Docker only: a host Hugging Face cache to mount read-only (see `vllm_compose`).

    Raises:
        ValueError: an unknown backend or GPU, a GPU model with Docker, or `hf_seed` with
            Kubernetes.
    """
    if backend == "docker":
        if gpu is not None:
            raise ValueError("the docker backend uses the machine's GPU; pass gpu=None")
        hooks = setting is not None and setting_has_hooks(setting)
        if setting in PROVISIONED_SETTINGS or hf_seed is not None:
            config = vllm_compose(hf_seed)
        elif hooks:
            config = _load_compose()
        else:
            return SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
        return SandboxEnvironmentSpec("docker", with_hook_service(config) if hooks else config)
    if hf_seed is not None:
        raise ValueError("hf_seed mounts a host folder and needs backend='docker'")
    if backend == "k8s":
        chosen = gpu or "l4"
        if chosen not in K8S_VALUES:
            raise ValueError(f"unknown GPU {chosen!r}; available: {sorted(K8S_VALUES)}")
        return SandboxEnvironmentSpec("k8s", str(K8S_VALUES[chosen]))
    raise ValueError(f"unknown sandbox backend {backend!r}; use 'docker' or 'k8s'")
