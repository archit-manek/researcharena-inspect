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
"""

from pathlib import Path
from typing import Final, Literal

from inspect_ai.util import SandboxEnvironmentSpec

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


def sandbox_spec(backend: Backend = "k8s", gpu: GpuName | None = None) -> SandboxEnvironmentSpec:
    """The Inspect sandbox for the agent.

    Args:
        backend: "k8s" for a Kubernetes cluster such as Hawk, "docker" for one Docker machine.
        gpu: the GPU model to ask Kubernetes for ("l4" if None). With Docker the machine's GPU
            is used, so a model cannot be chosen and must be None.

    Raises:
        ValueError: an unknown backend or GPU, or a GPU model with Docker.
    """
    if backend == "docker":
        if gpu is not None:
            raise ValueError("the docker backend uses the machine's GPU; pass gpu=None")
        return SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
    if backend == "k8s":
        chosen = gpu or "l4"
        if chosen not in K8S_VALUES:
            raise ValueError(f"unknown GPU {chosen!r}; available: {sorted(K8S_VALUES)}")
        return SandboxEnvironmentSpec("k8s", str(K8S_VALUES[chosen]))
    raise ValueError(f"unknown sandbox backend {backend!r}; use 'docker' or 'k8s'")
