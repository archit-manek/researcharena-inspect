"""Environment check: report the GPU a sandbox can see. It makes no model calls."""

import tempfile
from pathlib import Path
from typing import Literal

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentType, sandbox

EVAL_VERSION = "1-A"  # literal on this side branch; main reads it from eval.yaml

# Requests one GPU from the cluster; see the file for why it sets no runtime class.
K8S_VALUES = Path(__file__).parent / "gpu_probe_values.yaml"

# name, memory, driver and CUDA compute capability decide whether the eval's image can run:
# upstream builds its kernels for compute capability 8.0, 8.9 and 9.0.
NVIDIA_SMI = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version,compute_cap",
    "--format=csv,noheader",
]
# Node size as the sandbox sees it: CPU count and total memory.
NODE_SIZE = ["sh", "-c", "nproc; grep MemTotal /proc/meminfo"]
PROBE_TIMEOUT_SECONDS = 120
PROBE_METADATA_KEY = "gpu_probe"
NODE_SIZE_METADATA_KEY = "node_size"

# Karpenter labels every AWS GPU node with the GPU's short name, so this selector works
# whatever labels the operator gave the node pool.
GPU_NAME_LABEL = "karpenter.k8s.aws/instance-gpu-name"
GpuName = Literal["t4", "l4", "h100"]


def k8s_values(gpu_name: GpuName | None) -> Path:
    """Return the sandbox values file, with a node selector when a GPU model is asked for."""
    if gpu_name is None:
        return K8S_VALUES
    values = yaml.safe_load(K8S_VALUES.read_text())
    values["services"]["default"]["nodeSelector"] = {GPU_NAME_LABEL: gpu_name}
    path = Path(tempfile.mkdtemp(prefix="gpu_probe_")) / f"values_{gpu_name}.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


@solver
def run_nvidia_smi() -> Solver:
    """Run `nvidia-smi` in the sandbox and record what it prints."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            result = await sandbox().exec(NVIDIA_SMI, timeout=PROBE_TIMEOUT_SECONDS)
            success, report = result.success, (result.stdout if result.success else result.stderr)
        except (FileNotFoundError, PermissionError, TimeoutError) as error:
            # The local sandbox raises when the binary is missing; a container returns non-zero.
            success, report = False, f"{type(error).__name__}: {error}"
        state.metadata[PROBE_METADATA_KEY] = {"success": success, "report": report.strip()}
        try:
            size = await sandbox().exec(NODE_SIZE, timeout=PROBE_TIMEOUT_SECONDS)
            state.metadata[NODE_SIZE_METADATA_KEY] = (size.stdout or size.stderr).strip()
        except (FileNotFoundError, PermissionError, TimeoutError) as error:
            state.metadata[NODE_SIZE_METADATA_KEY] = f"{type(error).__name__}: {error}"
        state.output = ModelOutput.from_content(model="none", content=report.strip())
        return state

    return solve


@scorer(metrics=[accuracy()])
def gpu_visible() -> Scorer:
    """Score CORRECT when `nvidia-smi` ran and listed at least one GPU."""

    async def score(state: TaskState, target: Target) -> Score:
        probe = state.metadata.get(
            PROBE_METADATA_KEY, {"success": False, "report": "probe did not run"}
        )
        visible = bool(probe["success"]) and bool(probe["report"])
        return Score(
            value=CORRECT if visible else INCORRECT,
            answer=probe["report"],
            explanation="name, memory.total, driver_version, compute_cap"
            if visible
            else "no GPU visible",
        )

    return score


@task
def gpu_probe(
    sandbox_type: Literal["k8s", "local"] = "k8s", gpu_name: GpuName | None = None
) -> Task:
    """Report the GPU visible to a sandbox, without calling a model.

    Args:
        sandbox_type: "k8s" requests one GPU from a Kubernetes cluster (for example Hawk);
            "local" runs on the machine Inspect runs on.
        gpu_name: ask the cluster for this GPU model ("t4", "l4" or "h100"); None takes
            whatever GPU node the cluster picks.

    Returns:
        A one-sample task whose answer is the `nvidia-smi` line for each visible GPU.
    """
    sandbox_spec: SandboxEnvironmentType = (
        ("k8s", str(k8s_values(gpu_name))) if sandbox_type == "k8s" else "local"
    )
    return Task(
        dataset=MemoryDataset(
            [Sample(id="gpu-probe", input="Report the GPU visible to the sandbox.")]
        ),
        solver=run_nvidia_smi(),
        scorer=gpu_visible(),
        sandbox=sandbox_spec,
        time_limit=600,
        version=EVAL_VERSION,
    )
