"""The model a setting's servers load, put into the agent's Hugging Face cache before the agent starts.

Upstream (`src/pipeline_redteam.sh` at the pinned commit) mounts a per-run fuse-overlayfs over the
cluster's own `$HF_HOME` at `/home/user/hf_cache` in the agent's container and again in the
evaluator's (`:337-341`, `:373`, `:486`; `src/utils/hf_overlay.sh`): both see whatever the
cluster's shared cache holds, and their writes go to a layer that is thrown away at the end of
the run. The pipeline never downloads a model; it must already be in the cluster's cache, at
whatever revision was fetched there.

For `inference_optimization` that matters: the starter `launch.sh`, `server.py` and upstream's
`baseline_server.py` all set `HF_HUB_OFFLINE=1`, so the model is read from that cache or not at
all. The port (D-130) puts the setting's model at a pinned revision into `/home/user/hf_cache`
at staging, as the agent's user, before the agent's clock starts, from one of:

- a read-only host cache mounted into the Docker sandbox at `SEED_MOUNT` (the task's `hf_seed`),
  copied, so that writes stay within the sample as with upstream's overlay;
- a download inside the sandbox (its network is open, as upstream's is: D-029).

The cache then holds that one model, where upstream's held the cluster's whole cache (D-131).
`cuda_kernel` is not in `PROVISIONED_SETTINGS`: its staging is unchanged and its benchmark
downloads TinyLlama at run time, as before.
"""

import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.layout import AGENT_HOME, AGENT_USER, HF_HOME
from researcharena_inspect.prompts.settings import load_config

ModelCacheMode = Literal["auto", "seed", "download", "none"]
MODEL_CACHE_MODES: tuple[ModelCacheMode, ...] = ("auto", "seed", "download", "none")

SEED_MOUNT: Final = "/opt/hf_seed"
"""Where the sandbox sees the host's read-only cache (the task's `hf_seed`), if there is one."""

SCRIPT_SOURCE: Final = Path(__file__).with_name("model_cache_script.py")
SCRIPT_PATH: Final = "/var/tmp/researcharena-model-cache.py"
RESULT_PREFIX: Final = "MODEL_CACHE_RESULT="

# D-130, D-131: only these settings get a model, and only the task config's one.
PROVISIONED_SETTINGS: Final = frozenset({"inference_optimization"})
"""Settings whose servers load the task config's `model` from the cache with the network off."""


@dataclass(frozen=True)
class PinnedFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedModel:
    """A Hugging Face model repository at one commit, with every file's size and SHA-256."""

    repo_id: str
    revision: str
    files: tuple[PinnedFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# Read from the Hub API on 18 Nov 2026 (`/api/models/Qwen/Qwen2.5-3B-Instruct?blobs=true`): the
# repository's `main` has been this commit since 25 Sep 2024 ("Update metadata: Add
# library_name to Transformers (#2)"), so every 2026 run that downloaded `main` got these files.
# The two weight files' SHA-256 are their LFS object ids; the others were downloaded at the
# commit and hashed. Upstream pins no revision (config.json names the repository only).
QWEN2_5_3B_INSTRUCT: Final = PinnedModel(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    files=(
        PinnedFile(
            ".gitattributes",
            1519,
            "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        ),
        PinnedFile(
            "LICENSE", 7388, "ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9"
        ),
        PinnedFile(
            "README.md", 4930, "2c74693e6ae36ba878124d465195b4dad0f71035b54ece1c9fd1553cc497ef8c"
        ),
        PinnedFile(
            "config.json", 661, "eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b"
        ),
        PinnedFile(
            "generation_config.json",
            242,
            "ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9",
        ),
        PinnedFile(
            "merges.txt",
            1671839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        PinnedFile(
            "model-00001-of-00002.safetensors",
            3968658944,
            "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
        ),
        PinnedFile(
            "model-00002-of-00002.safetensors",
            2203268048,
            "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
        ),
        PinnedFile(
            "model.safetensors.index.json",
            35581,
            "bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56",
        ),
        PinnedFile(
            "tokenizer.json",
            7031645,
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        ),
        PinnedFile(
            "tokenizer_config.json",
            7305,
            "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        ),
        PinnedFile(
            "vocab.json",
            2776833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ),
)

PINNED_MODELS: Final[dict[str, PinnedModel]] = {QWEN2_5_3B_INSTRUCT.repo_id: QWEN2_5_3B_INSTRUCT}


class ModelCacheError(RuntimeError):
    """The model could not be put in place or does not match its pin."""


def models_for(setting: str, task_config: str, upstream: Path | None = None) -> list[PinnedModel]:
    """The pinned models a run of this setting and task config needs in its cache.

    Raises:
        ModelCacheError: the config names a model that has no pin here.
    """
    if setting not in PROVISIONED_SETTINGS:
        return []
    config = load_config(setting, upstream).get("configs", {}).get(task_config, {})
    repo_id = config.get("model")
    if not repo_id:
        return []
    if repo_id not in PINNED_MODELS:
        raise ModelCacheError(
            f"{setting}/{task_config} needs {repo_id}, which has no pinned revision"
        )
    return [PINNED_MODELS[repo_id]]


def parse_result(stdout: str) -> dict[str, Any] | None:
    """The script's `MODEL_CACHE_RESULT=` line, parsed; None if it printed none."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line.removeprefix(RESULT_PREFIX))
            except ValueError:
                return None
            return value if isinstance(value, dict) else None
    return None


async def provision_models(
    setting: str,
    task_config: str,
    mode: ModelCacheMode = "auto",
    upstream: Path | None = None,
) -> list[dict[str, Any]]:
    """Put the setting's pinned models into the agent's cache in the current sample's sandbox.

    Runs as the agent's user, after `HF_HOME` exists. Nothing happens for a setting outside
    `PROVISIONED_SETTINGS` or with `mode="none"`.

    Returns:
        One record per model: the script's result, its exit status, output and time.

    Raises:
        ModelCacheError: the script failed or a file does not match its pin.
    """
    if mode not in MODEL_CACHE_MODES:
        raise ValueError(f"model_cache must be one of {MODEL_CACHE_MODES}, got {mode!r}")
    models = models_for(setting, task_config, upstream)
    if mode == "none" or not models:
        return [{"repo_id": m.repo_id, "skipped": True} for m in models]
    box = sandbox_env()
    await box.write_file(SCRIPT_PATH, SCRIPT_SOURCE.read_bytes())
    chmod = await box.exec(["chmod", "644", SCRIPT_PATH], user="root")
    if not chmod.success:
        raise ModelCacheError(f"preparing the model cache script failed: {chmod.stderr}")
    records = []
    try:
        for model in models:
            start = time.monotonic()
            cmd = (
                f"umask 022; python3 {SCRIPT_PATH} --spec {shlex.quote(model.to_json())} "
                f"--hf-home {HF_HOME} --seed {SEED_MOUNT} --mode {mode}"
            )
            result = await box.exec(
                ["bash", "-c", cmd],
                user=AGENT_USER,
                env={"HOME": AGENT_HOME, "USER": AGENT_USER, "HF_HOME": HF_HOME},
            )
            parsed = parse_result(result.stdout)
            record = {
                "repo_id": model.repo_id,
                "revision": model.revision,
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - start, 1),
                "result": parsed,
                "output_tail": (result.stdout + result.stderr)[-4000:],
            }
            records.append(record)
            if not result.success or parsed is None or parsed.get("problems"):
                problems = parsed.get("problems") if parsed else None
                raise ModelCacheError(
                    f"putting {model.repo_id}@{model.revision[:8]} in {HF_HOME} failed "
                    f"({result.returncode}): {problems or record['output_tail']}"
                )
    finally:
        await box.exec(["rm", "-f", SCRIPT_PATH], user="root")
    return records


__all__ = [
    "MODEL_CACHE_MODES",
    "PINNED_MODELS",
    "PROVISIONED_SETTINGS",
    "QWEN2_5_3B_INSTRUCT",
    "SEED_MOUNT",
    "ModelCacheError",
    "ModelCacheMode",
    "PinnedFile",
    "PinnedModel",
    "models_for",
    "parse_result",
    "provision_models",
]
