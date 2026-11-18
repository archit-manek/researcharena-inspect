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

`utility_finetuning` (D-170): the agent fine-tunes the task config's model, which upstream's
cluster cache held because `preparation_code/download_hf_cache/resources.json` lists it, with
the two datasets the same file lists (`openai/gsm8k` main train and test, `ScaleAI/gsm1k`
test). Its agent and evaluator do not run offline (the network is open, D-029), so the cache is
not needed for the run to work; it is filled so that the agent finds what upstream's agent found,
the GSM8K test split and GSM1K included. The datasets go in as their repositories' files at a
pinned revision (`hub/datasets--...`, as `huggingface_hub` downloads them); upstream's cache
also held the `datasets` library's own Arrow copies (`datasets/`), which depend on that
library's version and are not reproduced. `resources.json`'s second model
(`HuggingFaceTB/SmolLM3-3B-Base`, 6 GB, for the `smollm3_3b` config) is left out, as D-131
leaves out the cluster's other models.
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

# D-130, D-131, D-170: only these settings get a model, and only the task config's one.
PROVISIONED_SETTINGS: Final = frozenset({"inference_optimization", "utility_finetuning"})
"""Settings whose agent finds the task config's `model` in its cache before it starts (offline
servers for inference; the model to fine-tune for capabilities)."""


@dataclass(frozen=True)
class PinnedFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedModel:
    """A Hugging Face repository at one commit, with every file's size and SHA-256."""

    repo_id: str
    revision: str
    files: tuple[PinnedFile, ...]
    repo_type: str = "model"
    """`model` (`hub/models--...`) or `dataset` (`hub/datasets--...`)."""
    builds: tuple[tuple[str | None, str, int], ...] = ()
    """For a dataset: the `datasets` library's prepared builds (config, split, records) that
    upstream's `download_resources.py` leaves in the cache by calling `load_dataset` for
    `resources.json`'s configs and splits, so that a load by name works with no network
    (D-187). A setting with more datasets adds its pins, with their builds, to
    `SETTING_DATASETS`."""

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

# D-170: read from the Hub API on 18 Nov 2026 (`/api/models/Qwen/Qwen2.5-0.5B-Instruct?blobs=true`):
# `main` has been this commit since 25 Sep 2024, before every published run. The weights'
# SHA-256 is its LFS object id; the other files were downloaded at the commit and hashed. Two
# published capabilities submissions have exactly these weights (D-171).
QWEN2_5_0_5B_INSTRUCT: Final = PinnedModel(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    files=(
        PinnedFile(
            ".gitattributes",
            1519,
            "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
        ),
        PinnedFile(
            "LICENSE", 11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
        ),
        PinnedFile(
            "README.md", 4917, "b19c806a904db6dc878a0462e70b551f6b7ac78dfbb88c2eb966ca2b9109ae15"
        ),
        PinnedFile(
            "config.json", 659, "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"
        ),
        PinnedFile(
            "generation_config.json",
            242,
            "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
        ),
        PinnedFile(
            "merges.txt",
            1671839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        PinnedFile(
            "model.safetensors",
            988097824,
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
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

# D-170: the two datasets of `utility_finetuning/preparation_code/download_hf_cache/resources.json`,
# at the commits the Hub's `main` had on 18 Nov 2026 (`openai/gsm8k` since 23 Mar 2026,
# `ScaleAI/gsm1k` since 1 Apr 2025). The revision upstream's cluster cache held is not recorded.
# Parquet and Arrow SHA-256 are LFS object ids; the small files were downloaded and hashed.
OPENAI_GSM8K: Final = PinnedModel(
    repo_id="openai/gsm8k",
    revision="740312add88f781978c0658806c59bc2815b9866",
    repo_type="dataset",
    builds=(("main", "train", 7473), ("main", "test", 1319)),  # D-187
    files=(
        PinnedFile(
            ".gitattributes",
            1610,
            "0209b6c969d1b93c1fc521b6da74b1561d6933e6be7b3a223c3aa9ef6a3f6dbc",
        ),
        PinnedFile(
            "README.md", 7935, "a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1"
        ),
        PinnedFile(
            "eval.yaml", 920, "a115b103292f0d24a1483de6b5eac826292bd6c66f7f9f8fc31ffbc560260984"
        ),
        PinnedFile(
            "main/test-00000-of-00001.parquet",
            419088,
            "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        ),
        PinnedFile(
            "main/train-00000-of-00001.parquet",
            2306545,
            "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
        ),
        PinnedFile(
            "socratic/test-00000-of-00001.parquet",
            486995,
            "477dba7028204b465491b8f346ec774262ccd77147d942a0881e94cc6da7c99e",
        ),
        PinnedFile(
            "socratic/train-00000-of-00001.parquet",
            2677259,
            "54eb5fd2105a9126ac6410541b2e9dbe0199701258957c9af02d9ed675c90378",
        ),
    ),
)
SCALEAI_GSM1K: Final = PinnedModel(
    repo_id="ScaleAI/gsm1k",
    revision="bc09569d09a614b9b530edc7f076fb214ac10493",
    repo_type="dataset",
    builds=((None, "test", 1205),),  # D-187
    files=(
        PinnedFile(
            ".gitattributes",
            2461,
            "e7a120ab07b1bc5b486be249e9fc6c83d59448d0093e1dfebe95d1566d9cafc0",
        ),
        PinnedFile(
            "README.md", 312, "88465e71c259d0be2904e15d150257fad5b0abe754c90692b08fc1a303d09559"
        ),
        PinnedFile(
            "data/dataset_dict.json",
            20,
            "24761d3c08510c22918dfca6bc94100b9e5cff3fa323b34f2cc11916fcf69064",
        ),
        PinnedFile(
            "data/test/data-00000-of-00001.arrow",
            389520,
            "bf7b848f2badbdcb138d6fbc1f246a1902337ce4c00cc079e4f35baf8ed10764",
        ),
        PinnedFile(
            "data/test/dataset_info.json",
            239,
            "882b153360935614dc9a2cb098815b4f4682d6af8c1c25a9a86109525ea5fc8b",
        ),
        PinnedFile(
            "data/test/state.json",
            247,
            "6faed36ab4458fc5bd8e4e78dcd859b7341a1dcf46bf9277e3bc77cd127d32b1",
        ),
    ),
)

PINNED_MODELS: Final[dict[str, PinnedModel]] = {
    m.repo_id: m for m in (QWEN2_5_3B_INSTRUCT, QWEN2_5_0_5B_INSTRUCT)
}

SETTING_DATASETS: Final[dict[str, tuple[PinnedModel, ...]]] = {
    # D-170: resources.json's datasets, in the cache whatever the task config.
    "utility_finetuning": (OPENAI_GSM8K, SCALEAI_GSM1K),
}
"""Dataset repositories a setting's agent finds in its cache besides the model."""


class ModelCacheError(RuntimeError):
    """The model could not be put in place or does not match its pin."""


def models_for(setting: str, task_config: str, upstream: Path | None = None) -> list[PinnedModel]:
    """The pinned repositories a run of this setting and task config needs in its cache.

    The task config's model, then the setting's datasets (`SETTING_DATASETS`).

    Raises:
        ModelCacheError: the config names a model that has no pin here.
    """
    if setting not in PROVISIONED_SETTINGS:
        return []
    config = load_config(setting, upstream).get("configs", {}).get(task_config, {})
    repo_id = config.get("model")
    pinned: list[PinnedModel] = []
    if repo_id:
        if repo_id not in PINNED_MODELS:
            raise ModelCacheError(
                f"{setting}/{task_config} needs {repo_id}, which has no pinned revision"
            )
        pinned.append(PINNED_MODELS[repo_id])
    return pinned + list(SETTING_DATASETS.get(setting, ()))


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


def seed_datasets(
    seed: Path, setting: str = "utility_finetuning", image: str | None = None
) -> None:
    """Prepare the setting's dataset builds in a host seed, with the network (D-187).

    Runs this module's script in the agent's image (its `datasets` version) in `download` mode
    with `seed` as `HF_HOME`: the pinned repositories go to `<seed>/hub/` (kept if already
    there) and each build to `<seed>/datasets/`, from where staging copies it. As
    `python -m researcharena_inspect.environment.model_cache <seed folder> [setting]`.
    """
    import os
    import subprocess

    from researcharena_inspect.environment.image import IMAGE

    seed.mkdir(parents=True, exist_ok=True)
    for pin in SETTING_DATASETS.get(setting, ()):
        subprocess.run(
            [
                *("docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"),
                *("-e", "HOME=/tmp", "-v", f"{seed.resolve()}:/seed"),
                *("-v", f"{SCRIPT_SOURCE}:/tmp/model_cache_script.py:ro"),
                *("--entrypoint", "python3", image or IMAGE, "/tmp/model_cache_script.py"),
                *("--spec", pin.to_json(), "--hf-home", "/seed", "--mode", "download"),
            ],
            check=True,
        )


if __name__ == "__main__":
    import sys

    seed_datasets(Path(sys.argv[1]), *sys.argv[2:3])


__all__ = [
    "MODEL_CACHE_MODES",
    "OPENAI_GSM8K",
    "PINNED_MODELS",
    "PROVISIONED_SETTINGS",
    "QWEN2_5_0_5B_INSTRUCT",
    "QWEN2_5_3B_INSTRUCT",
    "SCALEAI_GSM1K",
    "SEED_MOUNT",
    "SETTING_DATASETS",
    "ModelCacheError",
    "ModelCacheMode",
    "PinnedFile",
    "PinnedModel",
    "models_for",
    "parse_result",
    "provision_models",
    "seed_datasets",
]
