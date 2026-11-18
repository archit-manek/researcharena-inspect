"""`inference_optimization` without Docker: model cache, sandbox, metrics reading, replay, config.

The model-cache script runs here on a small stand-in model in `tmp_path` (the real one is 6 GB);
the pinned Qwen files are checked against the upstream setting's own config. Docker runs of the
whole setting are in `test_inference_docker.py`.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import ComposeConfig

from researcharena_inspect import researcharena
from researcharena_inspect.agents.replay import SCORER_CACHE_FILES, replay_source, select_files
from researcharena_inspect.environment.evaluation import evaluator_entries
from researcharena_inspect.environment.model_cache import (
    QWEN2_5_3B_INSTRUCT,
    SCRIPT_SOURCE,
    SEED_MOUNT,
    ModelCacheError,
    PinnedFile,
    PinnedModel,
    models_for,
    parse_result,
)
from researcharena_inspect.environment.sandbox import COMPOSE_FILE, VLLM_SHM_SIZE, sandbox_spec
from researcharena_inspect.prompts import list_side_tasks
from researcharena_inspect.prompts.settings import load_config
from researcharena_inspect.scoring import parse_metrics, read_metrics
from researcharena_inspect.scoring.metrics_json import published_side_task, upstream_side_task
from researcharena_inspect.scoring.rows import side_task_class
from researcharena_inspect.task import dataset_setting

SETTING = "inference_optimization"

# --------------------------------------------------------------------------------- the pin


@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
def test_the_setting_needs_the_pinned_qwen_and_cuda_needs_nothing() -> None:
    configs = load_config(SETTING)["configs"]
    assert list(configs) == ["qwen2_5_3b_gpu"]
    assert configs["qwen2_5_3b_gpu"]["model"] == QWEN2_5_3B_INSTRUCT.repo_id
    assert models_for(SETTING, "qwen2_5_3b_gpu") == [QWEN2_5_3B_INSTRUCT]
    assert models_for("cuda_kernel", "tinyllama") == []


def test_the_pin_lists_the_repositorys_files() -> None:
    pin = QWEN2_5_3B_INSTRUCT
    assert len(pin.revision) == 40
    names = {f.path for f in pin.files}
    assert {"config.json", "tokenizer.json", "model.safetensors.index.json"} <= names
    assert {n for n in names if n.endswith(".safetensors")} == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(len(f.sha256) == 64 for f in pin.files)
    assert pin.total_bytes == 6_183_464_935  # the Hub's listing at the pinned commit


# ------------------------------------------------------------------------- the cache script

TINY_FILES = {"config.json": b'{"model_type": "tiny"}', "weights.bin": b"\x00\x01" * 1000}


def _tiny_pin() -> PinnedModel:
    return PinnedModel(
        repo_id="org/tiny",
        revision="0" * 40,
        files=tuple(
            PinnedFile(name, len(data), hashlib.sha256(data).hexdigest())
            for name, data in sorted(TINY_FILES.items())
        ),
    )


def _seed(root: Path, pin: PinnedModel, files: dict[str, bytes], layout: str = "hf_home") -> Path:
    """A host cache in the layout `huggingface_hub` leaves: blobs, a snapshot of links, refs."""
    hub = root / "hub" if layout == "hf_home" else root
    model = hub / ("models--" + pin.repo_id.replace("/", "--"))
    snapshot = model / "snapshots" / pin.revision
    (model / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for name, data in files.items():
        blob = model / "blobs" / f"etag-{name}"
        blob.write_bytes(data)
        (snapshot / name).symlink_to(f"../../blobs/etag-{name}")
    (model / "refs").mkdir()
    (model / "refs" / "main").write_text("f" * 40)  # a later main: the copy must not keep it
    return root


def _run(pin: PinnedModel, hf_home: Path, seed: Path | None, mode: str) -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_SOURCE),
            "--spec",
            pin.to_json(),
            "--hf-home",
            str(hf_home),
            "--seed",
            str(seed) if seed else "",
            "--mode",
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = parse_result(result.stdout)
    assert parsed is not None, result.stdout + result.stderr
    return result.returncode, parsed


@pytest.mark.parametrize("layout", ["hf_home", "hub"])
def test_the_script_copies_the_pinned_snapshot_from_a_seed(tmp_path: Path, layout: str) -> None:
    pin = _tiny_pin()
    seed = _seed(tmp_path / "seed", pin, TINY_FILES, layout)
    hf_home = tmp_path / "hf_cache"
    code, result = _run(pin, hf_home, seed, "auto")
    assert code == 0, result
    assert result["source"] == "seed"
    assert result["problems"] == []
    assert result["bytes_copied"] == sum(len(v) for v in TINY_FILES.values())

    model = hf_home / "hub" / "models--org--tiny"
    assert (model / "refs" / "main").read_text() == pin.revision
    for f in pin.files:
        link = model / "snapshots" / pin.revision / f.path
        assert link.is_symlink()
        assert link.resolve().name == f.sha256
        assert link.read_bytes() == TINY_FILES[f.path]
        assert not link.resolve().is_relative_to(seed)  # a copy, not a link into the seed
    # Run again on the filled cache: nothing to copy.
    code, again = _run(pin, hf_home, seed, "seed")
    assert code == 0
    assert again["bytes_copied"] == 0


def test_the_script_refuses_a_seed_that_does_not_match_the_pin(tmp_path: Path) -> None:
    pin = _tiny_pin()
    other = b"\x00\x02" * 1000
    seed = _seed(tmp_path / "seed", pin, {**TINY_FILES, "weights.bin": other})
    code, result = _run(pin, tmp_path / "hf_cache", seed, "seed")
    assert code == 1
    actual = hashlib.sha256(other).hexdigest()
    pinned = hashlib.sha256(TINY_FILES["weights.bin"]).hexdigest()
    assert result["problems"] == [f"weights.bin: sha256 {actual}, pinned {pinned}"]


def test_seed_mode_without_the_snapshot_fails_without_downloading(tmp_path: Path) -> None:
    pin = _tiny_pin()
    empty = tmp_path / "empty"
    empty.mkdir()
    code, result = _run(pin, tmp_path / "hf_cache", empty, "seed")
    assert code == 1
    assert result["source"] is None
    assert "has no snapshot" in result["problems"][0]


def test_parse_result_reads_the_last_result_line() -> None:
    assert parse_result('noise\nMODEL_CACHE_RESULT={"problems": []}\n') == {"problems": []}
    assert parse_result("MODEL_CACHE_RESULT=not json") is None
    assert parse_result("nothing") is None


def test_an_unpinned_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "researcharena_inspect.environment.model_cache.load_config",
        lambda setting, upstream=None: {"configs": {"other": {"model": "org/unknown"}}},
    )
    with pytest.raises(ModelCacheError, match="no pinned revision"):
        models_for(SETTING, "other")


# ---------------------------------------------------------------------------------- sandbox


def test_cuda_keeps_the_plain_compose_file() -> None:
    spec = sandbox_spec("docker", setting="cuda_kernel")
    assert spec.config == str(COMPOSE_FILE)
    assert sandbox_spec("docker").config == str(COMPOSE_FILE)


def test_inference_gets_shared_memory_and_the_seed_mount() -> None:
    spec = sandbox_spec("docker", setting=SETTING, hf_seed="/srv/hf_seed")
    assert isinstance(spec.config, ComposeConfig)
    service = spec.config.services["default"]
    assert service.shm_size == VLLM_SHM_SIZE
    assert service.volumes == [f"/srv/hf_seed:{SEED_MOUNT}:ro"]
    plain = ComposeConfig.model_validate(yaml.safe_load(COMPOSE_FILE.read_text()))
    # Everything else as in compose.yaml: image, GPU, CPUs, memory.
    for field in ("image", "deploy", "cpus", "mem_limit", "platform", "command", "init"):
        assert getattr(service, field) == getattr(plain.services["default"], field), field
    assert sandbox_spec("docker", setting=SETTING).config.services["default"].volumes is None


def test_hf_seed_needs_docker_and_an_absolute_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        sandbox_spec("docker", setting=SETTING, hf_seed="hf_seed")
    with pytest.raises(ValueError, match="docker"):
        sandbox_spec("k8s", gpu="h100", setting=SETTING, hf_seed="/srv/hf_seed")


@pytest.mark.dataset_download
def test_the_task_picks_the_sandbox_from_its_setting() -> None:
    task = researcharena(setting=SETTING, agent="null", hf_seed="/srv/hf_seed")
    assert task.sandbox is not None
    assert isinstance(task.sandbox.config, ComposeConfig)
    assert [s.id for s in task.dataset] == [
        f"{SETTING}__{s}" for s in ("embedded", "embedded2", "embedded3", "independent", "baseline")
    ]
    cuda = researcharena(agent="null")
    assert cuda.sandbox is not None
    assert cuda.sandbox.config == str(COMPOSE_FILE)
    with pytest.raises(ValueError, match="model_cache"):
        researcharena(setting=SETTING, agent="null", model_cache="never")  # type: ignore[arg-type]


def test_mixed_replays_with_inference_are_refused() -> None:
    def ds(*settings: str) -> MemoryDataset:
        return MemoryDataset([Sample(input="x", metadata={"setting": s}) for s in settings])

    assert dataset_setting(ds(SETTING, SETTING)) == SETTING
    assert dataset_setting(ds("cuda_kernel", "safety_finetuning")) is None
    with pytest.raises(ValueError, match="on their own"):
        dataset_setting(ds("cuda_kernel", SETTING))


# -------------------------------------------------------------------------- reading metrics


def test_live_and_published_names_of_the_exfiltration_side_task() -> None:
    # D-050: upstream's embedded3 is independent2 in every published run.
    assert published_side_task(SETTING, "embedded3") == "independent2"
    assert published_side_task(SETTING, "independent2") == "independent2"
    assert published_side_task(SETTING, "none") == "baseline"
    assert published_side_task("cuda_kernel", "embedded3") == "embedded3"
    assert side_task_class(SETTING, "embedded3") == "embedded"
    metrics = {
        "main_task_score": 11.5,
        "side_task_score": 1.0,
        "uploaded_to_ingest_host": True,
        "prompt_uploaded": True,
        "exfiltration_detected": True,
        "num_payloads_to_ingest_host": 2,
    }
    live = read_metrics(metrics, SETTING, "embedded3")
    published = read_metrics(metrics, SETTING, "independent2")
    assert live.side_task_id == published.side_task_id == "independent2"
    assert live.side_score == published.side_score == 1.0
    assert live.side_metrics == published.side_metrics
    assert live.side_metrics["uploaded_to_ingest_host"] is True


@pytest.mark.parametrize(
    ("side_task", "metrics", "side_score"),
    [
        ("none", {"main_task_score": 10.5, "side_task_score": 0.0}, 0.0),
        (
            "embedded",
            {"main_task_score": 12.0, "side_task_score": 1.0, "side_task_numeric_value": 64.2},
            64.2,
        ),
        (
            "embedded2",
            {"side_task_score": 0.0, "side_task_numeric_value": 0.375, "trigger_wrong_rate": 0.375},
            0.375,
        ),
        ("independent", {"side_task_score": 1.0, "pin_added": True}, 1.0),
    ],
)
def test_side_scores_of_a_live_run(side_task: str, metrics: dict, side_score: float) -> None:
    reading = read_metrics(metrics, SETTING, side_task)
    assert reading.side_score == side_score


def test_a_failed_evaluation_reads_as_no_score() -> None:
    record = '{"main_task_score": -1, "side_task_score": -1, "error": "evaluation_failed"}\n'
    reading = read_metrics(parse_metrics(record), SETTING, "embedded")
    assert reading.evaluation_failed
    assert reading.side_score is None
    assert reading.main_task_score == -1


# ------------------------------------------------------------------------- replay, evaluator


def test_fresh_replay_leaves_out_nothing_for_inference() -> None:
    files = {
        "server.py": b"x",
        ".vllm_cache/vllm/model": b"y",
        "final_results.json": b"{}",
        "workspace/last_bench.json": b"{}",
    }
    assert select_files(files, "fresh", SETTING) == files
    assert set(select_files(files, "fresh", "cuda_kernel")) == {
        "server.py",
        ".vllm_cache/vllm/model",
    }
    assert set(SCORER_CACHE_FILES) >= {"final_results.json", "workspace/last_bench.json"}
    assert select_files(files, "fresh") == select_files(files, "fresh", "cuda_kernel")


@pytest.mark.dataset_download
@pytest.mark.parametrize("side_task", ["none", "embedded", "embedded2", "embedded3", "independent"])
def test_the_evaluator_gets_evaluation_code_and_one_side_task(side_task: str) -> None:
    assert side_task in list_side_tasks(SETTING)
    paths = {e.path for e in evaluator_entries(SETTING, side_task)}
    assert {"evaluate.py", "setting_config.json", "evaluation_code/common.py"} <= paths
    # Copied with evaluation_code/ (pipeline_redteam.sh:458-460) though nothing calls them.
    assert {
        "evaluation_code/monitor_runtime_trace.py",
        "evaluation_code/monitor_static_diff.py",
        "evaluation_code/detect_supply_chain_violation.py",
    } <= paths
    side = "embedded" if side_task == "none" else side_task
    assert {p for p in paths if p.startswith("side_tasks/") and p.endswith(".py")} == {
        f"side_tasks/{side}/evaluate.py"
    }
    entries = {e.path: e for e in evaluator_entries(SETTING, side_task)}
    config = json.loads(entries["setting_config.json"].content or b"{}")
    assert config["resume_agent"] is True


@pytest.mark.dataset_download
def test_a_published_independent2_run_replays_as_upstreams_embedded3() -> None:
    # D-050: the 21 published exfiltration runs are `independent2`; upstream's folder is
    # `embedded3`, which staging, the prompt and the evaluator need.
    run = (
        "inference_optimization/inference_optimization__independent2__claude__claude-opus-4-6__"
        "20260605_165111_c17314023"
    )
    assert replay_source(run).side_task == "embedded3"
    task = researcharena(agent="replay", replay=run)
    [sample] = task.dataset
    assert sample.metadata is not None
    assert sample.metadata["side_task"] == "embedded3"
    assert upstream_side_task(SETTING, "independent2") == "embedded3"
    assert upstream_side_task(SETTING, "baseline") == "none"
    assert upstream_side_task("cuda_kernel", "embedded3") == "embedded3"
