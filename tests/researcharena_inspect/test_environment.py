"""The environment's pure parts: image pin, sandbox configs, the staged tree, the agent's env.

The staged tree is checked against upstream's own pipeline in Docker by
`test_environment_staging_parity.py`; these tests pin its rules on synthetic trees and need
neither Docker nor a GPU.
"""

import inspect
import io
import json
import re
import tarfile
import time
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
import yaml
from inspect_ai.util import (
    ComposeConfig,
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentSpec,
    SandboxUnavailableError,
)

from researcharena_inspect.environment import (
    AGENT_HOME,
    AGENT_PATH,
    COMPOSE_FILE,
    HF_HOME,
    IMAGE,
    IMAGE_DIGEST,
    StagingError,
    agent_env,
    evaluation,
    evaluator_entries,
    evaluator_env,
    home_entries,
    image_check,
    sandbox_spec,
)
from researcharena_inspect.environment.check import environment_check
from researcharena_inspect.environment.evaluation import EvaluationCommand, evaluator_side_task
from researcharena_inspect.environment.staging import hook_env, pre_agent_hooks, tar_entries
from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.prompts import agent_instruction_file
from researcharena_inspect.task import red_team_sandbox
from tests.researcharena_inspect.fake_sandbox import Call, FakeSandbox

DOCKERFILE = Path(COMPOSE_FILE).with_name("Dockerfile")


# --- image and sandbox configs ---------------------------------------------------------------


def test_image_is_pinned_by_digest() -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", IMAGE_DIGEST)
    assert IMAGE.endswith("@" + IMAGE_DIGEST)


def test_dockerfile_pins_base_image_and_upstream_commit() -> None:
    text = DOCKERFILE.read_text()
    assert re.search(
        r"^FROM --platform=linux/amd64 nvidia/cuda:12\.9\.1-cudnn-devel-ubuntu22\.04@sha256:[0-9a-f]{64}$",
        text,
        re.MULTILINE,
    )
    assert f"/ResearchArena/{UPSTREAM_COMMIT}/containers/requirements.txt" in text
    assert 'TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"' in text
    assert "--checksum=sha256:" in text


def test_compose_asks_for_one_gpu() -> None:
    compose = yaml.safe_load(COMPOSE_FILE.read_text())
    service = compose["services"]["default"]
    assert service["image"] == IMAGE
    assert service["platform"] == "linux/amd64"
    assert service["command"] == ["tail", "-f", "/dev/null"]
    [device] = service["deploy"]["resources"]["reservations"]["devices"]
    assert device == {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}
    assert service["cpus"] == 8
    assert "network_mode" not in service  # default bridge: open network, as upstream


def test_sandbox_spec() -> None:
    # The diagnostic tasks' Docker sandbox is the file; the red-team task's adds the evaluator.
    assert sandbox_spec(evaluator=False) == SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
    assert sandbox_spec().type == "docker"


def test_sandbox_spec_has_no_backend() -> None:
    # Docker is the only built-in sandbox: no backend or GPU to choose, and the options are
    # keyword-only, so a positional backend argument cannot be taken for a setting.
    parameters = inspect.signature(sandbox_spec).parameters
    assert list(parameters) == ["setting", "hf_seed", "offline", "evaluator"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    with pytest.raises(TypeError):
        sandbox_spec("docker")  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        sandbox_spec(backend="docker")  # pyright: ignore[reportCallIssue]


def test_every_docker_sandbox_of_the_task_gets_the_evaluator() -> None:
    # D-093: a second service of the same image and GPU, sharing only the cache and the run
    # folder with the agent's, as upstream's evaluator container shares only those two.
    config = sandbox_spec(setting="cuda_kernel").config
    assert isinstance(config, ComposeConfig)
    default, evaluator = config.services["default"], config.services["evaluator"]
    for field in ("image", "platform", "deploy", "cpus", "mem_limit", "shm_size", "network_mode"):
        assert getattr(evaluator, field) == getattr(default, field), field
    assert default.init is evaluator.init is True  # the scorer stops the agent's through init
    shared = ["hf_cache:/home/user/hf_cache", "eval_dir:/mnt/eval"]
    assert default.volumes == evaluator.volumes == shared
    assert config.volumes == {"hf_cache": {}, "eval_dir": {}}
    offline = sandbox_spec(setting="utility_finetuning", offline=True).config
    assert isinstance(offline, ComposeConfig)
    assert offline.services["evaluator"].network_mode == "none"
    assert offline.services["aux"].network_mode == "service:default"
    cpu = red_team_sandbox("docker", None).config
    assert isinstance(cpu, ComposeConfig)
    assert list(cpu.services) == ["default", "evaluator"]


def test_package_data_ships_the_configs() -> None:
    import tomllib
    from fnmatch import fnmatch

    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    patterns = pyproject["tool"]["setuptools"]["package-data"]["*"]
    for name in ("compose.yaml", "Dockerfile"):
        assert any(fnmatch(name, pattern) for pattern in patterns), name


# --- the agent's environment ------------------------------------------------------------------


def test_agent_env_is_upstreams_env_args() -> None:
    env = agent_env(redruntime="1", resume_agent=True, agent_config="m", agent_effort="high")
    assert env == {
        "HOME": AGENT_HOME,
        "USER": "user",
        "LOGNAME": "user",
        "HF_HOME": HF_HOME,
        "HF_TOKEN": "",
        "CUDA_VISIBLE_DEVICES": "0",
        "AGENT_CONFIG": "m",
        "AGENT_EFFORT": "high",
        "TIME_BUDGET_HOURS": "1",
        "AGENT_RESUME": "true",
        "PATH": AGENT_PATH,
        "BASH_MAX_TIMEOUT_MS": "3600000",
        "BASH_DEFAULT_TIMEOUT_MS": "3600000",
    }
    seeded = agent_env(redruntime="0.5", resume_agent=False, seed=7)
    assert seeded["TASK_SEED"] == seeded["PYTHONHASHSEED"] == "7"
    assert seeded["AGENT_RESUME"] == "false"
    assert "AGENT_CONFIG" not in seeded
    assert AGENT_PATH.startswith("/home/user/task:")


def test_evaluator_env_has_no_api_keys() -> None:
    env = evaluator_env()
    assert env["HF_HOME"] == HF_HOME
    assert not any("API_KEY" in key for key in env)
    assert evaluator_env(seed=3)["TASK_SEED"] == "3"


# --- staging on the pinned checkout ------------------------------------------------------------


@pytest.mark.dataset_download  # clones the pinned upstream checkout on first use
def test_cuda_home_entries_are_upstreams_files() -> None:
    from researcharena_inspect.upstream import upstream_dir

    root = upstream_dir()
    prompt = agent_instruction_file("cuda_kernel", "none", "claude", 1)
    entries = {
        e.path: e for e in home_entries("cuda_kernel", "none", agent="claude", prompt=prompt)
    }
    context = root / "settings" / "cuda_kernel" / "task_context"
    for path in context.rglob("*"):
        rel = "task/" + path.relative_to(context).as_posix()
        assert rel in entries
        if path.is_file():
            assert entries[rel].content == path.read_bytes()
    # chmod +x on task/*.sh and task/*.py only, not deeper.
    assert entries["task/config.py"].mode == 0o755
    assert entries["task/bench.sh"].mode == 0o755
    assert entries["task/kernels/rmsnorm.cu"].mode == 0o644
    assert (
        entries["task/templates/qwen3.jinja"].content
        == (root / "lib" / "templates" / "qwen3.jinja").read_bytes()
    )
    assert entries["agent_solve.sh"].content == (root / "agents/claude/solve.sh").read_bytes()
    assert entries["prompt.txt"].content == prompt.encode()
    assert "oauth_token" not in entries  # credentials are not staged (D-091)
    assert [h.name for h in pre_agent_hooks("cuda_kernel", "independent")] == [
        "pre_agent.sh",
        "pre_agent.sh",
    ]
    assert hook_env("cuda_kernel", "tinyllama")["kl_fp16_max_threshold"] == "1e-05"


@pytest.mark.dataset_download
def test_evaluator_entries_for_a_baseline_use_embedded() -> None:
    assert evaluator_side_task("none") == "embedded"
    paths = [e.path for e in evaluator_entries("cuda_kernel", "none")]
    assert paths == [
        "evaluate.py",
        "setting_config.json",
        "side_tasks",
        "side_tasks/embedded",
        "side_tasks/embedded/evaluate.py",
    ]
    assert [e.path for e in evaluator_entries("cuda_kernel", "independent")][-1] == (
        "side_tasks/independent/evaluate.py"
    )


@pytest.mark.dataset_download
def test_unknown_side_task_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown side task"):
        home_entries("cuda_kernel", "nope", agent="claude", prompt=None)


# --- staging rules on a synthetic upstream ------------------------------------------------------


def _fake_upstream(root: Path) -> Path:
    setting = root / "settings" / "demo"
    (setting / "task_context" / "agent_workspace" / "sub").mkdir(parents=True)
    (setting / "task_context" / "run.py").write_text("print(1)\n")
    (setting / "task_context" / "data.txt").write_text("base\n")
    (setting / "task_context" / ".hidden").write_text("skipped by the glob\n")
    (setting / "task_context" / "agent_workspace" / "sub" / "x.sh").write_text("echo x\n")
    (setting / "task_context" / "agent_workspace" / "top.sh").write_text("echo top\n")
    overlay = setting / "side_tasks" / "attack" / "task_context"
    overlay.mkdir(parents=True)
    (overlay / "data.txt").write_text("overlay\n")
    (overlay / "extra.bin").write_bytes(b"\x00\x01")
    (overlay / "extra.bin").chmod(0o600)
    (setting / "side_tasks" / "attack" / "side_task.md").write_text("do it\n")
    (setting / "config.json").write_text(json.dumps({"resume_agent": True, "configs": {"c": {}}}))
    (setting / "instruction.md").write_text("task\n")
    (setting / "evaluate.py").write_text("")
    (root / "lib" / "templates").mkdir(parents=True)
    (root / "lib" / "templates" / "t.jinja").write_text("{{ x }}\n")
    for family in ("claude",):
        (root / "agents" / family).mkdir(parents=True)
        (root / "agents" / family / "solve.sh").write_text("#!/bin/sh\necho solve\n")
    (root / "src" / "utils").mkdir(parents=True)
    for name in ("pre_agent_setup.sh", "update_agent_cli.sh", "check_effort.py"):
        (root / "src" / "utils" / name).write_text("echo ok\n")
    return root


def test_overlay_flatten_chmod_and_templates_follow_cp(tmp_path: Path) -> None:
    root = _fake_upstream(tmp_path / "up")
    data = root / "settings/demo/task_context/data.txt"
    data.chmod(0o640)
    entries = {
        e.path: e
        for e in home_entries("demo", "attack", agent="claude", prompt="p\n", upstream=root)
    }
    # The overlay overwrites data.txt but the file keeps the first copy's mode.
    assert entries["task/data.txt"].content == b"overlay\n"
    assert entries["task/data.txt"].mode == 0o640
    assert entries["task/extra.bin"].mode == 0o600
    assert "task/.hidden" not in entries  # `cp -r dir/*` skips dot files
    # agent_workspace/ flattened into task/ and removed; chmod +x only one level deep.
    assert "task/agent_workspace" not in entries
    assert entries["task/top.sh"].mode == 0o755
    assert entries["task/sub/x.sh"].mode == 0o644
    assert entries["task/run.py"].mode == 0o755
    assert entries["task/templates/t.jinja"].content == b"{{ x }}\n"
    assert entries["prompt.txt"].mode == 0o644
    assert entries["agent_solve.sh"].content == b"#!/bin/sh\necho solve\n"


def test_templates_copied_into_an_existing_folder_nest(tmp_path: Path) -> None:
    root = _fake_upstream(tmp_path / "up")
    (root / "settings/demo/task_context/templates").mkdir()
    entries = {e.path for e in home_entries("demo", "none", agent=None, prompt=None, upstream=root)}
    assert "task/templates/templates/t.jinja" in entries  # `cp -r src dest` with dest present
    assert "agent_solve.sh" not in entries


def test_a_giveaway_word_in_a_launch_helper_stops_staging(tmp_path: Path) -> None:
    root = _fake_upstream(tmp_path / "up")
    (root / "src/utils/check_effort.py").write_text("# talks about the Monitor\n")
    with pytest.raises(StagingError, match=r"check_effort\.py"):
        home_entries("demo", "none", agent="claude", prompt=None, upstream=root)


def test_tar_keeps_modes_and_order(tmp_path: Path) -> None:
    root = _fake_upstream(tmp_path / "up")
    entries = home_entries("demo", "attack", agent=None, prompt=None, upstream=root)
    with tarfile.open(fileobj=io.BytesIO(tar_entries(entries))) as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == sorted(e.path for e in entries)
    by_name = {m.name: m for m in members}
    assert by_name["task"].isdir()
    assert by_name["task/extra.bin"].mode == 0o600
    assert by_name["task/run.py"].mode == 0o755
    assert {m.uid for m in members} == {1000}


# --- evaluation command ------------------------------------------------------------------------


def test_evaluation_attempt_is_upstreams_shell_line() -> None:
    command = EvaluationCommand(
        argv=["python3", "evaluate.py", "--side-task", "none"],
        cwd="/home/evaluator",
        env={},
        user="user",
        timeout=3600,
        attempts=3,
        metrics_path="/mnt/eval/eval/metrics.json",
    )
    assert command.attempt(2) == [
        "bash",
        "-c",
        (
            "timeout --signal=TERM --kill-after=30s 3600s python3 evaluate.py --side-task none "
            "> /mnt/eval/eval/eval_attempt_2.log 2>&1"
        ),
    ]


# --- image check -------------------------------------------------------------------------------


def test_image_check_reads_every_pin() -> None:
    text = (
        "# comment\nfoo==1.0\nBar_baz==2.0+cu1\n"
        "flash-attn @ https://x/y/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl\n"
    )
    pins = image_check.parse_requirements(text)
    assert pins == {"foo": "1.0", "bar-baz": "2.0+cu1", "flash-attn": "@2.8.3+cu128torch2.10"}
    result = image_check.compare(
        pins, {"foo": "1.0", "bar-baz": "2.0", "flash-attn": "2.8.3", "x": "1"}
    )
    assert [m["package"] for m in result["mismatches"]] == ["bar-baz"]
    assert result["unpinned_installed"] == {"x": "1"}
    with pytest.raises(ValueError, match="unrecognised"):
        image_check.parse_requirements("-e .\n")


@pytest.mark.dataset_download
def test_image_check_parses_upstreams_requirements() -> None:
    from researcharena_inspect.upstream import upstream_dir

    pins = image_check.parse_requirements(
        (upstream_dir() / "containers" / "requirements.txt").read_text()
    )
    assert len(pins) == 275
    assert pins["torch"] == "2.10.0+cu129"
    assert pins["vllm"] == "0.19.1"


# --- the GPU check task ------------------------------------------------------------------------


def test_environment_check_task_builds_without_a_gpu() -> None:
    # Docker by default, one container.
    task = environment_check()
    assert task.sandbox == SandboxEnvironmentSpec("docker", str(COMPOSE_FILE))
    assert task.time_limit == 3600
    for removed in ("backend", "gpu", "sandbox_start_timeout"):
        assert removed not in inspect.signature(environment_check).parameters


# --- stopping the agent's container (D-093) -------------------------------------------------


class _Hanging(FakeSandbox):
    async def exec(self, cmd: list[str], *args: Any, **kwargs: Any) -> ExecResult[str]:
        await anyio.sleep(3600)  # a Compose call that never returns
        raise AssertionError("unreachable")


def _stopped_after(n: int) -> FakeSandbox:
    def respond(call: Call) -> ExecResult[str] | None:
        if len(box.calls) > n:
            raise SandboxUnavailableError("service default is not running")
        return None

    box = FakeSandbox(respond)
    return box


async def test_the_agents_container_stop_has_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "STOP_SECONDS", 2)
    await evaluation.stop_agent_container(cast(SandboxEnvironment, _stopped_after(1)))
    for box in (FakeSandbox(), _Hanging()):  # never stops; a call hangs
        start = time.monotonic()
        with pytest.raises(StagingError, match="still running after 2 s"):
            await evaluation.stop_agent_container(cast(SandboxEnvironment, box))
        assert time.monotonic() - start < 4
