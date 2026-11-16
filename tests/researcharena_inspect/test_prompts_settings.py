"""Settings discovery agrees with upstream's discovery scripts."""

import json
import subprocess
from pathlib import Path

import pytest

from researcharena_inspect.prompts import settings
from researcharena_inspect.upstream import upstream_dir

pytestmark = pytest.mark.dataset_download  # clones upstream on first use


def _upstream_lines(function: str, *args: str) -> list[str]:
    root = upstream_dir()
    script = (
        f'source "{root}/src/utils/settings_discovery.sh"; '
        f'source "{root}/src/utils/side_task_discovery.sh"; '
        f'{function} "$@"'
    )
    result = subprocess.run(
        ["bash", "-c", script, "_", *args], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def _upstream_true(function: str, *args: str) -> bool:
    root = upstream_dir()
    script = (
        f'source "{root}/src/utils/settings_discovery.sh"; '
        f'source "{root}/src/utils/side_task_discovery.sh"; '
        f'{function} "$@"'
    )
    result = subprocess.run(["bash", "-c", script, "_", *args], capture_output=True, check=False)
    return result.returncode == 0


def test_settings_match_upstream_listing() -> None:
    expected = _upstream_lines("list_available_settings", str(settings.settings_root()))
    assert settings.list_settings() == expected
    assert expected == [
        "cuda_kernel",
        "inference_optimization",
        "safety_finetuning",
        "utility_finetuning",
    ]


@pytest.mark.parametrize(
    "setting",
    ["cuda_kernel", "inference_optimization", "safety_finetuning", "utility_finetuning"],
)
def test_side_tasks_match_upstream_listing(setting: str) -> None:
    expected = _upstream_lines("list_available_side_tasks", str(settings.settings_root() / setting))
    assert settings.list_side_tasks(setting) == expected
    assert expected[-1] == settings.BASELINE


SETTING_CANDIDATES = ["cuda_kernel", "", "none", "bogus", "../settings/cuda_kernel", "side_tasks"]
SIDE_TASK_CANDIDATES = ["embedded", "embedded2", "none", "", "baseline", "independent2", "x"]


@pytest.mark.parametrize("candidate", SETTING_CANDIDATES)
def test_setting_validity_matches_upstream(candidate: str) -> None:
    expected = _upstream_true("is_valid_setting", candidate, str(settings.settings_root()))
    assert settings.is_valid_setting(candidate) == expected


@pytest.mark.parametrize("candidate", SIDE_TASK_CANDIDATES)
@pytest.mark.parametrize("setting", ["cuda_kernel", "safety_finetuning"])
def test_side_task_validity_matches_upstream(setting: str, candidate: str) -> None:
    setting_dir = str(settings.settings_root() / setting)
    expected = _upstream_true("is_valid_side_task", candidate, setting_dir)
    assert settings.is_valid_side_task(setting, candidate) == expected


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_discovery_rules_match_upstream_on_a_synthetic_tree(tmp_path: Path) -> None:
    root = tmp_path / "settings"
    for name in ("b_setting", "a_setting", "a2"):
        for f in ("config.json", "instruction.md", "evaluate.py"):
            _write(root / name / f)
    _write(root / "no_evaluate" / "config.json")
    _write(root / "no_evaluate" / "instruction.md")
    for f in ("config.json", "instruction.md", "evaluate.py"):
        _write(root / ".hidden" / f)
    _write(root / "a_setting" / "side_tasks" / "zeta" / "side_task.md")
    _write(root / "a_setting" / "side_tasks" / "alpha_2" / "side_task.md")
    _write(root / "a_setting" / "side_tasks" / "alpha2" / "side_task.md")
    _write(root / "a_setting" / "side_tasks" / "no_md" / "evaluate.py")
    _write(root / "a_setting" / "side_tasks" / ".hidden" / "side_task.md")
    _write(root / "a_setting" / "side_tasks" / "file_not_dir")

    assert settings.list_settings(tmp_path) == _upstream_lines("list_available_settings", str(root))
    for name in ("a_setting", "b_setting"):
        assert settings.list_side_tasks(name, tmp_path) == _upstream_lines(
            "list_available_side_tasks", str(root / name)
        )


def test_task_configs_keep_config_json_order() -> None:
    # pipeline_redteam.sh:66-73 takes the first key; the agent parity tests run that block.
    for setting in settings.list_settings():
        config_path = settings.settings_root() / setting / "config.json"
        names = list(json.loads(config_path.read_text(encoding="utf-8"))["configs"])
        assert settings.task_configs(setting) == names
        assert settings.default_task_config(setting) == names[0]


def test_resume_agent_per_setting() -> None:
    assert {s: settings.resume_agent(s) for s in settings.list_settings()} == {
        "cuda_kernel": True,
        "inference_optimization": True,
        "safety_finetuning": False,
        "utility_finetuning": False,
    }


def test_resume_agent_must_be_a_bool(tmp_path: Path) -> None:
    _write(tmp_path / "settings" / "s" / "config.json", json.dumps({"configs": {"c": {}}}))
    with pytest.raises(ValueError, match="resume_agent"):
        settings.resume_agent("s", tmp_path)
