"""Settings, side tasks and task configs, read from the pinned upstream checkout.

Mirrors upstream's discovery scripts at the pinned commit:

- `src/utils/settings_discovery.sh`: a setting is a directory under `settings/` that holds
  `config.json`, `instruction.md` and `evaluate.py`.
- `src/utils/side_task_discovery.sh`: a side task is a directory under
  `settings/<setting>/side_tasks/` that holds `side_task.md`. The literal `none` (the
  baseline) is always valid and is listed last.
- `src/pipeline_redteam.sh:66-73`: with no task config given, the run uses the first key
  of `configs` in `config.json`.

Every function takes an optional `upstream` folder (a checkout of upstream's tree) and
defaults to the pinned checkout from `researcharena_inspect.upstream.upstream_dir()`.
"""

import json
from pathlib import Path
from typing import Any

from researcharena_inspect.upstream import upstream_dir

# The side task upstream uses for a baseline run (`run.sh --baseline` passes "none").
BASELINE = "none"

_SETTING_FILES = ("config.json", "instruction.md", "evaluate.py")


def upstream_root(upstream: Path | None = None) -> Path:
    """The upstream tree to read: `upstream` if given, else the pinned checkout."""
    return upstream if upstream is not None else upstream_dir()


def settings_root(upstream: Path | None = None) -> Path:
    """Upstream's `settings/` folder."""
    return upstream_root(upstream) / "settings"


def _subdirectories(folder: Path) -> list[Path]:
    # Bash's `"$dir"/*/` glob: directories (or links to them), not hidden ones.
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if not p.name.startswith(".") and p.is_dir()]


def is_valid_setting(setting: str, upstream: Path | None = None) -> bool:
    """Whether `setting` names a setting (upstream's `is_valid_setting`)."""
    if not setting:
        return False
    folder = settings_root(upstream) / setting
    return all((folder / name).is_file() for name in _SETTING_FILES)


def list_settings(upstream: Path | None = None) -> list[str]:
    """Names of the settings, sorted (upstream's `list_available_settings`).

    Upstream pipes the names through `sort`; Python's sort equals `sort` in the C locale,
    and in any locale for upstream's names, which use only `[a-z0-9_]`.
    """
    root = settings_root(upstream)
    names = [p.name for p in _subdirectories(root) if is_valid_setting(p.name, upstream)]
    return sorted(names)


def is_valid_side_task(setting: str, side_task: str, upstream: Path | None = None) -> bool:
    """Whether `side_task` is valid for `setting` (upstream's `is_valid_side_task`).

    `none` (the baseline) is always valid.
    """
    if not side_task:
        return False
    if side_task == BASELINE:
        return True
    return (settings_root(upstream) / setting / "side_tasks" / side_task / "side_task.md").is_file()


def list_side_tasks(setting: str, upstream: Path | None = None) -> list[str]:
    """The setting's side tasks, sorted, then `none` (upstream's `list_available_side_tasks`)."""
    folder = settings_root(upstream) / setting / "side_tasks"
    names = sorted(p.name for p in _subdirectories(folder) if (p / "side_task.md").is_file())
    return [*names, BASELINE]


def load_config(setting: str, upstream: Path | None = None) -> dict[str, Any]:
    """The setting's `config.json`."""
    path = settings_root(upstream) / setting / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def task_configs(setting: str, upstream: Path | None = None) -> list[str]:
    """The setting's task config names, in `config.json` order."""
    return list(load_config(setting, upstream).get("configs", {}))


def default_task_config(setting: str, upstream: Path | None = None) -> str:
    """The task config a run uses when none is given: the first in `config.json`.

    Upstream falls back to the literal "default" when the config has no `configs`
    (`pipeline_redteam.sh:66-73`); every pinned setting has at least one.
    """
    names = task_configs(setting, upstream)
    return names[0] if names else "default"


def resume_agent(setting: str, upstream: Path | None = None) -> bool:
    """Whether the red-team agent is re-prompted until its time budget is spent.

    This is `resume_agent` in the setting's `config.json`, which upstream requires to be a
    bool (`pipeline_redteam.sh:83-90`, `prompt_builder.py:82-88`).

    Raises:
        ValueError: `resume_agent` is missing or not a bool.
    """
    value = load_config(setting, upstream).get("resume_agent")
    if not isinstance(value, bool):
        raise ValueError(f"{setting}/config.json needs resume_agent: true or false, got {value!r}")
    return value
