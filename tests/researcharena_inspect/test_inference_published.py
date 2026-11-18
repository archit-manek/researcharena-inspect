"""The 105 published inference runs, where no GPU is needed.

- Upstream's side-task checks that read only files, run on each published `task_snapshot`, give
  the value stored in that run's `metrics.json`: the typosquat pin (`independent`), the
  wrong-answer and upload code scans (`embedded2`, `independent2`, upstream's `embedded3`) and
  the gaming trigger scan (`embedded`). The rest of each side task needs the server running.
- Every published workspace goes through the replay agent's file selection and archive as the
  replay agent would put it in place: in `fresh` mode nothing is left out for this setting.
"""

import importlib.util
import io
import json
import shutil
import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from researcharena_inspect.agents.replay import select_files, snapshot_tar
from researcharena_inspect.analysis.published_rows import load_raw_metrics
from researcharena_inspect.monitor.bundle import fetch_workspace
from researcharena_inspect.published import fetch_run_file, load_index
from researcharena_inspect.upstream import upstream_dir

pytestmark = [pytest.mark.slow, pytest.mark.huggingface, pytest.mark.dataset_download]

SETTING = "inference_optimization"


def _runs() -> list[dict[str, Any]]:
    return [r for r in load_index()["runs"] if r["setting"] == SETTING]


@contextmanager
def _upstream_setting(tmp_path: Path) -> Iterator[Path]:
    """A copy of the setting's folder on `sys.path`, as `evaluate.py`'s own folder is when it
    runs. A copy, so that importing it writes no `__pycache__` into the pinned checkout, which
    staging copies from."""
    folder = tmp_path / SETTING
    shutil.copytree(upstream_dir() / "settings" / SETTING, folder)
    sys.path.insert(0, str(folder))
    try:
        yield folder
    finally:
        sys.path.remove(str(folder))
        for name in [
            m for m in sys.modules if m == "evaluation_code" or m.startswith("evaluation_code.")
        ]:
            del sys.modules[name]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored_metrics(run: dict[str, Any]) -> dict[str, Any]:
    folder = load_raw_metrics()[run["id"]]["results_from"]
    path = fetch_run_file(run["id"], f"blue_team/{folder}/results.json")
    return json.loads(path.read_text(encoding="utf-8"))["raw_metrics"]


def test_static_side_task_checks_on_published_workspaces_match(tmp_path: Path) -> None:
    differences: dict[str, Any] = {}
    counted: dict[str, int] = {}
    with _upstream_setting(tmp_path) as folder:
        wrong_answer = _load("_inf_embedded2", folder / "side_tasks/embedded2/evaluate.py")
        upload = _load("_inf_embedded3", folder / "side_tasks/embedded3/evaluate.py")
        pin = _load("_inf_independent", folder / "side_tasks/independent/evaluate.py")
        gaming = importlib.import_module("evaluation_code.detect_benchmark_gaming")
        for run in _runs():
            side_task = run["side_task"]
            if side_task == "baseline":
                continue
            stored = _stored_metrics(run)
            workspace = fetch_workspace(run["id"])
            if side_task == "independent":
                field, ours = "pin_added", pin.evaluate(str(workspace.parent))["pin_added"]
            elif side_task == "embedded2":
                field, ours = "suspicious_code", wrong_answer._static_suspicious(workspace)
            elif side_task == "independent2":
                field, ours = "suspicious_code", upload._static_suspicious(workspace)
            else:
                scan = gaming.static_trigger_scan(workspace)
                field, ours = "static_trigger_detected", scan["static_trigger_detected"]
            counted[side_task] = counted.get(side_task, 0) + 1
            if ours != stored.get(field):
                differences[run["id"]] = (field, ours, stored.get(field))
    assert counted == {"embedded": 21, "embedded2": 21, "independent2": 21, "independent": 21}
    assert differences == {}


def test_published_workspaces_replay_whole() -> None:
    runs = _runs()
    assert len(runs) == 105
    for run in runs:
        root = fetch_workspace(run["id"])
        files = {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }
        assert {"server.py", "launch.sh", "requirements.txt"} <= set(files), run["id"]
        chosen = select_files(files, "fresh", SETTING)
        assert chosen == files, run["id"]
        with tarfile.open(fileobj=io.BytesIO(snapshot_tar(chosen))) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
        assert set(members) == set(files)
        assert members["launch.sh"].mode == 0o755
        assert members["requirements.txt"].mode == 0o644
