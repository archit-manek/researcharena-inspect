"""Staging parity with upstream's own `pipeline_redteam.sh`, both run in Docker (pass rule 2).

For `cuda_kernel` baseline and each side task, the agent's home the port stages in an Inspect
Docker sandbox must equal the home upstream's pipeline stages: every path, type, mode, size
and SHA-256, and the same git repository (branch, tree, commits, index, untracked files,
objects). After a stand-in for the agent's work, the evaluator's job folder and the
`task_snapshot` it is given must be equal too.

Same image on both sides (`python:3.12-bookworm`), same fixed `date +%s`, same git identity.
Differences by design: the port also has the empty model-cache folder `hf_cache` in the agent's
home and in the evaluator's (upstream's is a bind mount in both containers; the harness's host
branch has none), and the port's files belong to uid 1000. The port's evaluator's job folder
and snapshot are read in the evaluator's own container (D-093).
"""

import json
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

from researcharena_inspect.environment import (
    AGENT_HOME,
    AGENT_UID,
    AGENT_USER,
    EVAL_HOME,
    EVALUATOR_SERVICE,
    SNAPSHOT_DIR,
    TASK_DIR,
    prepare_evaluation,
    stage_agent_workspace,
)
from researcharena_inspect.environment.staging import STAGING_METADATA_KEY, AgentFamily
from researcharena_inspect.prompts import list_side_tasks
from tests.researcharena_inspect.upstream_redteam_pipeline import (
    AGENT_WORK,
    DATE_SHIM,
    MANIFEST_SOURCE,
    PipelineRun,
    docker_available,
    export_arena,
    port_boxes,
    port_sandbox,
    prepare_work,
    run_pipeline,
)

pytestmark = [
    pytest.mark.docker,
    pytest.mark.slow,
    pytest.mark.dataset_download,  # clones the pinned upstream checkout on first use
    pytest.mark.skipif(not docker_available(), reason="needs a running Docker daemon"),
]

SETTING = "cuda_kernel"
FAMILIES: list[AgentFamily] = ["claude", "codex", "opencode"]
CLAUDE: AgentFamily = "claude"
CASES: list[tuple[str, AgentFamily]] = [
    *((side_task, CLAUDE) for side_task in list_side_tasks(SETTING)),
    ("none", "codex"),
    ("none", "opencode"),
]
PORT_ONLY = {"hf_cache"}


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], PipelineRun]:
    work = tmp_path_factory.mktemp("pipeline")
    arena = export_arena(work / "arena")
    prepare_work(work)
    return {
        (side_task, agent): run_pipeline(work, arena, side_task=side_task, agent=agent)
        for side_task, agent in CASES
    }


async def _manifest(root: str, service: str = "default") -> dict[str, Any]:
    result = await sandbox(service).exec(
        ["python3", "/opt/test/tree_manifest.py", root, "--owner"], user=AGENT_USER
    )
    assert result.success, result.stderr
    return json.loads(result.stdout)


@solver
def install_test_tools() -> Solver:
    """The fixed `date` and the manifest program, as in upstream's container."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for box in port_boxes():
            await box.write_file("/usr/local/bin/date", DATE_SHIM)
            await box.write_file("/opt/test/tree_manifest.py", MANIFEST_SOURCE.read_text())
            await box.write_file("/opt/test/agent_work.sh", AGENT_WORK)
            chmod = await box.exec(["chmod", "-R", "a+rx", "/usr/local/bin/date", "/opt/test"])
            assert chmod.success
        return state

    return solve


@solver
def capture_and_prepare_evaluation() -> Solver:
    """Record the staged home, do the stand-in agent work, prepare the evaluator, record it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata["home"] = await _manifest(AGENT_HOME)
        work = await sandbox().exec(
            ["sh", "/opt/test/agent_work.sh"], user=AGENT_USER, cwd=TASK_DIR
        )
        assert work.success, work.stderr
        command = await prepare_evaluation(SETTING, state.metadata["side_task"])
        state.metadata["eval_home"] = await _manifest(EVAL_HOME, EVALUATOR_SERVICE)
        state.metadata["snapshot"] = await _manifest(SNAPSHOT_DIR, EVALUATOR_SERVICE)
        state.metadata["eval_argv"] = command.argv
        state.metadata["eval_user"] = command.user
        state.metadata["eval_cwd"] = command.cwd
        return state

    return solve


@pytest.fixture(scope="module")
def port(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str], dict[str, Any]]:
    folder = tmp_path_factory.mktemp("port")
    # One task per agent family: the setup solver stages that family's launch helpers.
    tasks = [
        Task(
            name=f"staging_{family}",
            dataset=MemoryDataset(
                [
                    Sample(
                        id=f"{side_task}-{agent}",
                        input="stage",
                        metadata={
                            "setting": SETTING,
                            "side_task": side_task,
                            "agent": agent,
                            "redruntime": "1",
                        },
                    )
                    for side_task, agent in CASES
                    if agent == family
                ]
            ),
            solver=[
                install_test_tools(),
                stage_agent_workspace(agent=family),
                capture_and_prepare_evaluation(),
            ],
            sandbox=port_sandbox(),
        )
        for family in FAMILIES
    ]
    logs = eval(tasks, model="mockllm/model", log_dir=str(folder / "logs"), display="none")
    out = {}
    for log in logs:
        assert log.status == "success", log.error
        assert log.samples is not None
        for sample in log.samples:
            assert sample.error is None, sample.error
            out[(sample.metadata["side_task"], sample.metadata["agent"])] = sample.metadata
    return out


def _assert_same_git(ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    assert sorted(ours) == sorted(theirs)
    for repo in theirs:
        for key in theirs[repo]:
            assert ours[repo][key] == theirs[repo][key], (repo, key)


RECENT = 1_600_000_000
"""Sep 2020. Upstream's `cp -r` gives copies the time they are made; a staged file must not
keep an archive's default time (1970)."""


def _without_owner(manifest: dict[str, Any]) -> dict[str, Any]:
    return {path: entry[:-1] for path, entry in manifest["entries"].items()}


@pytest.mark.parametrize(("side_task", "agent"), CASES)
def test_agent_home_matches_pipeline(
    oracle: dict[tuple[str, str], PipelineRun],
    port: dict[tuple[str, str], dict[str, Any]],
    side_task: str,
    agent: str,
) -> None:
    upstream = oracle[(side_task, agent)]
    assert upstream.home, upstream.log
    assert upstream.agent_cwd.endswith("/home/task")
    mine = port[(side_task, agent)]["home"]

    ours = _without_owner(mine)
    assert set(ours) - set(upstream.home["entries"]) == PORT_ONLY
    for path in PORT_ONLY:
        del ours[path]
    assert ours == upstream.home["entries"]
    _assert_same_git(mine["git"], upstream.home["git"])
    assert {entry[-1] for entry in mine["entries"].values()} == {AGENT_UID}
    assert upstream.home["oldest_mtime"] > RECENT
    assert mine["oldest_mtime"] > RECENT

    # What the run's case is about.
    git = mine["git"]["task"]
    assert git["log"] == ["Initial starter kernels|user|user@localhost"]
    untracked = [line for line in git["status"] if line.startswith("??")]
    assert untracked == (["?? requirements.txt"] if side_task == "independent" else [])
    assert ours["agent_solve.sh"][3] == upstream.home["entries"]["agent_solve.sh"][3]
    hooks = port[(side_task, agent)][STAGING_METADATA_KEY]["hooks"]
    expected = ["settings/cuda_kernel/hooks/pre_agent.sh"]
    if side_task == "independent":
        expected.append("settings/cuda_kernel/side_tasks/independent/hooks/pre_agent.sh")
    assert [h["hook"] for h in hooks] == expected
    assert all(h["returncode"] == 0 for h in hooks)


@pytest.mark.parametrize(("side_task", "agent"), CASES)
def test_evaluator_view_matches_pipeline(
    oracle: dict[tuple[str, str], PipelineRun],
    port: dict[tuple[str, str], dict[str, Any]],
    side_task: str,
    agent: str,
) -> None:
    upstream = oracle[(side_task, agent)]
    assert upstream.eval_home is not None, upstream.log
    assert upstream.snapshot is not None, upstream.log
    mine = port[(side_task, agent)]

    eval_home = _without_owner(mine["eval_home"])
    assert set(eval_home) - set(upstream.eval_home["entries"]) == PORT_ONLY
    for path in PORT_ONLY:
        del eval_home[path]
    assert eval_home == upstream.eval_home["entries"]
    assert _without_owner(mine["snapshot"]) == upstream.snapshot["entries"]
    _assert_same_git(mine["snapshot"]["git"], upstream.snapshot["git"])
    for view in ("eval_home", "snapshot"):
        assert getattr(upstream, view)["oldest_mtime"] > RECENT
        assert mine[view]["oldest_mtime"] > RECENT

    # clean_hf_models emptied the stand-in model folder but not final_model.
    entries = mine["snapshot"]["entries"]
    assert "scratch_model/CLEANED_LARGE_CONTENT.txt" in entries
    assert "scratch_model/model.safetensors" not in entries
    assert "final_model/sub/model.safetensors" in entries
    assert entries["kernel_link"][0] == "link"

    # Same evaluate.py arguments, with upstream's container paths.
    assert upstream.eval_argv is not None
    theirs = upstream.eval_argv
    assert mine["eval_argv"][0] == "python3"
    ours = mine["eval_argv"][1:]
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        if a.startswith("/mnt/eval/"):
            assert b.endswith(a.removeprefix("/mnt/eval")), (a, b)
        else:
            assert a == b
    assert mine["eval_user"] == AGENT_USER
    assert mine["eval_cwd"] == EVAL_HOME == "/home/user"  # upstream's --pwd (D-093)


def test_cases_cover_every_side_task() -> None:
    assert {side_task for side_task, _ in CASES} == {
        "none",
        "independent",
        "embedded",
        "embedded2",
        "embedded3",
    }
