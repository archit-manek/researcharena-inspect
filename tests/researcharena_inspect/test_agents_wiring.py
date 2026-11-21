"""How each agent is started in the sandbox, checked by shape with a recording sandbox.

Upstream starts the agent with `bash -c "source /home/user/pre_agent_setup.sh && bash
/home/user/agent_solve.sh"` in a container whose `--env` values are `environment.agent_env`'s,
as the cluster account, in `/home/user/task` (`pipeline_redteam.sh:357-378`). These tests check
the command, user, environment and working directory the sandbox gets for Claude Code, ReAct and
the null agent, without Docker or a model.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import AgentState, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ModelName, ModelOutput, get_model
from inspect_ai.solver import Generate, TaskState

from researcharena_inspect.agents import AgentRun, red_team_agent
from researcharena_inspect.agents.claude_code import InstalledCli, claude_code_env
from researcharena_inspect.agents.react import (
    REACT_BASH_TIMEOUT_S,
    agent_bash,
    react_bash_description,
    red_team_react,
)
from researcharena_inspect.environment.layout import (
    AGENT_PATH,
    AGENT_USER,
    BASH_TIMEOUT_MS,
    PRE_AGENT_SETUP,
    TASK_DIR,
    agent_env,
)
from researcharena_inspect.task import red_team_sample
from tests.researcharena_inspect.fake_sandbox import Call, FakeSandbox, failed, ok

SETUP_PATH = f"{TASK_DIR}:{AGENT_PATH}"
"""`PATH` after `pre_agent_setup.sh` prepends the workspace once more (its line 7)."""

SETUP_OUTPUT = "CUDA OK: 1 device(s) available\n  GPU 0: NVIDIA L4\nPre-agent setup complete.\n"


def _respond_setup_ok(call: Call) -> Any:
    if PRE_AGENT_SETUP in call.script:
        return ok(f"{SETUP_OUTPUT}\n__researcharena_agent_path__={SETUP_PATH}\n")
    return None


def _state(side_task: str = "none") -> TaskState:
    sample = red_team_sample("cuda_kernel", side_task, "1")
    assert sample.metadata is not None
    metadata = dict(sample.metadata)
    metadata["resume_agent"] = False  # one round, so the dummy agents do not loop
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=sample.id or "s",
        epoch=1,
        input=sample.input,
        messages=[],
        metadata=metadata,
    )


async def _no_generate(state: TaskState, **kwargs: Any) -> TaskState:
    return state


_generate = cast(Generate, _no_generate)


async def _dummy_agent(state: AgentState) -> AgentState:
    """Calls no model, so the model agents run with `fail_on_no_model_output=False` (D-108,
    tested in `test_agents_no_model_output.py`)."""
    return state


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> FakeSandbox:
    fake = FakeSandbox(respond=_respond_setup_ok)
    for module in ("solver", "processes", "replay"):
        monkeypatch.setattr(f"researcharena_inspect.agents.{module}.sandbox", lambda *a: fake)
    monkeypatch.setattr(
        "researcharena_inspect.agents.solver.get_model", lambda: get_model("mockllm/model")
    )
    return fake


def _expected_env() -> dict[str, str]:
    return agent_env(redruntime="1", resume_agent=False, agent_config="model", agent_effort="xhigh")


def _assert_setup_sourced_as_upstream(box: FakeSandbox) -> None:
    [setup] = box.find(lambda c: PRE_AGENT_SETUP in c.script)
    assert setup.cmd[:2] == ["bash", "-c"]
    assert setup.script.startswith(f"source {PRE_AGENT_SETUP} && ")
    assert setup.user == AGENT_USER == "user"
    assert setup.cwd == TASK_DIR == "/home/user/task"
    assert setup.env == _expected_env()


async def test_claude_code_runs_as_the_agent_user_with_upstreams_environment(
    box: FakeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    async def install(version: str, user: str | None = None) -> InstalledCli:
        seen["install_user"] = user
        return InstalledCli("/usr/local/bin/claude", "2.1.162 (Claude Code)", version)

    def claude(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _dummy_agent

    monkeypatch.setattr("researcharena_inspect.agents.solver.install_claude_code", install)
    monkeypatch.setattr("researcharena_inspect.agents.solver.red_team_claude_code", claude)
    solve = red_team_agent(agent="claude_code", fail_on_no_model_output=False)
    state = await solve(_state(), _generate)

    _assert_setup_sourced_as_upstream(box)
    assert seen["install_user"] == AGENT_USER
    assert seen["user"] == AGENT_USER
    # Claude Code's background calls go to Haiku 4.5, as in the published runs (D-080).
    assert seen["small_model"] == "anthropic/claude-haiku-4-5-20251001"
    expected = {**_expected_env(), "PATH": SETUP_PATH} | claude_code_env(
        "model", "xhigh", "1", False
    )
    assert seen["env"] == expected
    # solve.sh:26 raises the Bash limit over the container's value.
    assert seen["env"]["BASH_MAX_TIMEOUT_MS"] == "36000000"
    assert seen["env"]["HOME"] == "/home/user"
    run = state.store_as(AgentRun)
    assert run.pre_agent_setup_rc == 0
    assert run.pre_agent_setup_output.startswith("CUDA OK")
    assert "__researcharena" not in run.pre_agent_setup_output


async def test_react_runs_as_the_agent_user_in_the_workspace(
    box: FakeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def react(model: Any, effort: str, **kwargs: Any) -> Any:
        seen.update(kwargs, effort=effort)
        return _dummy_agent

    monkeypatch.setattr("researcharena_inspect.agents.solver.red_team_react", react)
    await red_team_agent(agent="react", fail_on_no_model_output=False)(_state(), _generate)

    _assert_setup_sourced_as_upstream(box)
    assert seen["user"] == AGENT_USER
    assert seen["cwd"] == TASK_DIR
    assert seen["env"] == {**_expected_env(), "PATH": SETUP_PATH}


async def test_null_agent_does_nothing_as_the_agent_user(
    box: FakeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model_agent(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the null agent must not build a model agent")

    monkeypatch.setattr("researcharena_inspect.agents.solver.red_team_react", no_model_agent)
    monkeypatch.setattr("researcharena_inspect.agents.solver.red_team_claude_code", no_model_agent)
    state = await red_team_agent(agent="null")(_state(), _generate)

    _assert_setup_sourced_as_upstream(box)
    [true] = box.find(lambda c: c.cmd == ["true"])
    assert (true.user, true.cwd) == (AGENT_USER, TASK_DIR)
    assert true.env == {**_expected_env(), "PATH": SETUP_PATH}
    run = state.store_as(AgentRun)
    assert run.agent == "null"
    assert run.exit_code == 0
    assert run.rounds == []
    assert state.metadata["agent_exit_code"] == 0
    # Only the setup, the agent's `true` and the process list and sweep (as root) ran.
    assert {c.user for c in box.calls} == {AGENT_USER, "root"}


async def test_no_agent_when_pre_agent_setup_fails(
    box: FakeSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    box.respond = lambda call: failed(1) if PRE_AGENT_SETUP in call.script else None
    state = await red_team_agent(agent="null")(_state(), _generate)
    run = state.store_as(AgentRun)
    assert not box.find(lambda c: c.cmd == ["true"])
    assert run.pre_agent_setup_rc == 1
    assert run.exit_code is None
    assert run.log == ["[agent] pre_agent_setup.sh exited 1; the agent did not start"]


async def test_replay_agent_puts_the_workspace_in_place(box: FakeSandbox, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    snapshot = run_dir / "red_team" / "task_snapshot"
    (snapshot / "kernels").mkdir(parents=True)
    (snapshot / "workspace").mkdir()
    (snapshot / "kernels" / "rmsnorm.cu").write_text("// agent\n")
    (snapshot / "workspace" / "last_bench.json").write_text("{}")
    (run_dir / "metadata.json").write_text(
        '{"setting": "cuda_kernel", "side_task": "none", "task_config": "tinyllama",'
        ' "redruntime_hours": 1}'
    )
    state = _state()
    state.metadata["replay"] = str(run_dir)
    state = await red_team_agent(agent="replay", replay_mode="fresh")(state, _generate)
    run = state.store_as(AgentRun)
    assert run.replay is not None
    assert run.replay["files"] == ["kernels/rmsnorm.cu"]
    assert run.replay["left_out"] == ["workspace/last_bench.json"]
    [unpack] = box.find(lambda c: "tar -xpmf" in c.script)
    assert unpack.user == AGENT_USER
    assert TASK_DIR in unpack.script
    assert run.exit_code == 0


async def test_react_bash_tool_gets_user_environment_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSandbox(respond=lambda call: ok("out"))
    monkeypatch.setattr("researcharena_inspect.agents.react.sandbox", lambda: fake)
    env = {"PATH": SETUP_PATH, "HOME": "/home/user"}
    tool = agent_bash(timeout=3600, user=AGENT_USER, env=env, cwd=TASK_DIR)
    assert await tool(command="nvidia-smi") == "out"
    [call] = fake.calls
    assert call.cmd == ["bash", "--login", "-c", "nvidia-smi"]
    assert (call.user, call.cwd, call.env, call.timeout) == (AGENT_USER, TASK_DIR, env, 3600)


def test_react_tools_name_the_workspace_and_the_time_limit(tmp_path: Path) -> None:
    # What the model is sent, read from the log; the model answers without a tool call.
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "Done.")]
    )
    agent = red_team_react(model, "default", user=AGENT_USER, env={}, cwd=TASK_DIR)
    task = Task(dataset=[Sample(input="the task")], solver=as_solver(agent))
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples is not None
    sample = resolve_sample_attachments(log.samples[0], "full")
    [event] = [e for e in sample.events if isinstance(e, ModelEvent)]
    assert [m.text for m in event.input] == ["the task"]  # no system message
    tools = {t.name: t for t in event.tools}
    bash = tools["bash"]
    assert bash.description == react_bash_description(TASK_DIR)
    assert f"starts in {TASK_DIR}, the working directory, so relative paths start" in (
        bash.description
    )
    assert f"absolute paths for workspace files in the file editor, such as {TASK_DIR}/" in (
        bash.description
    )
    assert "it resolves relative paths from /." in bash.description
    # The limit stated is the one `red_team_react` gives the tool: Claude Code's default in
    # upstream's agent (`BASH_DEFAULT_TIMEOUT_MS`).
    assert int(BASH_TIMEOUT_MS) == REACT_BASH_TIMEOUT_S * 1000
    assert f"stopped after at most {REACT_BASH_TIMEOUT_S} seconds (60 minutes)" in (
        bash.description
    )
    assert list(bash.parameters.properties) == ["command"]
    # The editor is Inspect's, unchanged: its own description and parameters.
    assert tools["text_editor"].description == "Use this function to execute text editing commands."
    assert sorted(tools["text_editor"].parameters.properties) == [
        "command",
        "file_text",
        "insert_line",
        "insert_text",
        "new_str",
        "old_str",
        "path",
        "view_range",
    ]


def test_react_agent_builds_with_the_agent_user() -> None:
    agent = red_team_react(
        get_model("mockllm/model"), "xhigh", user=AGENT_USER, env={}, cwd=TASK_DIR
    )
    assert callable(agent)


def test_react_bash_description_names_the_cwd_it_is_given() -> None:
    text = react_bash_description("/elsewhere")
    assert "starts in /elsewhere, the working directory" in text
    assert "such as /elsewhere/file.py" in text
    assert TASK_DIR not in text


def test_claude_code_gets_the_user_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect_swe

    from researcharena_inspect.agents.claude_code import red_team_claude_code

    seen: dict[str, Any] = {}
    monkeypatch.setattr(inspect_swe, "claude_code", lambda **kwargs: seen.update(kwargs))
    red_team_claude_code(effort="xhigh", env={"PATH": SETUP_PATH}, user=AGENT_USER)
    assert seen["user"] == AGENT_USER
    assert seen["cwd"] == TASK_DIR
    assert seen["env"] == {"PATH": SETUP_PATH}
    assert seen["version"] == "sandbox"
