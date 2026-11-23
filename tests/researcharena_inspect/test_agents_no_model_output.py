"""D-108: a model agent that got no response from any model call ends with an error, unscored.

The red-team solver runs through `eval()` with `mockllm/model` and a recording sandbox (no
Docker, no paid model): ReAct, and a stand-in for Claude Code that calls the task's model as
Inspect's bridge does, retries a failed call once as the CLI retries some API errors, and
raises as Inspect SWE does when the CLI exits non-zero. Each case reads the log back from
disk. `test_agents_docker.py` runs the same cases with the real CLI.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from inspect_ai import Task, eval
from inspect_ai.agent import Agent, AgentState
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.model import Model, ModelOutput, get_model
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from researcharena_inspect.agents import AgentRun, red_team_agent
from researcharena_inspect.agents.claude_code import InstalledCli
from researcharena_inspect.agents.solver import MODEL_AGENTS
from researcharena_inspect.environment.layout import PRE_AGENT_SETUP
from researcharena_inspect.task import red_team_sample
from tests.researcharena_inspect.fake_sandbox import Call, FakeSandbox, failed, ok

BUDGET_HOURS = "0.007"  # 25 s; every round here ends at once

NOT_FOUND = "Error code: 404 - {'error': {'message': 'model not found'}}"


class ModelNotFoundError(Exception):
    """What OpenRouter's 404 looked like in a live run; the bridge forwards its status to the CLI."""

    status_code = 404

    def __init__(self) -> None:
        super().__init__(NOT_FOUND)


Step = ModelOutput | Exception


def answer(text: str = "Done.") -> ModelOutput:
    return ModelOutput.from_content("mockllm/model", text)


def refusal() -> ModelOutput:
    return ModelOutput.from_content(
        "mockllm/model", "I can't help with that.", stop_reason="content_filter"
    )


def scripted_model(steps: list[Step] | None) -> Model:
    """Answers or raises call by call; `None`: every call raises `ModelNotFoundError`."""
    calls: Iterator[Step] = iter(steps or [])

    def outputs(*args: Any) -> ModelOutput:
        step = next(calls, None) if steps is not None else ModelNotFoundError()
        if step is None:
            raise AssertionError("the agent made more model calls than scripted")
        if isinstance(step, Exception):
            raise step
        return step

    return get_model("mockllm/model", custom_outputs=outputs, memoize=False)


def stand_in_claude_code(**kwargs: Any) -> Agent:
    """Claude Code as the solver sees it: model calls, then a return or Inspect SWE's error."""

    async def run(state: AgentState) -> AgentState:
        model = get_model()  # the task's model, as the bridge serves it
        for _ in range(2):  # one retry of a failed API call
            try:
                output = await model.generate(state.messages)
            except ModelNotFoundError:
                continue
            # A refusal returns too: Inspect SWE does not raise for the CLI's exit after it.
            state.messages.append(output.message)
            state.output = output
            return state
        raise RuntimeError("Error executing claude code agent 1: API Error: 404 model not found")

    return cast(Agent, run)


@scorer(metrics=[mean()])
def scored():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1)

    return score


def _setup_ok(call: Call) -> Any:
    if PRE_AGENT_SETUP in call.script:
        return ok("Pre-agent setup complete.\n__researcharena_agent_path__=/usr/bin:/bin\n")
    return None


@pytest.fixture
def box(monkeypatch: pytest.MonkeyPatch) -> FakeSandbox:
    fake = FakeSandbox(respond=_setup_ok)
    for module in ("solver", "processes", "replay"):
        monkeypatch.setattr(f"researcharena_inspect.agents.{module}.sandbox", lambda *a: fake)

    async def install(version: str, user: str | None = None) -> InstalledCli:
        return InstalledCli("/usr/local/bin/claude", "2.1.162 (Claude Code)", version)

    monkeypatch.setattr("researcharena_inspect.agents.solver.install_claude_code", install)
    monkeypatch.setattr(
        "researcharena_inspect.agents.solver.red_team_claude_code", stand_in_claude_code
    )
    return fake


def run_sample(
    tmp_path: Path,
    agent: str,
    model: Model,
    fail_on_no_model_output: bool = True,
    resume: bool = True,
    replay: str | None = None,
) -> EvalSample:
    sample = red_team_sample("cuda_kernel", "none", BUDGET_HOURS)
    assert sample.metadata is not None
    sample.metadata["resume_agent"] = resume
    if replay is not None:
        sample.metadata["replay"] = replay
    task = Task(
        dataset=[sample],
        solver=red_team_agent(agent=agent, fail_on_no_model_output=fail_on_no_model_output),
        scorer=scored(),
    )
    log_dir = tmp_path / f"{agent}-{fail_on_no_model_output}"
    [log] = eval(task, model=model, display="none", log_dir=str(log_dir))
    # Read back from disk: what a later reader of the log gets.
    on_disk: EvalLog = read_eval_log(log.location)
    assert on_disk.samples is not None
    [logged] = on_disk.samples
    return logged


def _model_events(sample: EvalSample) -> list[ModelEvent]:
    return [e for e in sample.events if isinstance(e, ModelEvent)]


def _assert_scored(sample: EvalSample) -> None:
    assert sample.error is None, sample.error
    assert sample.scores is not None
    assert sample.scores["scored"].value == 1


@pytest.mark.parametrize("agent", MODEL_AGENTS)
def test_every_model_call_failing_ends_the_sample_with_an_error(
    box: FakeSandbox, tmp_path: Path, agent: str
) -> None:
    sample = run_sample(tmp_path, agent, scripted_model(None))

    assert sample.error is not None
    message = sample.error.message
    assert message.startswith("NoModelOutputError(")
    assert f"No model output: the {agent} agent (model)" in message
    assert "(D-108)" in message
    assert 'its last model call failed: ModelNotFoundError("Error code: 404' in message
    assert "fail_on_no_model_output=False" in message
    assert not sample.scores
    # Every model call was made and failed: round 0 failed, the resume round failed and the
    # loop stopped (solve.sh:128-130).
    events = _model_events(sample)
    assert events
    assert all(e.error is not None and "model not found" in e.error for e in events)
    # The run's record and metadata are still in the log, to read the cause from.
    run = sample.store_as(AgentRun)
    assert run.agent == agent
    assert [(r["phase"], r["rc"]) for r in run.rounds] == [("exec", 1), ("resume", 1)]
    assert run.exit_code == 1
    assert run.red_start is not None
    assert run.red_end is not None
    assert run.log[-1] == "[agent] resume failed rc=1; stopping resume loop"
    assert any("exec error" in line for line in run.log)
    assert sample.metadata["agent"] == agent
    assert sample.metadata["agent_exit_code"] == 1
    assert sample.metadata["agent_rounds"] == 2
    assert sample.metadata["agent_time_taken_s"] is not None
    if agent == "claude_code":
        assert run.cli_version == "2.1.162 (Claude Code)"


@pytest.mark.parametrize("agent", MODEL_AGENTS)
def test_without_the_check_the_workspace_is_scored_as_upstream(
    box: FakeSandbox, tmp_path: Path, agent: str
) -> None:
    sample = run_sample(tmp_path, agent, scripted_model(None), fail_on_no_model_output=False)
    _assert_scored(sample)
    assert sample.metadata["agent_exit_code"] == 1
    assert sample.metadata["agent_rounds"] == 2


# Round 0's first call fails; a later call answers; the calls after it fail again.
LATER_ANSWER: dict[str, list[Step]] = {
    # The stand-in CLI retries within a round: round 0 answers on its retry; round 1 fails.
    "claude_code": [ModelNotFoundError(), answer(), ModelNotFoundError(), ModelNotFoundError()],
    # ReAct: round 0 fails, round 1 answers, round 2 fails.
    "react": [ModelNotFoundError(), answer(), ModelNotFoundError()],
}


@pytest.mark.parametrize("agent", MODEL_AGENTS)
def test_a_response_after_failed_calls_is_scored(
    box: FakeSandbox, tmp_path: Path, agent: str
) -> None:
    sample = run_sample(tmp_path, agent, scripted_model(LATER_ANSWER[agent]))
    _assert_scored(sample)
    events = _model_events(sample)
    assert events[0].error is not None  # the first call failed
    assert events[-1].error is not None  # and so did the last
    assert sum(e.error is None for e in events) == 1


@pytest.mark.parametrize("agent", MODEL_AGENTS)
def test_a_refusal_is_model_output(box: FakeSandbox, tmp_path: Path, agent: str) -> None:
    # The only response is a refusal, in the first call; every later call fails.
    steps: list[Step] = [refusal(), ModelNotFoundError(), ModelNotFoundError()]
    sample = run_sample(tmp_path, agent, scripted_model(steps))
    _assert_scored(sample)
    [returned] = [e for e in _model_events(sample) if e.error is None]
    assert returned.output.stop_reason == "content_filter"


@pytest.mark.parametrize("agent", MODEL_AGENTS)
def test_a_run_with_output_is_the_same_with_and_without_the_check(
    box: FakeSandbox, tmp_path: Path, agent: str
) -> None:
    runs = []
    for check in (True, False):
        sample = run_sample(
            tmp_path, agent, scripted_model([answer()]), fail_on_no_model_output=check, resume=False
        )
        _assert_scored(sample)
        run = sample.store_as(AgentRun)
        runs.append(
            (
                sample.scores,
                [(r["phase"], r["rc"]) for r in run.rounds],
                run.exit_code,
                [m.text for m in sample.messages],
                {k: v for k, v in sample.metadata.items() if k != "agent_time_taken_s"},
            )
        )
    assert runs[0] == runs[1]


def test_no_agent_start_is_no_model_output(box: FakeSandbox, tmp_path: Path) -> None:
    # pre_agent_setup.sh failed, so the agent never started (pipeline_redteam.sh:378).
    box.respond = lambda call: failed(1) if PRE_AGENT_SETUP in call.script else None
    sample = run_sample(tmp_path, "react", scripted_model([]))
    assert sample.error is not None
    assert "it made no model call" in sample.error.message
    assert "pre_agent_setup.sh exited 1; the agent did not start" in sample.error.message
    assert sample.store_as(AgentRun).pre_agent_setup_rc == 1


def _replay_run(tmp_path: Path) -> str:
    run_dir = tmp_path / "run"
    snapshot = run_dir / "red_team" / "task_snapshot"
    (snapshot / "kernels").mkdir(parents=True)
    (snapshot / "kernels" / "rmsnorm.cu").write_text("// agent\n")
    (run_dir / "metadata.json").write_text(
        '{"setting": "cuda_kernel", "side_task": "none", "task_config": "tinyllama",'
        ' "redruntime_hours": 1}'
    )
    return str(run_dir)


AGENTS_WITHOUT_A_MODEL: dict[str, Callable[[Path], str | None]] = {
    "null": lambda tmp_path: None,
    "replay": _replay_run,
}


@pytest.mark.parametrize("agent", list(AGENTS_WITHOUT_A_MODEL))
def test_agents_without_a_model_are_not_checked(
    box: FakeSandbox, tmp_path: Path, agent: str
) -> None:
    replay = AGENTS_WITHOUT_A_MODEL[agent](tmp_path)
    # A model that fails any call: these agents make none.
    sample = run_sample(tmp_path, agent, scripted_model([]), replay=replay)
    _assert_scored(sample)
    assert _model_events(sample) == []
    assert sample.metadata["agent_exit_code"] == 0
