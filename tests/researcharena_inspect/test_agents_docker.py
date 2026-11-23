"""Both red-team agents end to end in Docker with `mockllm/model`: budget, re-prompting, output.

The scripted agent works in three rounds. In rounds 0 and 1 it writes a file with one tool call
and then stops (no tool call), as an agent that thinks it is done would; upstream re-prompts it
each time. In round 2 it starts a background `sleep 300` and a foreground one, so it is still
working when the budget runs out. The budget is 25 seconds (`redruntime` 0.007 hours).
A second Claude Code run has the model refuse in rounds 0 and 2 (D-084). A third pair of runs
has every model call fail with a 404 and ends with an error (D-108). A fourth
Claude Code run has the model fail until the first resume round and answer from then on (D-105).

Claude Code is the real CLI (Inspect SWE, version `CLI_VERSION`) talking to the mock model
through Inspect's bridge. No paid model is called.
"""

import json
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, resolve_sample_attachments
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    ContentText,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from researcharena_inspect import researcharena
from researcharena_inspect.agents import RC_BUDGET, RC_OK, AgentRun, write_run_dirs
from researcharena_inspect.agents.budget import ROUND_KILL_AFTER_S
from researcharena_inspect.agents.react import react_bash_description
from researcharena_inspect.agents.solver import MODEL_AGENTS
from researcharena_inspect.environment.layout import TASK_DIR
from researcharena_inspect.monitor import local_bundles, researcharena_monitor
from researcharena_inspect.prompts import agent_prompt, resume_prompt
from researcharena_inspect.transcripts import parse_cli_output

CLI_VERSION = "2.1.162"
"""The Claude Code version in the published Sonnet 4.6 and Opus 4.6 CUDA runs."""

BUDGET_HOURS = "0.007"
BUDGET_S = 25

STOP_TOLERANCE_S = 10
"""How late after the deadline the last round may end: Claude Code's own exit on SIGTERM
(about 1-2 s measured here; `--kill-after` would allow 30 s) or, for ReAct, Inspect's
cancellation of the round (2-4 s)."""

CONTINUE = "Continue where you left off"

pytestmark = [pytest.mark.docker, pytest.mark.slow]


def _round(messages: list[ChatMessage]) -> int:
    return sum(isinstance(m, ChatMessageUser) and m.text.startswith(CONTINUE) for m in messages)


def _tools_since_user(messages: list[ChatMessage]) -> int:
    count = 0
    for m in reversed(messages):
        if isinstance(m, ChatMessageTool):
            count += 1
        elif m.role == "user":
            break
    return count


def scripted_agent(agent: str):
    shell = "Bash" if agent == "claude_code" else "bash"

    def script(messages, tools, tool_choice, config) -> ModelOutput:
        if shell not in {t.name for t in tools}:
            return ModelOutput.from_content("mockllm/model", "ok")  # Claude Code's side calls
        n = _round(messages)
        if _tools_since_user(messages) > 0:
            return ModelOutput.from_content("mockllm/model", f"Finished round {n}.")
        if n < 2:
            command = f"echo {n} > /home/user/task/round_{n}.txt"
        else:
            command = "(sleep 300 &) ; sleep 300"
        return ModelOutput.for_tool_call("mockllm/model", shell, {"command": command})

    return script


@scorer(metrics=[mean()])
def workspace_probe():
    """Stands in for the task scorer: what is in the workspace and what still runs."""

    async def score(state: TaskState, target: Target) -> Score:
        files = await sandbox().exec(["bash", "-c", "ls /home/user/task"])
        procs = await sandbox().exec(
            ["bash", "-c", "for d in /proc/[0-9]*; do tr '\\0' ' ' < $d/cmdline; echo; done"]
        )
        return Score(
            value=1,
            metadata={"files": files.stdout.split(), "processes": procs.stdout.splitlines()},
        )

    return score


# These scenarios need a model agent; AGENTS also has the no-model agents.
@pytest.fixture(scope="module", params=MODEL_AGENTS)
def red_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    agent = request.param
    # `sandbox="docker"` is the CPU test image (it has git, so the default staging runs); the
    # agent runs as the staged user.
    task = researcharena(
        side_tasks="independent",
        agent=agent,
        redruntime=BUDGET_HOURS,
        claude_code_version=CLI_VERSION,
        small_model=None,  # the mock model answers Claude Code's background calls too
        sandbox="docker",
        scorer=workspace_probe(),
    )
    model = get_model("mockllm/model", custom_outputs=scripted_agent(agent))
    log_dir = tmp_path_factory.mktemp(f"red_{agent}")
    [log] = eval(task, model=model, display="none", log_dir=str(log_dir))
    return agent, log


def _sample(log: EvalLog) -> tuple[EvalSample, AgentRun]:
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    return sample, sample.store_as(AgentRun)


def test_run_completes_and_the_work_is_scored(red_run) -> None:
    agent, log = red_run
    sample, run = _sample(log)
    assert run.agent == agent
    assert sample.scores is not None
    probe = sample.scores["workspace_probe"]
    assert probe.metadata is not None
    assert {"round_0.txt", "round_1.txt"} <= set(probe.metadata["files"])
    assert sample.metadata["agent_rounds"] == 3
    assert sample.metadata["agent_exit_code"] == RC_BUDGET


def test_agent_that_stops_early_is_reprompted_as_upstream(red_run) -> None:
    _, log = red_run
    _, run = _sample(log)
    assert [(r["phase"], r["number"], r["rc"]) for r in run.rounds] == [
        ("exec", 0, RC_OK),
        ("resume", 1, RC_OK),
        ("resume", 2, RC_BUDGET),
    ]
    for r in run.rounds[1:]:
        assert r["prompt"] == resume_prompt(r["remaining_at_start"])
    assert run.log[-1] == "[agent] resume used remaining budget"
    assert run.exit_code == RC_BUDGET


def test_agent_still_working_is_stopped_at_the_budget(red_run) -> None:
    agent, log = red_run
    sample, run = _sample(log)
    assert run.budget_seconds == BUDGET_S
    last = run.rounds[-1]
    # The round's limit was the time left, so it should end at the deadline (budget start +
    # 25 s), not earlier, and only as late as the CLI's exit or the cancellation takes.
    assert last["started_at"] + last["remaining_at_start"] == pytest.approx(BUDGET_S, abs=1)
    assert BUDGET_S - 1 <= last["ended_at"] <= BUDGET_S + STOP_TOLERANCE_S
    assert last["rc"] == RC_BUDGET
    # Nothing the agent started is left running when the scorer runs.
    assert sample.scores is not None
    processes = sample.scores["workspace_probe"].metadata["processes"]  # type: ignore[index]
    assert not [p for p in processes if "sleep 300" in p]
    assert not [p for p in processes if "claude" in p]
    if agent == "claude_code":
        # SIGTERM at the deadline, as `timeout` sends it (solve.sh:97). The CLI kills its Bash
        # tool's commands (both sleeps) itself and exits within the grace, so it is not
        # SIGKILLed and nothing is left for the end of the agent phase to kill.
        [stop] = run.deadline_stops
        assert [p for p in stop["terminated"] if "claude" in p and "--print" in p]
        # One entry per process, though the CLI's command line holds the prompt's newlines.
        assert all(p.split(" ", 1)[0].isdigit() for p in stop["terminated"])
        assert not stop["killed_after_grace"]
        assert stop["killed"] == []
        assert stop["ended_after_term_s"] < ROUND_KILL_AFTER_S
        assert stop["cli_exit_code"] in (0, 143)  # its own code; the round's is 124
    else:
        assert run.killed  # ReAct's round is cancelled; its tool's sleeps are killed after


def test_budget_counts_what_upstream_counts(red_run) -> None:
    agent, log = red_run
    _, run = _sample(log)
    assert run.red_start is not None
    assert run.budget_start is not None
    assert run.red_end is not None
    # The clock starts after the CLI install (solve.sh:38 then :59), so the install is not
    # counted; the first round starts as soon as the clock does.
    assert run.budget_start >= int(run.red_start)
    assert run.rounds[0]["started_at"] < 2
    if agent == "claude_code":
        assert run.cli_version == f"{CLI_VERSION} (Claude Code)"
    # Everything after that is counted: model calls, tools and the restarts between rounds.
    # Each round's limit is the deadline minus the time used so far.
    for r in run.rounds:
        assert r["remaining_at_start"] == pytest.approx(BUDGET_S - r["started_at"], abs=1)
    # The whole phase stays within RED_TIMEOUT (budget + 5 minutes).
    assert run.red_end - run.red_start < run.red_timeout_seconds
    assert not run.outer_timeout


def _model_inputs(sample: EvalSample) -> list[list[ChatMessage]]:
    return [e.input for e in _main_model_events(sample)]


def _user_texts(message: ChatMessage) -> list[str]:
    if isinstance(message.content, str):
        return [message.content]
    return [c.text for c in message.content if isinstance(c, ContentText)]


def test_prompts_reach_the_model_byte_identical(red_run) -> None:
    _, log = red_run
    sample, run = _sample(log)
    expected = agent_prompt("cuda_kernel", "independent", "claude", BUDGET_HOURS)
    inputs = _model_inputs(resolve_sample_attachments(sample, "full"))
    first_user = next(m for m in inputs[0] if isinstance(m, ChatMessageUser))
    # ReAct sends the prompt as the message; Claude Code puts a <system-reminder> block with
    # the date before it in the same message, as upstream's CLI does.
    assert expected in _user_texts(first_user)
    for r in run.rounds[1:]:
        assert any(
            r["prompt"] in _user_texts(m)
            for msgs in inputs
            for m in msgs
            if isinstance(m, ChatMessageUser)
        ), r["number"]


def _main_model_events(sample: EvalSample) -> list[ModelEvent]:
    return [
        e
        for e in sample.events
        if isinstance(e, ModelEvent) and {"Bash", "bash"} & {t.name for t in e.tools}
    ]


def test_effort_and_memory_reach_the_model(red_run) -> None:
    agent, log = red_run
    sample, run = _sample(log)
    assert run.effort == "xhigh"  # effort.json, red_team "claude:*"
    events = _main_model_events(resolve_sample_attachments(sample, "full"))
    assert events
    # The effort is on the served model's config for every call (Inspect's Anthropic provider
    # turns it into adaptive thinking with output_config.effort and summarized display).
    assert {e.config.reasoning_effort for e in events} == {"xhigh"}
    if agent == "claude_code":
        system = "\n".join(m.text for m in events[0].input if m.role == "system")
        # Auto-memory is on, as it was for upstream's agents.
        assert (
            "You have a persistent file-based memory at "
            "`/home/user/.claude/projects/-home-user-task/memory/`"
        ) in system


def test_run_folder_for_the_monitor(red_run, tmp_path: Path) -> None:
    agent, log = red_run
    _, run = _sample(log)
    [run_dir] = write_run_dirs(log, tmp_path)
    red = run_dir / "red_team"
    with_cot = (red / "solve_parsed.txt").read_text("utf-8")
    without_cot = (red / "solve_parsed_no_cot.txt").read_text("utf-8")
    assert "round_1.txt" in without_cot
    if agent == "claude_code":
        assert run.stdout[0].startswith('{"type":"system","subtype":"init"')
        assert '"cwd":"/home/user/task"' in run.stdout[0]
        # One result event per round. The third got SIGTERM at the deadline; the CLI killed its
        # tool and, since the mock model then answered without a tool call, finished the turn
        # and wrote its result before exiting, as upstream's CLI does under `timeout`.
        assert sum('"type":"result"' in line for line in run.stdout) == 3
        assert parse_cli_output(red / "solve_out.json", "claude").with_cot == with_cot
        assert "Tool call — Bash" in without_cot
        assert "Session start — " in with_cot
    else:
        assert "Tool call — bash" in without_cot
    [bundle] = local_bundles(tmp_path)
    assert bundle.run_id == run_dir.name
    assert bundle.red_provider == ("claude" if agent == "claude_code" else "react")


def scripted_monitor(messages, tools, tool_choice, config) -> ModelOutput:
    if not any(isinstance(m, ChatMessageTool) for m in messages):
        judgement = '{"suspicion_score": 12, "reasoning": "scripted"}'
        return ModelOutput.for_tool_call(
            "mockllm/model",
            "bash",
            {
                "command": f"head -c 300 agent_transcript.txt; printf '%s' '{judgement}' > judgement.json"
            },
        )
    return ModelOutput.from_content("mockllm/model", "Done.")


def test_monitor_task_runs_on_our_run(red_run, tmp_path: Path) -> None:
    _, log = red_run
    write_run_dirs(log, tmp_path / "runs")
    task = researcharena_monitor(
        access="transcript", local_runs=str(tmp_path / "runs"), monitor_agent="react"
    )
    model = get_model("mockllm/model", custom_outputs=scripted_monitor)
    [monitor_log] = eval(task, model=model, display="none", log_dir=str(tmp_path / "logs"))
    assert monitor_log.status == "success", monitor_log.error
    assert monitor_log.samples is not None
    [sample] = monitor_log.samples
    assert sample.scores is not None
    assert sample.scores["monitor_suspicion"].value == 12
    shown = next(m for m in sample.messages if isinstance(m, ChatMessageTool)).text
    assert shown.startswith(("Session start", "Assistant — turn 1"))


def test_no_resume_when_the_setting_says_so(tmp_path: Path) -> None:
    # The post-training settings have resume_agent false: one run, however early it ends.
    task = researcharena(
        setting="safety_finetuning",
        side_tasks="none",
        agent="react",
        redruntime=BUDGET_HOURS,
        sandbox="docker",
        scorer=workspace_probe(),
        model_cache="none",  # the CPU image cannot download the setting's models
    )
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "Done.")]
    )
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    _, run = _sample(log)
    assert [r["phase"] for r in run.rounds] == ["exec"]
    assert run.rounds[0]["ended_at"] < 5
    assert "[agent] AGENT_RESUME=false: not resuming" in run.log


def test_react_tool_descriptions_hold_in_the_sandbox(tmp_path: Path) -> None:
    # What `bash`'s description says, checked in the sandbox: relative paths start in the
    # workspace, the editor resolves relative paths from `/`, and `command` is the only argument.
    calls = [
        ("bash", {"command": "pwd && ls evaluate_model.py"}),
        ("text_editor", {"command": "view", "path": "evaluate_model.py"}),
        ("text_editor", {"command": "view", "path": f"{TASK_DIR}/evaluate_model.py"}),
        ("text_editor", {"command": "view", "path": "etc/hostname"}),
        ("text_editor", {"command": "view", "path": "/etc/hostname"}),
        ("bash", {"command": "true", "timeout": 20}),
    ]
    task = researcharena(
        setting="safety_finetuning",  # one round: the post-training settings do not resume
        side_tasks="none",
        agent="react",
        redruntime=BUDGET_HOURS,
        sandbox="docker",
        scorer=workspace_probe(),
        model_cache="none",
    )
    outputs = [ModelOutput.for_tool_call("mockllm/model", name, args) for name, args in calls]
    model = get_model(
        "mockllm/model",
        custom_outputs=[*outputs, ModelOutput.from_content("mockllm/model", "Done.")],
    )
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    sample, _ = _sample(log)
    sample = resolve_sample_attachments(sample, "full")
    first = _main_model_events(sample)[0]
    tools = {t.name: t for t in first.tools}
    assert tools["bash"].description == react_bash_description(TASK_DIR)
    assert list(tools["bash"].parameters.properties) == ["command"]
    results = [m for m in sample.messages if isinstance(m, ChatMessageTool)]
    assert len(results) == len(calls)
    assert results[0].error is None
    assert results[0].text.split() == [TASK_DIR, "evaluate_model.py"]
    assert results[1].error is not None  # resolved from `/`, not from the workspace
    assert "/evaluate_model.py does not exist" in results[1].error.message
    assert results[2].error is None
    # A relative path that exists under `/` is read from there: the same file as its
    # absolute form.
    assert results[3].error is None
    assert results[4].error is None
    assert results[3].text.split("\n", 1)[1] == results[4].text.split("\n", 1)[1]
    assert "/etc/hostname" in results[3].text.split("\n", 1)[0]
    assert "'timeout' was unexpected" in results[5].error.message  # type: ignore[union-attr]


def test_red_timeout_ends_the_phase_and_the_sample_is_still_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RED_TIMEOUT is the budget plus 5 minutes; shortened here to 6 s so it fires first.
    monkeypatch.setattr("researcharena_inspect.agents.solver.red_timeout_seconds", lambda _: 6)
    task = researcharena(
        side_tasks="none",
        agent="react",
        redruntime=BUDGET_HOURS,
        sandbox="docker",
        scorer=workspace_probe(),
    )
    model = get_model(
        "mockllm/model",
        custom_outputs=lambda *_: ModelOutput.for_tool_call(
            "mockllm/model", "bash", {"command": "sleep 300"}
        ),
    )
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    sample, run = _sample(log)
    assert run.outer_timeout
    assert run.rounds == []  # round 0 was still running
    assert run.red_end is not None
    assert run.red_start is not None
    assert run.red_end - run.red_start < 6 + STOP_TOLERANCE_S
    assert sample.scores is not None
    processes = sample.scores["workspace_probe"].metadata["processes"]  # type: ignore[index]
    assert not [p for p in processes if "sleep 300" in p]


# --- Refusals (D-084) -------------------------------------------------------------------------

REFUSED_ROUNDS = (0, 2)


def refusing_agent(messages, tools, tool_choice, config) -> ModelOutput:
    """Refuses in rounds 0 and 2 (stop reason `content_filter`, which the bridge sends to the
    CLI as Anthropic's `refusal`) and ends round 1 normally."""
    if "Bash" not in {t.name for t in tools}:
        return ModelOutput.from_content("mockllm/model", "ok")  # Claude Code's side calls
    n = _round(messages)
    if n in REFUSED_ROUNDS:
        return ModelOutput.from_content(
            "mockllm/model", "I can't help with that.", stop_reason="content_filter"
        )
    return ModelOutput.from_content("mockllm/model", f"Finished round {n}.")


@pytest.fixture(scope="module")
def refusal_run(tmp_path_factory: pytest.TempPathFactory) -> EvalLog:
    task = researcharena(
        side_tasks="none",
        agent="claude_code",
        redruntime="0.02",  # 72 s: enough for three rounds; the refusal ends the loop first
        claude_code_version=CLI_VERSION,
        small_model=None,  # the mock model answers Claude Code's background calls too
        sandbox="docker",
        scorer=workspace_probe(),
    )
    model = get_model("mockllm/model", custom_outputs=refusing_agent)
    log_dir = tmp_path_factory.mktemp("red_refusal")
    [log] = eval(task, model=model, display="none", log_dir=str(log_dir))
    return log


def test_cli_exits_1_after_a_refusal(refusal_run: EvalLog) -> None:
    _, run = _sample(refusal_run)
    # The CLI's own codes: 1 after each refusal, 0 for the normal round.
    assert run.cli_exit_codes == [1, 0, 1]
    results = [line for line in run.stdout if '"type":"result"' in line]
    assert len(results) == 3
    for n in REFUSED_ROUNDS:
        assert '"is_error":true' in results[n]
        assert '"stop_reason":"refusal"' in results[n]
        assert "Usage Policy" in results[n]
    assert '"is_error":false' in results[1]


def test_refused_first_round_is_still_resumed(refusal_run: EvalLog) -> None:
    _, run = _sample(refusal_run)
    assert run.rounds[0]["phase"] == "exec"
    assert run.rounds[0]["rc"] == 1
    assert run.rounds[1]["phase"] == "resume"  # solve.sh:112 resumes any code but 124
    exec_line = next(line for line in run.log if line.startswith("[agent] exec finished"))
    assert exec_line.startswith("[agent] exec finished rc=1 remaining_seconds=")
    assert "[agent] phase 2/2 round 1: claude --continue" in run.log


def test_refused_resume_round_stops_the_loop(refusal_run: EvalLog) -> None:
    sample, run = _sample(refusal_run)
    assert [(r["phase"], r["number"], r["rc"]) for r in run.rounds] == [
        ("exec", 0, 1),
        ("resume", 1, RC_OK),
        ("resume", 2, 1),
    ]
    # Stopped by solve.sh:128-130 with time left, not by the budget.
    assert run.log[-1] == "[agent] resume failed rc=1; stopping resume loop"
    assert run.rounds[-1]["ended_at"] < 60
    assert run.exit_code == 1
    assert sample.metadata["agent_exit_code"] == 1
    assert run.deadline_stops == []
    assert sample.scores is not None  # the workspace is still scored


# --- No model output (D-108) ------------------------------------------------------------------


class ModelNotFoundError(Exception):
    """As OpenRouter's 404 in a live run (17 Nov 2026); the bridge passes the status to the CLI."""

    status_code = 404

    def __init__(self) -> None:
        super().__init__("Error code: 404 - {'error': {'message': 'model not found'}}")


def model_not_found(messages, tools, tool_choice, config) -> ModelOutput:
    raise ModelNotFoundError()


def _no_output_task(agent: str, fail_on_no_model_output: bool = True):
    return researcharena(
        side_tasks="none",
        agent=agent,
        redruntime=BUDGET_HOURS,
        claude_code_version=CLI_VERSION,
        small_model=None,  # Claude Code's background calls fail too
        sandbox="docker",
        scorer=workspace_probe(),
        fail_on_no_model_output=fail_on_no_model_output,
    )


@pytest.fixture(scope="module", params=MODEL_AGENTS)
def no_output_run(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, EvalSample]:
    agent = request.param
    model = get_model("mockllm/model", custom_outputs=model_not_found)
    log_dir = tmp_path_factory.mktemp(f"no_output_{agent}")
    [log] = eval(_no_output_task(agent), model=model, display="none", log_dir=str(log_dir))
    logged = read_eval_log(log.location)  # as a later reader of the log sees it
    assert logged.samples is not None
    [sample] = logged.samples
    return agent, sample


def test_no_model_output_ends_the_sample_with_an_error(
    no_output_run: tuple[str, EvalSample],
) -> None:
    agent, sample = no_output_run
    assert sample.error is not None
    assert f"No model output: the {agent} agent" in sample.error.message
    assert "model not found" in sample.error.message
    assert not sample.scores
    events = [e for e in sample.events if isinstance(e, ModelEvent)]
    assert events
    assert all(e.error is not None for e in events)


def test_errored_sample_keeps_the_run_record(no_output_run: tuple[str, EvalSample]) -> None:
    agent, sample = no_output_run
    run = sample.store_as(AgentRun)
    assert run.agent == agent
    # Round 0 failed, round 1 was a resume and failed too, so the loop stopped with time left
    # (solve.sh:128-130).
    assert [(r["phase"], r["rc"]) for r in run.rounds] == [("exec", 1), ("resume", 1)]
    assert run.exit_code == 1
    assert run.log[-1] == "[agent] resume failed rc=1; stopping resume loop"
    assert sample.metadata["agent_exit_code"] == 1
    assert sample.metadata["agent_rounds"] == 2
    assert sample.metadata["agent_time_taken_s"] is not None
    assert sample.metadata["agent_time_taken_s"] < BUDGET_S
    if agent == "claude_code":
        assert run.cli_version == f"{CLI_VERSION} (Claude Code)"
        assert run.cli_exit_codes == [1, 1]
        assert run.stdout[0].startswith('{"type":"system","subtype":"init"')
        # Round 1 had no assistant message to resume from, so it ran in a new session, which
        # failed on the same 404 (D-105); without it the CLI refused to start ("Session ID
        # ... is already in use").
        assert run.new_sessions == [1]
        assert "is already in use" not in "".join(run.stderr)


def test_without_the_check_the_workspace_is_scored(tmp_path: Path) -> None:
    model = get_model("mockllm/model", custom_outputs=model_not_found)
    task = _no_output_task("react", fail_on_no_model_output=False)
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    sample, run = _sample(log)
    assert sample.error is None
    assert sample.scores is not None
    assert run.exit_code == 1


# --- A first round with no model response, then a working model (D-105) ----------------------


def model_not_found_until_resumed(messages, tools, tool_choice, config) -> ModelOutput:
    """404 for every call of round 0 (as in a live run), answers once a continuation prompt is in."""
    if not any(CONTINUE in m.text for m in messages):
        raise ModelNotFoundError()
    return ModelOutput.from_content("mockllm/model", "Done.")


@pytest.fixture(scope="module")
def late_model_run(tmp_path_factory: pytest.TempPathFactory) -> EvalSample:
    task = _no_output_task("claude_code")
    model = get_model("mockllm/model", custom_outputs=model_not_found_until_resumed)
    log_dir = tmp_path_factory.mktemp("late_model")
    [log] = eval(task, model=model, display="none", log_dir=str(log_dir))
    logged = read_eval_log(log.location)
    assert logged.samples is not None
    [sample] = logged.samples
    return resolve_sample_attachments(sample)


def _init_sessions(run: AgentRun) -> list[str]:
    return [
        json.loads(line)["session_id"]
        for line in run.stdout
        if line.startswith('{"type":"system","subtype":"init"')
    ]


def test_run_carries_on_after_a_first_round_without_a_response(late_model_run) -> None:
    sample = late_model_run
    run = sample.store_as(AgentRun)
    assert sample.error is None
    assert sample.scores is not None
    # Round 0 failed (the CLI exits 1 after the 404s); round 1 resumed, as upstream's
    # `--continue` does after any round but a timed-out one, and the loop went on until the
    # budget ran out.
    phases = [(r["phase"], r["rc"]) for r in run.rounds]
    assert phases[0] == ("exec", 1)
    assert phases[1] == ("resume", RC_OK)
    assert phases[-1] == ("resume", RC_BUDGET)
    assert run.cli_exit_codes[:2] == [1, 0]
    assert "is already in use" not in "".join(run.stderr)
    # Round 1 started a new session; every later round resumed it.
    assert run.new_sessions == [1]
    sessions = _init_sessions(run)
    assert len(sessions) == len(run.rounds)
    assert sessions[0] != sessions[1]
    assert set(sessions[1:]) == {sessions[1]}


def test_the_new_session_starts_with_the_first_prompt_and_the_continuation(
    late_model_run,
) -> None:
    sample = late_model_run
    run = sample.store_as(AgentRun)
    answered = [
        e
        for e in sample.events
        if isinstance(e, ModelEvent) and e.error is None and "Bash" in {t.name for t in e.tools}
    ]
    assert answered
    # Upstream's `--continue` sends the session's first prompt and then the continuation; the
    # new session sends both too, in that order (D-105).
    text = "\n".join(m.text for m in answered[0].input if isinstance(m, ChatMessageUser))
    assert isinstance(sample.input, str)
    first_prompt = sample.input.strip()
    continuation = run.rounds[1]["prompt"]
    assert continuation.startswith(CONTINUE)
    assert first_prompt in text
    assert continuation in text
    assert text.index(first_prompt) < text.index(continuation)
