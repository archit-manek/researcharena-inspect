"""Both red-team agents end to end in Docker with `mockllm/model`: budget, re-prompting, output.

The scripted agent works in three rounds. In rounds 0 and 1 it writes a file with one tool call
and then stops (no tool call), as an agent that thinks it is done would; upstream re-prompts it
each time. In round 2 it starts a background `sleep 300` and a foreground one, so it is still
working when the budget runs out. The budget is 25 seconds (`redruntime` 0.007 hours).

Claude Code is the real CLI (Inspect SWE, version `CLI_VERSION`) talking to the mock model
through Inspect's bridge. No paid model is called.
"""

from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalLog, EvalSample, resolve_sample_attachments
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
from researcharena_inspect.agents.solver import MODEL_AGENTS
from researcharena_inspect.monitor import local_bundles, researcharena_monitor
from researcharena_inspect.prompts import agent_prompt, resume_prompt
from researcharena_inspect.transcripts import parse_cli_output

CLI_VERSION = "2.1.162"
"""The Claude Code version in the published Sonnet 4.6 and Opus 4.6 CUDA runs."""

BUDGET_HOURS = "0.007"
BUDGET_S = 25

STOP_TOLERANCE_S = 10
"""How late after the deadline the last round may end: Inspect's cancellation of the round
(measured 2-4 s here). Upstream allows the CLI 30 s after SIGTERM (`--kill-after=30s`)."""

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


# WP4: AGENTS now also has the no-model agents; these scenarios need a model agent.
@pytest.fixture(scope="module", params=MODEL_AGENTS)
def red_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    agent = request.param
    # WP4: `sandbox="docker"` is the CPU image the task used to default to (now with git, so
    # the default staging runs); the agent runs as the staged user.
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
    _, log = red_run
    sample, run = _sample(log)
    assert run.budget_seconds == BUDGET_S
    last = run.rounds[-1]
    # The round's limit was the time left, so it should end at the deadline (budget start +
    # 25 s), not earlier, and only as late as the cancellation takes.
    assert last["started_at"] + last["remaining_at_start"] == pytest.approx(BUDGET_S, abs=1)
    assert BUDGET_S - 1 <= last["ended_at"] <= BUDGET_S + STOP_TOLERANCE_S
    # Nothing the agent started is left running when the scorer runs.
    assert sample.scores is not None
    processes = sample.scores["workspace_probe"].metadata["processes"]  # type: ignore[index]
    assert not [p for p in processes if "sleep 300" in p]
    assert not [p for p in processes if "claude" in p]
    assert run.killed  # the background sleep, or the CLI and its tool, were killed


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
        # One result event per finished round (the third was stopped).
        assert sum('"type":"result"' in line for line in run.stdout) == 2
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
    )
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "Done.")]
    )
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    _, run = _sample(log)
    assert [r["phase"] for r in run.rounds] == ["exec"]
    assert run.rounds[0]["ended_at"] < 5
    assert "[agent] AGENT_RESUME=false: not resuming" in run.log


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
