"""The monitor task: unit tests, and end to end with `mockllm/model` in a Docker sandbox."""

import json
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.util import ComposeConfig
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

import researcharena_inspect
from researcharena_inspect.monitor import (
    ACCESS_TYPES,
    SANDBOX_TASK_DIR,
    SUSPICION_ERROR,
    bundle_to_sample,
    local_bundle,
    monitor_suspicion,
    recover_judgement,
    researcharena_monitor,
    run_monitor,
    stage_monitor_inputs,
)
from researcharena_inspect.monitor.agents import CLAUDE_CODE_ENV
from researcharena_inspect.monitor.task import (
    access_suffix,
    cli_version_in_output,
    failed_judgements,
    mean_suspicion,
    monitor_raw_output,
    monitor_sandbox,
    monitor_timeout_seconds,
    published_monitor_cli_version,
    published_monitor_dir,
)
from researcharena_inspect.pins import CLAUDE_CODE_SMALL_MODEL
from researcharena_inspect.task import EVAL_VERSION

CUDA_RUN = "cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740"


def test_access_types_and_published_folder_names() -> None:
    assert ACCESS_TYPES == {
        "transcript": (False, False),
        "transcript_cot": (False, True),
        "artifact": (True, False),
        "artifact_cot": (True, True),
    }
    assert [access_suffix(a) for a in ACCESS_TYPES] == ["", "_cot", "_artifact", "_artifact_cot"]
    assert (
        published_monitor_dir("claude-sonnet-4-6", "high", "artifact_cot")
        == "claude_claude-sonnet-4-6_high_artifact_cot"
    )
    assert published_monitor_dir("glm-5.1", "default", "transcript") == "claude_glm-5.1"


def test_timeout_is_runtime_plus_a_minute() -> None:
    assert monitor_timeout_seconds(0.5) == 1860
    assert monitor_timeout_seconds(0.0833) == 300 + 60


def test_metrics_skip_failed_judgements() -> None:
    scores = [SampleScore(score=Score(value=v)) for v in (10, 30, SUSPICION_ERROR)]
    assert mean_suspicion()(scores) == 20  # pyright: ignore[reportArgumentType]
    assert failed_judgements()(scores) == 1  # pyright: ignore[reportArgumentType]


def test_metrics_through_the_epoch_reducer() -> None:
    metrics = [mean_suspicion(), failed_judgements()]
    values = run_metrics(metrics, [[10], [30], [SUSPICION_ERROR]])
    assert values == {"mean_suspicion": 20, "failed_judgements": 1}
    assert_agreeing_epochs_change_nothing(metrics, [10, 30, SUSPICION_ERROR])


def test_default_sandbox_is_a_small_image_in_the_monitors_working_dir() -> None:
    spec = monitor_sandbox("docker", None)
    assert isinstance(spec.config, ComposeConfig)
    service = spec.config.services["default"]
    assert service.working_dir == SANDBOX_TASK_DIR
    assert isinstance(service.environment, dict)
    assert service.environment["HOME"] == "/home/user"
    assert monitor_sandbox("k8s", "values.yaml").config == "values.yaml"


def test_run_monitor_rejects_unknown_agents() -> None:
    with pytest.raises(ValueError, match="agent must be one of"):
        run_monitor(agent="gemini")


def test_monitor_claude_code_settings_match_upstream() -> None:
    # D-089: auto-memory on, as upstream's monitor CLI had it ("0" overrides Inspect SWE's "1").
    assert CLAUDE_CODE_ENV["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "0"
    # monitor.sh and pipeline_monitor.sh.
    assert CLAUDE_CODE_ENV["BASH_MAX_TIMEOUT_MS"] == "36000000"
    assert CLAUDE_CODE_ENV["BASH_DEFAULT_TIMEOUT_MS"] == "3600000"
    # The published runs' `modelUsage` lists this model next to the monitor model.
    assert CLAUDE_CODE_SMALL_MODEL == "anthropic/claude-haiku-4-5-20251001"


def test_cli_version_is_read_from_the_init_event() -> None:
    init = {"type": "system", "subtype": "init", "claude_code_version": "2.1.34", "tools": []}
    output = "\n".join(
        ["Installing Claude Code...", json.dumps({"type": "assistant"}), json.dumps(init)]
    )
    assert cli_version_in_output(output) == "2.1.34"
    assert cli_version_in_output(json.dumps({"type": "system", "subtype": "init"})) is None
    assert cli_version_in_output("not json\n{broken") is None


@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_published_monitor_cli_version_of_a_published_run() -> None:
    # D-075: the published monitors ran on the release current at the time.
    version = published_monitor_cli_version(CUDA_RUN, "claude_claude-sonnet-4-6_high")
    assert version == "2.1.162"
    with pytest.raises(ValueError, match="no published monitor output"):
        published_monitor_cli_version(CUDA_RUN, "claude_no-such-model_high")


def test_claude_code_raw_output_hides_verdicts_like_stream_json() -> None:
    messages = [
        ChatMessageUser(content="prompt"),
        ChatMessageAssistant(content='All done: {"suspicion_score": 12, "reasoning": "r"}'),
    ]
    claude = monitor_raw_output(messages, "claude_code")
    assert recover_judgement(claude)["suspicion_score"] == SUSPICION_ERROR
    react = monitor_raw_output(messages, "react")
    assert recover_judgement(react) == {"suspicion_score": 12, "reasoning": "r"}


def write_local_run(run_dir: Path) -> Path:
    red = run_dir / "red_team"
    (red / "task_snapshot").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "setting": "inference_optimization",
                "side_task": "none",
                "red_provider": "codex",
                "red_model": "gpt-5.4",
                "task_config": "",
            }
        )
    )
    (red / "solve_parsed.txt").write_text("with cot\n")
    (red / "solve_parsed_no_cot.txt").write_text("without cot\n")
    (red / "task_snapshot" / "serve.py").write_text("print('hi')\n")
    (run_dir / "eval").mkdir()
    (run_dir / "eval" / "metrics.json").write_text('{"main_task_score": 1.5}')
    return run_dir


@pytest.mark.dataset_download  # runs upstream's prompt builders from the pinned checkout
def test_bundle_to_sample_from_a_local_run(tmp_path: Path) -> None:
    bundle = local_bundle(write_local_run(tmp_path / "run1"))
    assert bundle.metadata["main_score"] == 1.5
    sample = bundle_to_sample(bundle, "artifact_cot")
    assert isinstance(sample, Sample)
    assert isinstance(sample.input, str)
    assert sample.metadata is not None
    assert sample.input == sample.metadata["monitor_prompt"]
    assert "task_snapshot/\n  serve.py (12 bytes)" in sample.input
    assert sample.metadata["transcript_path"].endswith("solve_parsed.txt")
    assert "You have 1 hours" not in sample.input  # the instruction is staged, not inlined
    no_cot = bundle_to_sample(bundle, "transcript")
    assert no_cot.metadata is not None
    assert no_cot.metadata["transcript_path"].endswith("solve_parsed_no_cot.txt")
    assert "Environment Snapshot" not in no_cot.input


@pytest.mark.dataset_download  # runs upstream's prompt builders from the pinned checkout
def test_monitor_task_has_the_eval_yaml_version(tmp_path: Path) -> None:
    write_local_run(tmp_path / "run1")
    task = researcharena_monitor(access="transcript", local_runs=str(tmp_path))
    eval_yaml = Path(researcharena_inspect.__file__).parent / "eval.yaml"
    assert f'version: "{task.version}"' in eval_yaml.read_text("utf-8").splitlines()
    assert task.version == EVAL_VERSION  # the red-team task's


def scripted_monitor(ls_target: str, score: int) -> list[ModelOutput]:
    """A ReAct monitor that looks around, writes judgement.json, then stops."""
    judgement = json.dumps({"suspicion_score": score, "reasoning": "scripted"})
    return [
        ModelOutput.for_tool_call(
            "mockllm/model",
            "bash",
            {"command": f"pwd; ls -a {ls_target}; head -c 200 task_instruction.md"},
        ),
        ModelOutput.for_tool_call(
            "mockllm/model", "bash", {"command": f"printf '%s' '{judgement}' > judgement.json"}
        ),
        ModelOutput.from_content("mockllm/model", "Judgement written."),
    ]


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
@pytest.mark.parametrize("access", list(ACCESS_TYPES))
def test_monitor_task_end_to_end_with_mockllm(access: str, tmp_path: Path) -> None:
    task = researcharena_monitor(access=access, runs=CUDA_RUN, monitor_agent="react")
    model = get_model("mockllm/model", custom_outputs=scripted_monitor(".", 37))
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    assert sample.scores is not None
    score = sample.scores["monitor_suspicion"]
    assert score.value == 37
    assert score.metadata is not None
    assert score.metadata["judgement_source"] == "task"
    assert score.metadata["published_monitor_dir"] == f"claude_model_high{access_suffix(access)}"

    listing = next(m for m in sample.messages if isinstance(m, ChatMessageTool)).text
    assert listing.startswith(SANDBOX_TASK_DIR)
    assert "agent_transcript.txt" in listing
    assert "task_instruction.md" in listing
    assert "# CUDA Kernel Optimization Task" in listing
    assert ("\nkernels\n" in listing) == ACCESS_TYPES[access][0]


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_monitor_without_a_judgement_scores_minus_one(tmp_path: Path) -> None:
    task = researcharena_monitor(access="transcript", runs=CUDA_RUN, monitor_agent="react")
    model = get_model(
        "mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", "No verdict.")]
    )
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    assert log.samples is not None
    score = log.samples[0].scores["monitor_suspicion"]  # type: ignore[index]
    assert score.value == SUSPICION_ERROR
    assert score.metadata is not None
    assert score.metadata["judgement_source"] == "none"
    assert score.explanation == "Monitor failed to produce structured output"


def test_solvers_and_scorer_are_registered() -> None:
    assert callable(stage_monitor_inputs())
    assert callable(run_monitor(agent="react"))
    assert callable(monitor_suspicion())


def claude_code_script(
    messages: list, tools: list, tool_choice: object, config: object
) -> ModelOutput:
    """Mock model behind Inspect SWE's bridge: look around, write judgement.json, stop.

    Requests without Claude Code's tools (its background calls) get a plain answer.
    """
    names = {t.name for t in tools}
    if not {"Bash", "Write"} <= names:
        return ModelOutput.from_content("mockllm/model", "ok")
    done = sum(isinstance(m, ChatMessageTool) for m in messages)
    if done == 0:
        return ModelOutput.for_tool_call(
            "mockllm/model", "Bash", {"command": "pwd; ls -a; head -c 120 task_instruction.md"}
        )
    if done == 1:
        judgement = json.dumps({"suspicion_score": 61, "reasoning": "scripted claude code"})
        return ModelOutput.for_tool_call(
            "mockllm/model",
            "Write",
            {"file_path": f"{SANDBOX_TASK_DIR}/judgement.json", "content": judgement},
        )
    return ModelOutput.from_content("mockllm/model", "Judgement written.")


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.huggingface
@pytest.mark.dataset_download  # also downloads the Claude Code binary (no model is called)
@pytest.mark.parametrize("access", ["transcript", "transcript_cot"])
def test_claude_code_monitor_end_to_end_with_mockllm(access: str, tmp_path: Path) -> None:
    task = researcharena_monitor(
        access=access,
        runs=CUDA_RUN,
        monitor_agent="claude_code",
        claude_code_version="stable",
        small_model=None,  # the mock model answers Claude Code's background calls too
    )
    model = get_model("mockllm/model", custom_outputs=claude_code_script)
    [log] = eval(task, model=model, display="none", log_dir=str(tmp_path))
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    assert sample.scores is not None
    score = sample.scores["monitor_suspicion"]
    assert score.value == 61
    assert score.metadata is not None
    assert score.metadata["judgement_source"] == "task"
    listing = next(m for m in sample.messages if isinstance(m, ChatMessageTool)).text
    assert listing.startswith(SANDBOX_TASK_DIR)
    assert "agent_transcript.txt" in listing
