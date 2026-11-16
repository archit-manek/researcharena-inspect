"""What a red-team run leaves behind, and how it becomes a run folder in upstream's layout.

During a sample the solver keeps an `AgentRun` in the sample's store: the CLI's stdout lines
(upstream's `red_team/solve_out.json`), its stderr and the loop's log (`solve_err.txt`), the
rounds, the CLI version (`cli_version.txt`) and the elapsed time (`time_taken.txt`). The store
is saved in the `.eval` log, so the run can be rebuilt after the sandbox is gone.

`write_run_dir` turns one logged sample into the folder `src/pipeline_redteam.sh` writes
(`metadata.json`, `instruction.md`, `red_team/...`), with both monitor transcripts made by
upstream's parser, so `monitor.bundle.local_bundle` and the monitor task can read it. The
agent's final workspace (`red_team/task_snapshot/`) is not in the log; see the README.
"""

import json
import time
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from inspect_ai.util import StoreModel
from pydantic import Field

from researcharena_inspect.prompts import BASELINE
from researcharena_inspect.transcripts import (
    Transcripts,
    messages_to_stream_events,
    parse_cli_output,
    render_messages,
)


class AgentRun(StoreModel):
    """One sample's red-team run, kept in the sample's store."""

    agent: str = ""
    """`claude_code` or `react`."""
    model: str = ""
    """The task model's name, as upstream's `RED_MODEL` (e.g. `claude-sonnet-4-6`)."""
    effort: str = ""
    """The resolved effort (`RED_EFFORT_LEVEL`); `default` means none was set."""
    redruntime: str = ""
    budget_seconds: int = 0
    red_timeout_seconds: int = 0
    resume: bool = False
    cli_path: str | None = None
    cli_version: str | None = None
    """`claude --version` after the install (`version:` in `cli_version.txt`)."""
    cli_requested: str | None = None
    red_start: float | None = None
    """Epoch seconds when the agent phase began (`RED_START`), before the CLI install."""
    budget_start: int | None = None
    """`START_TS` of `solve.sh`: whole epoch seconds when the agent's clock started."""
    red_end: float | None = None
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    exit_code: int | None = None
    """The last round's code (`AGENT_RC`): 0, 124 (budget) or 1 (the agent raised)."""
    outer_timeout: bool = False
    """Whether `RED_TIMEOUT` (budget + 5 minutes) stopped the agent phase."""
    stdout: list[str] = Field(default_factory=list)
    """The CLI's stdout, one JSON event per line, every round in order (Claude Code only)."""
    stderr: list[str] = Field(default_factory=list)
    """The CLI's stderr chunks (Claude Code only)."""
    log: list[str] = Field(default_factory=list)
    """The `[agent] ...` lines `solve.sh` writes to stderr, and errors."""
    killed: list[str] = Field(default_factory=list)
    """Processes the agent left running, killed when it stopped ("<pid> <command line>")."""

    @property
    def time_taken_s(self) -> int | None:
        """`RED_ELAPSED`, in whole seconds (`time_taken.txt`)."""
        if self.red_start is None or self.red_end is None:
            return None
        return int(self.red_end) - int(self.red_start)


def red_provider(agent: str) -> str:
    """The provider name a run folder carries: `claude` for Claude Code, else the agent name."""
    return "claude" if agent == "claude_code" else agent


def raw_output(sample: EvalSample, run: AgentRun) -> str:
    """The text of `red_team/solve_out.json`.

    For Claude Code, the CLI's stdout. ReAct has no CLI, so its messages are converted to the
    `stream-json` events the Claude parser reads (`transcripts.messages_to_stream_events`).
    """
    # D-TBD-3g: upstream's file starts with the stdout of pre_agent_setup.sh ("CUDA OK: ...",
    # "Pre-agent setup complete."), which the parser skips; here it holds the CLI's lines only.
    if run.agent == "claude_code":
        return "".join(line + "\n" for line in run.stdout)
    events = messages_to_stream_events(sample.messages, model=run.model or None)
    return "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)


def sample_transcripts(sample: EvalSample, run: AgentRun, raw_path: Path) -> Transcripts:
    """The two monitor transcripts, by upstream's Claude parser."""
    if run.agent == "claude_code":
        return parse_cli_output(raw_path, "claude")
    return render_messages(sample.messages, model=run.model or None)


def _run_name(sample: EvalSample, run: AgentRun) -> str:
    # pipeline_redteam.sh:95-108: <setting>__<side tag>__<provider>__<model>__<time>_c<cluster>
    meta = sample.metadata
    side_tag = "baseline" if meta["side_task"] == BASELINE else meta["side_task"]
    started = time.gmtime(run.red_start if run.red_start is not None else 0)
    stamp = time.strftime("%Y%m%d_%H%M%S", started)
    model_safe = run.model.replace("/", "_")
    return (
        f"{meta['setting']}__{side_tag}__{red_provider(run.agent)}__{model_safe}"
        f"__{stamp}_e{sample.epoch}"
    )


def write_run_dir(sample: EvalSample, root: str | Path, experiment: str = "inspect") -> Path:
    """Write one logged sample as a run folder in upstream's layout under `root`.

    Returns:
        The run folder: `<root>/<run name>/` with `metadata.json`, `instruction.md` and
        `red_team/{solve_out.json, solve_err.txt, time_taken.txt, solve_parsed.txt,
        solve_parsed_no_cot.txt}` (and `cli_version.txt` for Claude Code).
    """
    run = sample.store_as(AgentRun)
    if not run.agent:
        raise ValueError(f"sample {sample.id} has no red-team run in its store")
    meta = sample.metadata
    run_dir = Path(root) / _run_name(sample, run)
    red = run_dir / "red_team"
    red.mkdir(parents=True, exist_ok=False)

    started = time.gmtime(run.red_start if run.red_start is not None else 0)
    metadata = {
        # The keys of pipeline_redteam.sh:146-162, in its order.
        "setting": meta["setting"],
        "side_task": meta["side_task"],
        "redruntime_hours": json.loads(run.redruntime or meta["redruntime"]),
        "red_provider": red_provider(run.agent),
        "red_model": run.model,
        "red_effort_level": run.effort,
        "resume_agent": run.resume,
        "cluster_id": str(sample.epoch),
        "experiment": experiment,
        "task_config": meta["task_config"],
        "seed": None,
        "baseline": meta["side_task"] == BASELINE,
        "timestamp": time.strftime("%Y%m%d_%H%M%S", started),
        # Not in upstream's file.
        "inspect_agent": run.agent,
        "inspect_sample_id": sample.id,
        "inspect_sample_uuid": sample.uuid,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=4) + "\n", "utf-8")
    (run_dir / "instruction.md").write_text(meta["instruction_file"], "utf-8")

    raw_path = red / "solve_out.json"
    raw_path.write_text(raw_output(sample, run), "utf-8")
    (red / "solve_err.txt").write_text(
        "".join(line + "\n" for line in run.log) + "".join(run.stderr), "utf-8"
    )
    if run.time_taken_s is not None:
        (red / "time_taken.txt").write_text(f"{run.time_taken_s}\n", "utf-8")
    if run.cli_version is not None:
        recorded = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", started)
        (red / "cli_version.txt").write_text(
            "binary: claude\n"
            f"package: claude-code native build (Inspect SWE)\n"
            f"path: {run.cli_path}\n"
            f"version: {run.cli_version}\n"
            f"requested_version: {run.cli_requested}\n"
            f"recorded_at: {recorded}\n",
            "utf-8",
        )

    transcripts = sample_transcripts(sample, run, raw_path)
    (red / "solve_parsed.txt").write_text(transcripts.with_cot, "utf-8")
    (red / "solve_parsed_no_cot.txt").write_text(transcripts.without_cot, "utf-8")
    return run_dir


def write_run_dirs(log: str | Path | EvalLog, root: str | Path) -> list[Path]:
    """Write every sample of an eval log as a run folder under `root`."""
    if not isinstance(log, EvalLog):
        log = read_eval_log(str(log))
    return [write_run_dir(sample, root) for sample in log.samples or []]
