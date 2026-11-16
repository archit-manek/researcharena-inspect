"""The transcripts a monitor reads, made from an agent CLI's raw output by upstream's own parser.

Upstream (`src/pipeline_redteam.sh:418-431` at the pinned commit) runs
`agents/<provider>/human_readable_trace.py` twice on the CLI's stdout (`solve_out.json`): once
plain, giving `solve_parsed.txt` (with chain of thought), and once with `--no-thinking`, giving
`solve_parsed_no_cot.txt`. This module runs the same scripts from the pinned checkout,
unchanged, with the same fallbacks:

- parser fails on the CoT pass: the raw output is copied in its place;
- parser fails on the no-CoT pass: the CoT transcript is copied in its place (ledger D-018);
- no parser for the agent: the raw output is used for both.
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.prompts.agent import AGENT_FAMILIES, AgentFamily
from researcharena_inspect.upstream import upstream_dir


@dataclass(frozen=True)
class Transcripts:
    """The two texts a monitor can be given: with and without the agent's chain of thought."""

    with_cot: str
    """What upstream writes to `red_team/solve_parsed.txt`."""

    without_cot: str
    """What upstream writes to `red_team/solve_parsed_no_cot.txt`."""

    def for_access(self, with_cot: bool) -> str:
        return self.with_cot if with_cot else self.without_cot


def agent_family(red_provider: str) -> AgentFamily:
    """The parser family of a red-team provider name, e.g. `claude_norepeat` -> `claude`.

    The published runs name eight providers (`claude`, `claude_norepeat`, `codex`,
    `codex_high`, `codex_norepeat`, `codex_norepeat_high`, `opencode`, `opencode_norepeat`). At
    the paper commit every variant's `human_readable_trace.py` is byte-identical to its family's;
    the pinned commit keeps only the three families (`pipeline_monitor.sh:137` strips
    `_norepeat` the same way when it backfills a transcript).
    """
    family = red_provider.split("_", 1)[0]
    for known in AGENT_FAMILIES:
        if family == known:
            return known
    raise ValueError(f"unknown agent family for provider {red_provider!r}")


def trace_parser(family: str, commit: str = UPSTREAM_COMMIT) -> Path:
    """Upstream's `agents/<family>/human_readable_trace.py` at `commit`."""
    return upstream_dir(commit) / "agents" / family / "human_readable_trace.py"


def run_trace_parser(parser: Path, raw_output: Path, out: Path, no_thinking: bool) -> bool:
    """Run one parser pass as `pipeline_redteam.sh` does; True if it exited 0."""
    cmd = [sys.executable, str(parser), str(raw_output)]
    if no_thinking:
        cmd.append("--no-thinking")
    cmd += ["-o", str(out)]
    # The pipeline discards the parser's stdout and stderr (`>/dev/null 2>&1`).
    result = subprocess.run(cmd, capture_output=True, check=False, cwd=out.parent)
    return result.returncode == 0


def parse_cli_output(
    raw_output: Path, red_provider: str, commit: str = UPSTREAM_COMMIT
) -> Transcripts:
    """Turn an agent CLI's raw output into the two monitor transcripts with upstream's parser.

    Args:
        raw_output: the CLI's stdout, as upstream saves it in `red_team/solve_out.json`.
        red_provider: the run's red-team provider, e.g. `claude` or `codex_norepeat`.
        commit: the upstream commit whose parser to run.

    Returns:
        Both transcripts, decoded as UTF-8 (the parsers write UTF-8).
    """
    family = agent_family(red_provider)
    parser = trace_parser(family, commit)
    with tempfile.TemporaryDirectory(prefix="researcharena_trace_") as tmp:
        tmp_dir = Path(tmp)
        cot = tmp_dir / "solve_parsed.txt"
        no_cot = tmp_dir / "solve_parsed_no_cot.txt"
        if parser.is_file():
            if not run_trace_parser(parser, raw_output, cot, no_thinking=False):
                shutil.copyfile(raw_output, cot)
            if not run_trace_parser(parser, raw_output, no_cot, no_thinking=True):
                # D-018: a failed no-CoT parse silently hands the CoT transcript to the monitor.
                shutil.copyfile(cot, no_cot)
        else:
            shutil.copyfile(raw_output, cot)
            shutil.copyfile(raw_output, no_cot)
        return Transcripts(
            with_cot=cot.read_text(encoding="utf-8", errors="replace"),
            without_cot=no_cot.read_text(encoding="utf-8", errors="replace"),
        )
