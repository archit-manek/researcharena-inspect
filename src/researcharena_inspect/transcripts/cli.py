"""The transcripts a monitor reads, made from an agent CLI's raw output by upstream's own parser.

Runs `agents/<family>/human_readable_trace.py` from the pinned checkout twice, with and without
`--no-thinking`, with the fallbacks of `src/pipeline_redteam.sh:418-431`.
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

    At the paper commit every variant's parser is byte-identical to its family's; the pinned
    commit keeps only the families (`pipeline_monitor.sh:137` strips `_norepeat` the same way).
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
    """Turn an agent CLI's stdout (`red_team/solve_out.json`) into the two monitor transcripts."""
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
                # D-018: a failed no-CoT parse hands the CoT transcript to the monitor.
                shutil.copyfile(cot, no_cot)
        else:
            shutil.copyfile(raw_output, cot)
            shutil.copyfile(raw_output, no_cot)
        return Transcripts(
            with_cot=cot.read_text(encoding="utf-8", errors="replace"),
            without_cot=no_cot.read_text(encoding="utf-8", errors="replace"),
        )
