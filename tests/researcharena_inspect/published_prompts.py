"""Recover prompt text from the authors' published runs, to compare with the port's builders.

The published runs (`aisa-group/ResearchArena-Trajectories` at the pinned revision) carry no
prompt files: no `instruction.md`, `prompt.txt` or `monitor_prompt.md`. The text survives
only where an agent or a monitor printed it during its run:

- Monitors read `task_instruction.md` with a tool. Claude Code's `Read` result shows each
  line after its number; Codex's `cat` or `sed -n '1,Np'` output is the file itself. This
  gives the monitor's main-task text exactly (up to a final newline).
- OpenCode red-team agents in published runs got their prompt as a command-line argument.
  When such an agent ran `ps`, the output shows that argument with newlines as spaces,
  possibly cut at the terminal width. Resume rounds show the "continue" prompt the same way.
- An agent that read `prompt.txt` in its home shows the whole prompt (Claude `Read`, `cat`).

Nothing in the published files shows a monitor's own prompt (its stream doesn't echo it).
"""

import json
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from huggingface_hub import HfApi

from researcharena_inspect.pins import HF_DATASET, HF_REVISION, PAPER_COMMIT
from researcharena_inspect.prompts import (
    ACCESS_TYPES,
    INSTRUCTION_FILENAME,
    AgentFamily,
    MonitorAccess,
    agent_prompt,
    monitor_task_instruction,
)
from researcharena_inspect.prompts.agent import RESUME_PROMPT_FORMAT
from researcharena_inspect.published import fetch_run_file
from researcharena_inspect.upstream import upstream_dir

FAMILIES = ("claude", "codex", "opencode")


def family_of(red_provider: str) -> str:
    """ "claude_norepeat" -> "claude", "codex_norepeat_high" -> "codex"."""
    return red_provider.split("_", 1)[0]


def sample_runs(runs: list[dict[str, Any]], per_family: int = 1) -> list[dict[str, Any]]:
    """Per setting and agent family: the first baseline run and the first side-task runs.

    Side-task runs rotate through the setting's side tasks so the sample covers several.
    Deterministic: runs are taken in `id` order.
    """
    chosen: list[dict[str, Any]] = []
    for setting in sorted({r["setting"] for r in runs}):
        in_setting = sorted((r for r in runs if r["setting"] == setting), key=lambda r: r["id"])
        side_tasks = sorted({r["side_task"] for r in in_setting} - {"baseline"})
        for i, family in enumerate(FAMILIES):
            mine = [r for r in in_setting if family_of(r["red_provider"]) == family]
            baselines = [r for r in mine if r["side_task"] == "baseline"][:per_family]
            chosen.extend(baselines)
            for j in range(per_family):
                wanted = side_tasks[(i * per_family + j) % len(side_tasks)]
                attack = [r for r in mine if r["side_task"] == wanted] or [
                    r for r in mine if r["side_task"] != "baseline"
                ]
                if attack:
                    chosen.append(attack[0])
    return chosen


def run_files(run_id: str) -> list[str]:
    """Paths of a published run's files, relative to the run's folder (Hugging Face tree API)."""
    api = HfApi()
    items = api.list_repo_tree(
        HF_DATASET, path_in_repo=run_id, recursive=True, repo_type="dataset", revision=HF_REVISION
    )
    prefix = run_id + "/"
    return sorted(
        item.path[len(prefix) :]
        for item in items
        if hasattr(item, "size") and "/task_snapshot/" not in "/" + item.path[len(prefix) :]
    )


def _json_lines(text: str) -> Iterator[dict[str, Any]]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


_NUMBERED_LINE = re.compile(r"^\s*(\d+)(?:\t|→)(.*)$")


def strip_read_numbers(content: str) -> list[str] | None:
    """The file lines of a Claude Code `Read` result ("N\\tline" or "     N→line")."""
    lines: list[str] = []
    for raw in content.split("\n"):
        match = _NUMBERED_LINE.match(raw)
        if match is None:
            if lines and (raw.startswith("<system-reminder>") or not raw.strip()):
                break
            return None
        if int(match.group(1)) != len(lines) + 1:
            return None
        lines.append(match.group(2))
    return lines or None


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


@dataclass(frozen=True)
class Recovered:
    """One copy of a file's text found in a transcript."""

    source: str  # what printed it, e.g. "claude Read", "codex cat"
    lines: list[str]  # the file's lines, without newlines


def claude_reads(stream: str, filename: str) -> list[Recovered]:
    """Whole-file reads of `filename` in a Claude Code stream-json log (Read or `cat`)."""
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    found: list[Recovered] = []
    for event in _json_lines(stream):
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                calls[block.get("id", "")] = (block.get("name", ""), block.get("input") or {})
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                name, args = calls[block["tool_use_id"]]
                text = _tool_result_text(block.get("content"))
                if name == "Read" and str(args.get("file_path", "")).endswith(filename):
                    if args.get("offset") or args.get("limit"):
                        continue
                    lines = strip_read_numbers(text)
                    if lines is not None:
                        found.append(Recovered("claude Read", lines))
                elif name == "Bash" and _is_plain_cat(str(args.get("command", "")), filename):
                    found.append(Recovered("claude cat", text.split("\n")))
    return found


def _is_plain_cat(command: str, filename: str) -> bool:
    command = command.strip()
    pattern = rf"cat\s+(?:\S*/)?{re.escape(filename)}"
    return re.fullmatch(pattern, command) is not None


_CODEX_SHELL = re.compile(r"""^/bin/(?:ba)?sh -lc (['"])(.*)\1$""", re.DOTALL)
_SED_RANGE = re.compile(r"""sed -n ['"]?1,(\d+)p['"]?\s+(?:\S*/)?(\S+)""")


def codex_reads(stream: str, filename: str) -> list[Recovered]:
    """Whole-file reads of `filename` in a Codex `--json` log (`cat` or `sed -n '1,Np'`)."""
    found: list[Recovered] = []
    for event in _json_lines(stream):
        item = event.get("item")
        if event.get("type") != "item.completed" or not isinstance(item, dict):
            continue
        if item.get("type") != "command_execution" or item.get("exit_code") != 0:
            continue
        command = str(item.get("command", ""))
        shell = _CODEX_SHELL.match(command)
        inner = shell.group(2) if shell else command
        output = str(item.get("aggregated_output", ""))
        lines = output.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if _is_plain_cat(inner, filename):
            found.append(Recovered("codex cat", lines))
            continue
        sed = _SED_RANGE.fullmatch(inner.strip())
        if sed and sed.group(2).endswith(filename) and int(sed.group(1)) > len(lines):
            found.append(Recovered("codex sed", lines))
    return found


@dataclass(frozen=True)
class ArgvPrompt:
    """A prompt seen as a command-line argument in `ps` output (newlines shown as spaces)."""

    resume: bool
    text: str


_OPENCODE_ARGV = re.compile(
    r"opencode run --format json (?:--thinking )?(?:--variant \S+ )?--model \S+ (--continue )?(.*)"
)


def opencode_argv_prompts(stream: str) -> list[ArgvPrompt]:
    """Prompts shown in `ps` output that an OpenCode agent printed during its run."""
    found: list[ArgvPrompt] = []
    seen: set[str] = set()
    for event in _json_lines(stream):
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        state = part.get("state")
        if not isinstance(state, dict):
            continue
        output = state.get("output")
        if not isinstance(output, str) or "opencode run" not in output:
            continue
        for line in output.split("\n"):
            match = _OPENCODE_ARGV.search(line)
            if match is None or not match.group(2).strip():
                continue
            text = match.group(2)
            if text in seen or text.startswith('"$'):  # a script listing, not a process
                continue
            seen.add(text)
            found.append(ArgvPrompt(resume=match.group(1) is not None, text=text))
    return found


def ps_view(text: str) -> str:
    """How `ps` shows a command-line argument: newlines as spaces, each non-ASCII byte as "?"."""
    shown = "".join(c if ord(c) < 128 else "?" * len(c.encode("utf-8")) for c in text)
    return shown.replace("\n", " ")


def common_prefix(a: str, b: str) -> int:
    """Length of the longest common prefix of `a` and `b`."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------- classification

# Published runs call the inference exfiltration side task "independent2"; the code calls it
# "embedded3" (ledger D-050).
SIDE_TASK_ALIASES = {("inference_optimization", "independent2"): "embedded3"}

IDENTICAL = "identical to b922f4b"
PAPER_ONLY = "identical to 58776e0 (paper) only"
NEITHER = "differs from both"
NO_PORT = "no b922f4b text (unknown name)"


def paper_era_prompt(
    setting: str, side_task: str, task_config: str, red_provider: str, redruntime: str | None
) -> str:
    """The paper commit's own `prompt_builder.py` output, as `$(...)` keeps it.

    With `redruntime` None the flag is left out, as the paper-era `pipeline_monitor.sh`
    does when it renders the monitor's task text.
    """
    root = upstream_dir(PAPER_COMMIT)
    args = [
        "python3",
        str(root / "src" / "prompt_builder.py"),
        "--setting",
        setting,
        "--side_task",
        side_task,
        "--task_config",
        task_config,
        "--red_provider",
        red_provider,
        "--settings_dir",
        str(root / "settings"),
    ]
    if redruntime is not None:
        args += ["--redruntime", redruntime]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return result.stdout.rstrip("\n")


@dataclass(frozen=True)
class Evidence:
    """One published copy of a prompt text, and how it compares."""

    run_id: str
    setting: str
    agent_family: str
    side_task: str
    kind: str  # "monitor task text", "agent prompt" or "resume prompt"
    where: str  # the monitor's folder, or "red_team"
    source: str  # what printed it
    status: str
    detail: str = ""


def monitor_access(dir_name: str) -> MonitorAccess:
    """The access type a published monitor folder name encodes."""
    return MonitorAccess(with_artifact="_artifact" in dir_name, with_cot=dir_name.endswith("_cot"))


def _first_difference(expected: list[str], got: list[str]) -> str:
    for i, (a, b) in enumerate(zip(expected, got, strict=False)):
        if a != b:
            return f"line {i + 1}: b922f4b {a[:70]!r} vs published {b[:70]!r}"
    return f"{len(expected)} lines (b922f4b) vs {len(got)} lines (published)"


def classify_lines(got: list[str], port: list[str] | None, paper: list[str]) -> tuple[str, str]:
    while got and got[-1] == "":
        got = got[:-1]
    if port is not None and got == port:
        return IDENTICAL, ""
    if got == paper:
        return PAPER_ONLY, _first_difference(port, got) if port is not None else ""
    if port is None:
        return NO_PORT, ""
    return NEITHER, _first_difference(port, got)


def classify_ps(got: str, port: str | None, paper: str) -> tuple[str, str]:
    def same(expected: str) -> bool:
        shown = ps_view(expected)
        return got == shown or (len(got) >= 200 and shown.startswith(got))

    if port is not None and same(port):
        return IDENTICAL, "" if got == ps_view(port) else f"first {len(got)} chars (cut by ps)"
    if same(paper):
        return PAPER_ONLY, ""
    if port is None:
        return NO_PORT, ""
    n = common_prefix(got, ps_view(port))
    return (
        NEITHER,
        f"same first {n} chars; then b922f4b {ps_view(port)[n : n + 60]!r} vs published {got[n : n + 60]!r}",
    )


def pick_monitor_dirs(run: dict[str, Any], files: list[str]) -> list[str]:
    """Per access type, the first Claude and the first Codex monitor with a published stream."""
    present = {f.split("/")[1] for f in files if f.endswith("monitor_out_sanitized.json")}
    chosen = []
    for access in ACCESS_TYPES:
        for family in ("claude", "codex"):
            for monitor in run["monitors"]:
                name = monitor["dir"]
                if (
                    name in present
                    and name.startswith(family + "_")
                    and monitor_access(name) == access
                ):
                    chosen.append(name)
                    break
    return chosen


def collect_evidence(run: dict[str, Any]) -> list[Evidence]:
    """Every prompt text recoverable from one published run, compared with both commits."""
    setting = run["setting"]
    side_task = "none" if run["side_task"] == "baseline" else run["side_task"]
    side_task = SIDE_TASK_ALIASES.get((setting, side_task), side_task)
    family = family_of(run["red_provider"])
    files = run_files(run["id"])
    base = {
        "run_id": run["id"],
        "setting": setting,
        "agent_family": family,
        "side_task": run["side_task"],
    }
    found: list[Evidence] = []

    # The monitor's main-task text, from each sampled monitor's read of task_instruction.md.
    port_text = monitor_task_instruction(setting, run["task_config"]).rstrip("\n").split("\n")
    paper_text = paper_era_prompt(setting, "none", run["task_config"], "none", None).split("\n")
    for name in pick_monitor_dirs(run, files):
        stream = read_text(
            fetch_run_file(run["id"], f"blue_team/{name}/monitor_out_sanitized.json")
        )
        for copy in claude_reads(stream, INSTRUCTION_FILENAME) + codex_reads(
            stream, INSTRUCTION_FILENAME
        ):
            status, detail = classify_lines(copy.lines, port_text, paper_text)
            found.append(
                Evidence(
                    **base,
                    kind="monitor task text",
                    where=name,
                    source=copy.source,
                    status=status,
                    detail=detail,
                )
            )

    # The agent's prompt: reads of prompt.txt, and OpenCode's argv in ps output.
    stream = read_text(fetch_run_file(run["id"], "red_team/solve_out_sanitized.json"))
    hours = str(run["hours"])
    try:
        port_prompt: str | None = agent_prompt(
            setting, side_task, cast(AgentFamily, family), hours, run["task_config"]
        )
    except ValueError:
        port_prompt = None
    paper_prompt = paper_era_prompt(
        setting, side_task, run["task_config"], run["red_provider"], hours
    )
    for copy in claude_reads(stream, "prompt.txt") + codex_reads(stream, "prompt.txt"):
        port_lines = port_prompt.split("\n") if port_prompt is not None else None
        status, detail = classify_lines(copy.lines, port_lines, paper_prompt.split("\n"))
        found.append(
            Evidence(
                **base,
                kind="agent prompt",
                where="red_team",
                source=copy.source,
                status=status,
                detail=detail,
            )
        )
    resume_pattern = (
        re.escape(RESUME_PROMPT_FORMAT)
        .replace(r"\{hours\}", r"\d+")
        .replace(r"\{minutes\}", r"\d+")
    )
    for shown in opencode_argv_prompts(stream):
        if shown.resume:
            status = IDENTICAL if re.fullmatch(resume_pattern, shown.text) else NEITHER
            found.append(
                Evidence(
                    **base,
                    kind="resume prompt",
                    where="red_team",
                    source="ps",
                    status=status,
                    detail=shown.text[50:80],
                )
            )
            continue
        status, detail = classify_ps(shown.text, port_prompt, paper_prompt)
        found.append(
            Evidence(
                **base,
                kind="agent prompt",
                where="red_team",
                source="ps",
                status=status,
                detail=detail,
            )
        )
    return found
