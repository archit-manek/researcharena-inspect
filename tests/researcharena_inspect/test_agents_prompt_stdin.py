"""Claude Code's first prompt goes on stdin, not on the command line (D-078, D-257).

No Docker: the command Inspect SWE builds is checked against its source, and the sandbox is a
recorder. `test_prompt_stdin_docker.py` runs the real CLI.
"""

import importlib
import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.util import ExecRemoteStreamingOptions

from researcharena_inspect.agents import claude_code as cc
from researcharena_inspect.agents.claude_code import prompt_on_stdin, stdin_launch

SWE = importlib.import_module("inspect_swe._claude_code.claude_code")
SHELL = ["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
STDIN_SHELL = ["bash", "-c", 'PROMPT=$(cat); printf \'%s\' "$PROMPT" | "$@"', "bash"]
"""Upstream's line (`solve.sh:104`): the shell reads the prompt, then pipes it to the CLI."""
FLAGS = ["--dangerously-skip-permissions", "--model", "m", "--print", "--output-format"]
FLAGS += ["stream-json", "--verbose"]
PROMPT = "# Task\n\n--print -- 'quoted' $HOME `x`\n\nSide task: do Y."


def launch(session_flag: str, prompt: str = PROMPT) -> list[str]:
    return [*SHELL, "/usr/local/bin/claude", session_flag, "sid", *FLAGS, "--", prompt]


def test_new_session_prompt_moves_to_stdin() -> None:
    cmd, prompt = stdin_launch(launch("--session-id")) or ([], "")
    assert prompt == PROMPT
    assert cmd[:4] == STDIN_SHELL
    # Nothing of the prompt is left on the command line, and no `--` either (as upstream's
    # `claude <flags>` reading stdin).
    assert all(PROMPT not in a and "Side task" not in a for a in cmd)


def test_new_session_keeps_every_flag() -> None:
    cmd, _ = stdin_launch(launch("--session-id")) or ([], "")
    assert cmd[4:] == ["/usr/local/bin/claude", "--session-id", "sid", *FLAGS]


def test_a_prompt_that_looks_like_flags_is_still_the_prompt() -> None:
    for prompt in ("--", "--resume", "-p", ""):
        moved = stdin_launch(launch("--session-id", prompt))
        assert moved is not None
        assert moved[1] == prompt
        assert moved[0][-1] == "--verbose"


def test_resumed_session_is_left_as_upstream_runs_it() -> None:
    # Upstream passes the continuation prompt as an argument (`--continue "<prompt>"`).
    assert stdin_launch(launch("--resume", "Continue where you left off")) is None


@pytest.mark.parametrize(
    "cmd",
    [
        ["claude", "--session-id", "sid", "--print", "--", PROMPT],  # no shell
        [*STDIN_SHELL, "claude", "--session-id", "sid", "--", PROMPT],
        [*SHELL, "claude", "--session-id", "sid", "--", "extra", "--", PROMPT],  # two `--`
        [*SHELL, "claude", "--session-id", "sid", *FLAGS, PROMPT],  # no `--`
        [*SHELL, "claude", "--continue", *FLAGS, "--", PROMPT],
        [*SHELL, "claude", "--", PROMPT],
    ],
)
def test_any_other_launch_is_refused(cmd: list[str]) -> None:
    with pytest.raises(RuntimeError, match="unexpected Claude Code launch"):
        stdin_launch(cmd)


def test_inspect_swe_builds_the_launch_stdin_launch_expects() -> None:
    """The installed Inspect SWE starts the CLI as `stdin_launch` reads it."""
    source = inspect.getsource(SWE.claude_code)
    assert 'cmd=["bash", "-c", \'exec 0</dev/null; "$@"\', "bash"]' in source
    assert '[claude_binary, "--session-id", session_id]' in source
    assert '[claude_binary, "--resume", session_id]' in source
    assert source.count('+ ["--", agent_prompt]') == 2
    assert source.count("sandbox_env(") == 2
    assert source.count(".exec_remote(") == 1


class Recorder:
    """A sandbox that records `exec_remote` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Any, bool]] = []
        self.name = "recorder"

    async def exec_remote(self, cmd: list[str], options: Any = None, *, stream: bool = True):
        self.calls.append((cmd, options, stream))
        return "process"


async def test_the_sandbox_sends_a_new_session_prompt_as_stdin() -> None:
    inner = Recorder()
    sbox = cc._StdinSandbox(inner)
    options = ExecRemoteStreamingOptions(cwd="/home/user/task", env={"A": "1"}, user="user")
    assert await sbox.exec_remote(cmd=launch("--session-id"), options=options) == "process"
    [(cmd, sent, stream)] = inner.calls
    assert stream is True
    assert cmd[:4] == STDIN_SHELL
    assert cmd[-1] == "--verbose"
    assert sent.input == PROMPT
    assert (sent.cwd, sent.env, sent.user, sent.stdin_open) == (
        "/home/user/task",
        {"A": "1"},
        "user",
        False,  # closed once the prompt is written: the CLI reads to the end
    )
    assert options.input is None  # Inspect SWE's own options are not changed
    assert sbox.name == "recorder"  # everything else passes through


async def test_the_sandbox_leaves_a_resumed_session_as_it_is() -> None:
    inner = Recorder()
    options = ExecRemoteStreamingOptions(cwd="/home/user/task")
    cmd = launch("--resume", "Continue where you left off")
    await cc._StdinSandbox(inner).exec_remote(cmd=cmd, options=options)
    assert inner.calls == [(cmd, options, True)]


def test_only_inside_the_context() -> None:
    assert not cc._prompt_on_stdin.get()
    with prompt_on_stdin():
        assert cc._prompt_on_stdin.get()
        replaced = SWE.sandbox_env
        with prompt_on_stdin():  # installed once
            assert SWE.sandbox_env is replaced
        assert cc._prompt_on_stdin.get()
    assert not cc._prompt_on_stdin.get()
    assert getattr(SWE.sandbox_env, cc._STDIN_ATTRIBUTE, False)


BIG_PROMPT = ("Ünïcode 'quotes' \"dq\" $HOME `cmd` \\ -- %s %%\n" * 4000).rstrip("\n")
"""Over 200 KB, more than a pipe holds (65,536 bytes), with shell and printf specials."""


@pytest.mark.parametrize("prompt", [PROMPT, BIG_PROMPT])
def test_the_shell_pipes_the_prompt_exactly(prompt: str, tmp_path: Path) -> None:
    out = tmp_path / "stdin"
    cli = ["sh", "-c", f"cat > {out}; exit 3"]  # a CLI that reads all of stdin
    result = subprocess.run(
        [*STDIN_SHELL, *cli], input=prompt.encode(), capture_output=True, check=False
    )
    assert result.returncode == 3  # the CLI's status, not the shell's
    assert out.read_bytes() == prompt.encode()


@pytest.mark.parametrize("prompt", [PROMPT, BIG_PROMPT])
def test_a_cli_that_exits_without_reading_keeps_its_stderr_and_code(prompt: str) -> None:
    cli = ["sh", "-c", "echo early >&2; exit 42"]
    result = subprocess.run(
        [*STDIN_SHELL, *cli], input=prompt.encode(), capture_output=True, check=False
    )
    assert result.returncode == 42  # the writer's SIGPIPE does not change it
    assert result.stderr == b"early\n"
