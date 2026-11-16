"""A sandbox that records what it is asked to run, for tests of command shape (no Docker)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.util import ExecResult


@dataclass
class Call:
    cmd: list[str]
    user: str | None
    cwd: str | None
    env: dict[str, str]
    timeout: int | None

    @property
    def script(self) -> str:
        """The `bash -c` script, or the command joined."""
        if len(self.cmd) >= 3 and self.cmd[:2] == ["bash", "-c"]:
            return self.cmd[2]
        return " ".join(self.cmd)


Responder = Callable[[Call], ExecResult[str] | None]


@dataclass
class FakeSandbox:
    """Answers `exec` with `respond(call)` (default: success, no output) and keeps files."""

    respond: Responder = lambda call: None
    calls: list[Call] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)

    async def exec(
        self,
        cmd: list[str],
        input: Any = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,  # noqa: ASYNC109 (the signature of sandbox().exec)
        **kwargs: Any,
    ) -> ExecResult[str]:
        call = Call(list(cmd), user, cwd, dict(env or {}), timeout)
        self.calls.append(call)
        result = self.respond(call)
        return result if result is not None else ExecResult(True, 0, "", "")

    async def write_file(self, file: str, contents: str | bytes) -> None:
        self.files[file] = contents.encode() if isinstance(contents, str) else contents

    async def read_file(self, file: str, text: bool = True) -> Any:
        if file not in self.files:
            raise FileNotFoundError(file)
        data = self.files[file]
        return data.decode() if text else data

    def find(self, predicate: Callable[[Call], bool]) -> list[Call]:
        return [c for c in self.calls if predicate(c)]


def ok(stdout: str = "") -> ExecResult[str]:
    return ExecResult(True, 0, stdout, "")


def failed(returncode: int = 1, stderr: str = "") -> ExecResult[str]:
    return ExecResult(False, returncode, "", stderr)
