"""Runs upstream's `agents/claude/solve.sh`, unchanged, on a simulated clock.

The continuation tests compare the port's loop with what upstream's script does for the same
agent behaviour. The script runs from the pinned checkout with stubs on PATH:

- `date +%s` reads a clock file instead of the system clock;
- `claude` records its argv, stdin and time limit, then "works" for the duration scripted for
  that call: it moves the clock forward by that much and exits with the scripted code, or, if
  the duration reaches the limit `timeout` gave it, moves the clock to the limit and exits 124
  as GNU `timeout` does when it stops a command;
- `timeout` passes its duration to `claude` and runs it;
- the CLI installer in the home folder does nothing.

So every round's prompt, time limit and exit code, the number of rounds, the script's exit
code and its stderr log come from upstream's own control flow.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from researcharena_inspect.upstream import upstream_dir

from .prompt_oracle import bash_allows_empty_arrays_under_nounset

START_EPOCH = 1_800_000_000

_CLAUDE = r"""#!/usr/bin/env bash
log="${STUB_LOG:?}"
n=$(( $(cat "$log/.count" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$log/.count"
d="$log/call$n"
mkdir -p "$d"
printf '%s\0' "$@" > "$d/argv"
cat > "$d/stdin"
echo "${STUB_LIMIT:?}" > "$d/limit"
line=$(sed -n "${n}p" "${STUB_SCRIPT:?}")
[ -n "$line" ] || line=$(tail -n 1 "$STUB_SCRIPT")
set -- $line
duration=$1
rc=$2
now=$(cat "${STUB_CLOCK:?}")
if [ "$duration" -ge "$STUB_LIMIT" ]; then
    echo $(( now + STUB_LIMIT )) > "$STUB_CLOCK"
    exit 124
fi
echo $(( now + duration )) > "$STUB_CLOCK"
exit "$rc"
"""

_TIMEOUT = r"""#!/usr/bin/env bash
# timeout [options] DURATION COMMAND...
while [ $# -gt 0 ]; do
    case "$1" in
        -*) shift ;;
        *) break ;;
    esac
done
export STUB_LIMIT="${1%s}"
shift
exec "$@"
"""

_DATE = r"""#!/usr/bin/env bash
if [ "${1:-}" = "+%s" ]; then cat "${STUB_CLOCK:?}"; else exec /bin/date "$@"; fi
"""


@dataclass(frozen=True)
class UpstreamCall:
    argv: list[str]
    stdin: str
    limit: int

    @property
    def prompt(self) -> str:
        """What the CLI was asked: stdin for the first run, the last argument when resumed."""
        return self.argv[-1] if "--continue" in self.argv else self.stdin


@dataclass(frozen=True)
class UpstreamLoop:
    calls: list[UpstreamCall]
    exit_code: int
    stderr: list[str]
    end_clock: int


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def run_solve_sh(
    tmp: Path,
    rounds: list[tuple[int, int]],
    hours: str = "1",
    resume: bool = True,
    prompt: str = "Do the task.\nIt has `backticks` and $dollars.",
) -> UpstreamLoop:
    """Run upstream's Claude `solve.sh` with scripted rounds of (duration_s, exit code)."""
    script = upstream_dir() / "agents" / "claude" / "solve.sh"
    bin_dir, home, log = tmp / "bin", tmp / "home", tmp / "log"
    for d in (bin_dir, home, log):
        d.mkdir(parents=True)
    _write_executable(bin_dir / "claude", _CLAUDE)
    _write_executable(bin_dir / "timeout", _TIMEOUT)
    _write_executable(bin_dir / "date", _DATE)
    _write_executable(home / "update_agent_cli.sh", "#!/usr/bin/env bash\nexit 0\n")
    (tmp / "rounds.txt").write_text("".join(f"{d} {rc}\n" for d, rc in rounds), "utf-8")
    (tmp / "clock").write_text(f"{START_EPOCH}\n", "utf-8")
    (tmp / "prompt.txt").write_text(prompt + "\n", "utf-8")  # as `echo "$PROMPT"` writes it
    env = {
        "HOME": str(home),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "STUB_SCRIPT": str(tmp / "rounds.txt"),
        "STUB_CLOCK": str(tmp / "clock"),
        "PROMPT_FILE": str(tmp / "prompt.txt"),
        "CLAUDE_OAUTH_TOKEN": "stub-token",
        "AGENT_CONFIG": "claude-sonnet-4-6",
        # bash 3.2 (macOS) treats an empty array as unbound under `set -u`; a set effort keeps
        # solve.sh's effort array non-empty there. Upstream's own default is xhigh anyway.
        "AGENT_EFFORT": "xhigh",
        "AGENT_RESUME": "true" if resume else "false",
        "TIME_BUDGET_HOURS": hours,
    }
    assert bash_allows_empty_arrays_under_nounset() or env["AGENT_EFFORT"] != "default"
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        cwd=tmp,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = []
    n = 1
    while (log / f"call{n}").is_dir():
        d = log / f"call{n}"
        argv = [a.decode("utf-8") for a in (d / "argv").read_bytes().split(b"\0")[:-1]]
        calls.append(
            UpstreamCall(
                argv=argv,
                stdin=(d / "stdin").read_text("utf-8"),
                limit=int((d / "limit").read_text("utf-8")),
            )
        )
        n += 1
    return UpstreamLoop(
        calls=calls,
        exit_code=result.returncode,
        stderr=result.stderr.splitlines(),
        end_clock=int((tmp / "clock").read_text("utf-8")),
    )
