"""Stopping the agent when its time is up, as upstream's container teardown does.

At the pinned commit the agent runs in its own Apptainer container (`--containall`,
`src/pipeline_redteam.sh:357-378`). When `solve.sh` exits, the container's PID namespace goes
with it, so nothing the agent started (background jobs, servers, benchmarks) outlives the
agent; `free_gpu_memory` (`:61-64`, `:415`) then kills any process still holding the GPU. The
scorer runs in a fresh container on a copy of the workspace (`:438`, `:478-500`).

In Inspect the agent's container outlives the solver: on Docker the workspace is cleaned and
copied in it before it is stopped (D-251), and with a sandbox that has no `evaluator`
service (a custom `sandbox`), the scorer runs in it (D-093). So the port does the teardown explicitly
(D-081): it lists the sandbox's processes before the agent starts and, after the last round,
kills every process that is new, except Inspect's own sandbox tools and the shell doing the
killing. It reads `/proc` and uses only bash and coreutils, so it works in images without `ps`
or `pkill`. It is needed: cancelling a round leaves Inspect's `bash` tool commands and, for a
moment, the Claude Code CLI running in the container.

`signal_cli` is the per-round stop of `timeout --signal=TERM --kill-after=30s` (`solve.sh:97`):
it signals the round's Claude Code CLI and the processes of its process group, the ones
`timeout` reaches (see `signal_cli`).
"""

from typing import Literal

from inspect_ai.util import sandbox

INSPECT_TOOLS_MARKER = "inspect-sandbox-tools"
"""Part of the command line of Inspect's in-sandbox server and model proxy, which stay."""

_LIST_PIDS = "cd /proc && for d in [0-9]*; do printf '%s ' \"$d\"; done"

# $1: PIDs to keep, space-separated. Prints "<pid> <command line>" for each process killed.
_KILL_NEW = r"""
keep=" $1 "
p=$$
while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
  keep="$keep$p "
  stat=$(cat "/proc/$p/stat" 2>/dev/null) || break
  rest=${stat##*) }
  set -- $rest
  p=$2
done
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$keep" in *" $pid "*) continue ;; esac
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
  case "$cmd" in *__MARKER__*) continue ;; esac
  if kill -9 "$pid" 2>/dev/null; then printf '%s %s\n' "$pid" "$cmd"; fi
done
""".replace("__MARKER__", INSPECT_TOOLS_MARKER)

# $1: signal (TERM or KILL); $2: PIDs to keep, space-separated. Prints "<pid> <command line>"
# (newlines flattened, first 300 characters) for each process signalled. The CLI is a new process (not in $2, not Inspect's) whose argv[0]
# is `claude` (or `claude-<version>`) and whose flags before `--` include `--print` and
# `--output-format stream-json`,
# as Inspect SWE starts it; one with such a CLI among its ancestors (a `claude` the agent ran
# itself) is not the round's CLI. The CLI's descendants in its process group are signalled too.
_SIGNAL_CLI = r"""
sig=$1
keep=" $2 "
declare -A parent group
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  stat=$(cat "$d/stat" 2>/dev/null) || continue
  rest=${stat##*) }
  set -- $rest
  parent[$pid]=$2
  group[$pid]=$3
done
is_new() {
  case "$keep" in *" $1 "*) return 1 ;; esac
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null) || return 1
  case "$cmd" in *__MARKER__*) return 1 ;; esac
}
is_cli() {
  local -a argv
  local a prev="" printed="" json=""
  mapfile -d '' argv < "/proc/$1/cmdline" 2>/dev/null || return 1
  [ "${#argv[@]}" -gt 0 ] || return 1
  case "${argv[0]##*/}" in claude | claude-*) ;; *) return 1 ;; esac
  for a in "${argv[@]:1}"; do
    [ "$a" = -- ] && break
    [ "$a" = --print ] && printed=1
    if [ "$a" = --output-format=stream-json ] ||
      { [ "$prev" = --output-format ] && [ "$a" = stream-json ]; }; then
      json=1
    fi
    prev=$a
  done
  [ -n "$printed" ] && [ -n "$json" ]
}
# under PID ANCESTOR: whether ANCESTOR is PID or one of PID's ancestors
under() {
  local p=$1
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    [ "$p" = "$2" ] && return 0
    p=${parent[$p]}
  done
  return 1
}
clis=" "
for pid in "${!parent[@]}"; do
  if is_new "$pid" && is_cli "$pid"; then clis="$clis$pid "; fi
done
targets=""
for c in $clis; do
  p=${parent[$c]}
  nested=""
  while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
    case "$clis" in *" $p "*) nested=1; break ;; esac
    p=${parent[$p]}
  done
  [ -n "$nested" ] && continue
  targets="$targets $c"
  for pid in "${!parent[@]}"; do
    [ "$pid" != "$c" ] && [ "${group[$pid]}" = "${group[$c]}" ] || continue
    if is_new "$pid" && under "$pid" "$c"; then targets="$targets $pid"; fi
  done
done
for pid in $targets; do
  cmd=$(tr '\0\n' '  ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-300)
  if kill -s "$sig" "$pid" 2>/dev/null; then printf '%s %s\n' "$pid" "$cmd"; fi
done
# timeout sends SIGCONT after SIGTERM, so a stopped process can act on it.
if [ "$sig" = TERM ]; then
  for pid in $targets; do kill -s CONT "$pid" 2>/dev/null; done
fi
exit 0
""".replace("__MARKER__", INSPECT_TOOLS_MARKER)

SIGNAL_TIMEOUT_S = 30
"""Bound on one `signal_cli` call, so an unresponsive sandbox cannot hold up the round."""


async def list_processes(sandbox_name: str | None = None) -> list[int]:
    """PIDs of the processes running in the sandbox now."""
    result = await sandbox(sandbox_name).exec(["bash", "-c", _LIST_PIDS], user="root")
    if not result.success:
        raise RuntimeError(f"could not list the sandbox's processes: {result.stderr}")
    return sorted(int(pid) for pid in result.stdout.split())


async def kill_new_processes(keep: list[int], sandbox_name: str | None = None) -> list[str]:
    """Kill (SIGKILL) every process not in `keep`, other than Inspect's own.

    Returns:
        One "<pid> <command line>" entry per process killed.
    """
    pids = " ".join(str(p) for p in keep)
    result = await sandbox(sandbox_name).exec(
        ["bash", "-c", _KILL_NEW, "kill_new", pids], user="root"
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


async def signal_cli(
    signal: Literal["TERM", "KILL"], keep: list[int], sandbox_name: str | None = None
) -> list[str]:
    """Send `signal` to the round's Claude Code CLI and its process group, as `timeout` does.

    GNU `timeout` (without `--foreground`) runs the command in a new process group and, at the
    limit, signals the command and that group, then sends SIGCONT after SIGTERM; after
    `--kill-after` it sends SIGKILL the same way. The group is the CLI and its descendants that
    stayed in it. Claude Code starts its Bash tool's shell in a new session (measured with
    2.1.162), so `timeout` does not reach the tool's commands: the CLI deals with them itself.
    Here the CLI's group also holds the shell Inspect started it with, which `timeout` has no
    counterpart of, so only the CLI's own descendants in its group are signalled.

    Args:
        signal: `TERM` at the deadline, `KILL` after the grace.
        keep: PIDs running before the agent started (`list_processes`); the CLI is new.
        sandbox_name: the sandbox; the default one if omitted.

    Returns:
        One "<pid> <command line>" entry per process signalled; empty if no CLI was running.
    """
    pids = " ".join(str(p) for p in keep)
    result = await sandbox(sandbox_name).exec(
        ["bash", "-c", _SIGNAL_CLI, "signal_cli", signal, pids],
        user="root",
        timeout=SIGNAL_TIMEOUT_S,
    )
    if not result.success:
        raise RuntimeError(f"could not signal the CLI ({signal}): {result.stderr}")
    return [line for line in result.stdout.splitlines() if line.strip()]
