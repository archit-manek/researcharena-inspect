"""Stopping the agent when its time is up, as upstream's container teardown does.

At the pinned commit the agent runs in its own Apptainer container (`--containall`,
`src/pipeline_redteam.sh:357-378`). When `solve.sh` exits, the container's PID namespace goes
with it, so nothing the agent started (background jobs, servers, benchmarks) outlives the
agent; `free_gpu_memory` (`:61-64`, `:415`) then kills any process still holding the GPU. The
scorer runs in a fresh container on a copy of the workspace (`:438`, `:478-500`).

In Inspect the scorer runs in the same sandbox, so the port does the teardown explicitly
(D-081): it lists the sandbox's processes before the agent starts and, after the last round,
kills every process that is new, except Inspect's own sandbox tools and the shell doing the
killing. It reads `/proc` and uses only bash and coreutils, so it works in images without `ps`
or `pkill`. It is needed: cancelling a round leaves Inspect's `bash` tool commands and, for a
moment, the Claude Code CLI running in the container.
"""

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
