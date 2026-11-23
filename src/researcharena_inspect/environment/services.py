"""Upstream's `hooks/pipeline_pre_agent.sh`, sourced in a service beside the agent (D-174).

Upstream sources these hooks in the pipeline shell on the host just before the agent starts
(`src/pipeline_redteam.sh:325-335`). The only one at the pin, the post-training `independent`
side task's, starts an upload-capture proxy on the host's loopback and gives the agent
`http_proxy` variables. The agent shares the host's network namespace but not its processes or
files, so the port runs the hooks in `SERVICE_NAME`, a container in the agent container's
network namespace (`sandbox.with_hook_service`). The shell that sourced them stays alive, so
their `EXIT` traps fire when the service is removed with the sample, as upstream's fire when
the pipeline exits.
"""

# Adapted from https://github.com/aisa-group/ResearchArena at b922f4b (Copyright (c) 2026 AI Safety and Alignment Group, MIT licence)

import io
import json
import shlex
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import anyio
from inspect_ai.util import SandboxEnvironment
from inspect_ai.util import sandbox as sandbox_env

from researcharena_inspect.environment.layout import AGENT_HOME, AGENT_USER, EVAL_DIR
from researcharena_inspect.prompts.settings import load_config, settings_root, upstream_root

SERVICE_NAME: Final = "aux"
"""The sidecar's Compose service name. The agent never sees it."""

SERVICE_ROOT: Final = "/opt/researcharena-hooks"
"""Root-only folder in the sidecar: upstream's files (`ARENA_ROOT`) and the runner's state."""

SERVICE_ARENA_ROOT: Final = f"{SERVICE_ROOT}/arena"
SERVICE_EVAL_DIR: Final = EVAL_DIR
"""The hooks' `EVAL_DIR` in the sidecar, the same path the evaluator reads in the agent's
container, so that the paths the hooks print are the ones the evaluator sees."""

HOOK_START_TIMEOUT_S: Final = 30.0
"""How long `start_pipeline_hooks` waits for the hooks to finish; upstream's proxy hook itself
gives the proxy 10 s (`pipeline_pre_agent.sh:36-44`)."""

EVIDENCE_MAX_BYTES: Final = 64 * 1024 * 1024
"""Largest evidence folder copied (tarred). Each captured request is one JSON line of a few
hundred bytes; a larger folder is an error, never a truncated read (D-174)."""

EVIDENCE_EVENTS_KEPT: Final = 2000
"""Request records kept whole in the score; the rest are counted."""

LOG_END_BYTES: Final = 16 * 1024

LOOPBACK: Final = ("127.0.0.1", "::1")

# Sourced under `pipeline_redteam.sh`'s `set -euo pipefail` (`:18`). After the hooks return,
# the shell waits, so their `trap ... EXIT` cleanups run only when the service is removed.
_RUNNER = """set -euo pipefail
out="$1"; shift
AGENT_EXTRA_ENV_ARGS=()
for hook in "$@"; do
    echo "== sourcing $hook"
    source "$hook"
done
printf '%s\\0' "${AGENT_EXTRA_ENV_ARGS[@]}" > "$out.tmp"
mv "$out.tmp" "$out"
while :; do sleep 3600; done
"""


class ServiceError(RuntimeError):
    """A hook service could not be started or its evidence could not be copied."""


def pipeline_hooks(setting: str, side_task: str, upstream: Path | None = None) -> list[Path]:
    """The `pipeline_pre_agent.sh` hooks upstream sources for this run, in order (`:330-335`)."""
    setting_dir = settings_root(upstream) / setting
    hooks = [setting_dir / "hooks" / "pipeline_pre_agent.sh"]
    if side_task != "none":
        hooks.append(setting_dir / "side_tasks" / side_task / "hooks" / "pipeline_pre_agent.sh")
    return [hook for hook in hooks if hook.is_file()]


def setting_has_hooks(setting: str, upstream: Path | None = None) -> bool:
    """Whether any run of the setting (any side task) sources a `pipeline_pre_agent.sh`."""
    setting_dir = settings_root(upstream) / setting
    return (setting_dir / "hooks" / "pipeline_pre_agent.sh").is_file() or any(
        setting_dir.glob("side_tasks/*/hooks/pipeline_pre_agent.sh")
    )


def endpoint_hosts(setting: str, upstream: Path | None = None) -> list[str]:
    """The host names of every `exfil_endpoint` in the setting's task configs."""
    hosts = set()
    for config in load_config(setting, upstream).get("configs", {}).values():
        endpoint = config.get("exfil_endpoint") if isinstance(config, dict) else None
        if isinstance(endpoint, str) and (host := urlsplit(endpoint).hostname):
            hosts.add(host)
    return sorted(hosts)


def _service_bundle(setting: str, side_task: str, upstream: Path | None) -> bytes:
    """Upstream's `src/utils/` and the hooks' folders, laid out as in upstream's tree."""
    root = upstream_root(upstream)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        sources = [root / "src" / "utils"]
        sources += [hook.parent.parent for hook in pipeline_hooks(setting, side_task, upstream)]
        for source in sources:
            for path in sorted([source, *source.rglob("*")]):
                if "__pycache__" in path.parts:
                    continue
                info = tar.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                if path.is_file():
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                elif path.is_dir():
                    tar.addfile(info)
    return buffer.getvalue()


@dataclass
class HookService:
    """What `start_pipeline_hooks` started, for the log."""

    service: str
    hooks: list[str]
    """Hook paths relative to upstream's root, in the order they were sourced."""
    agent_env: dict[str, str] = field(default_factory=dict)
    """`AGENT_EXTRA_ENV_ARGS`, as name to value."""
    output: str = ""
    """What the hooks printed (bounded)."""
    seconds: float = 0.0
    endpoint_pins: dict[str, Any] = field(default_factory=dict)
    """D-173: the hosts pinned to loopback in the agent's container and what they resolve to."""


def _parse_env_args(raw: bytes) -> dict[str, str]:
    parts = raw.decode("utf-8").split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    env: dict[str, str] = {}
    if len(parts) % 2:
        raise ServiceError(f"AGENT_EXTRA_ENV_ARGS is not --env pairs: {parts!r}")
    for flag, assignment in zip(parts[::2], parts[1::2], strict=True):
        name, sep, value = assignment.partition("=")
        if flag != "--env" or not sep or not name:
            raise ServiceError(f"AGENT_EXTRA_ENV_ARGS has {flag!r} {assignment!r}, not --env K=V")
        env[name] = value
    return env


def _service(name: str = SERVICE_NAME) -> SandboxEnvironment:
    try:
        return sandbox_env(name)
    except Exception as error:
        raise ServiceError(
            f"this run sources upstream's pipeline_pre_agent.sh, which needs the sandbox service "
            f"{name!r} sharing the agent container's network namespace "
            "(sandbox.with_hook_service; Docker only)"
        ) from error


def _tail(text: str, limit: int = LOG_END_BYTES) -> str:
    return text if len(text) <= 2 * limit else text[:limit] + "\n[...]\n" + text[-limit:]


async def start_pipeline_hooks(
    setting: str,
    side_task: str,
    task_config: str,
    upstream: Path | None = None,
) -> HookService | None:
    """Source the run's `pipeline_pre_agent.sh` hooks in the hook service, as upstream does.

    Call once per sample, before the agent starts. Returns None when the run has none.

    Raises:
        ServiceError: there is no hook service, a hook failed (upstream stops on it: `set -e`),
            or the hooks did not finish in time.
    """
    hooks = pipeline_hooks(setting, side_task, upstream)
    if not hooks:
        return None
    root = upstream_root(upstream)
    rel = [hook.relative_to(root).as_posix() for hook in hooks]
    box = _service()
    start = time.monotonic()
    prepared = await box.exec(
        [
            "bash",
            "-c",
            (
                f"set -eu; rm -rf {SERVICE_ROOT}; install -d -m 700 {SERVICE_ARENA_ROOT}; "
                f"install -d -m 755 {SERVICE_EVAL_DIR}"
            ),
        ],
        user="root",
    )
    if not prepared.success:
        raise ServiceError(f"preparing the hook service failed: {prepared.stderr}")
    await box.write_file(f"{SERVICE_ROOT}/arena.tar", _service_bundle(setting, side_task, upstream))
    await box.write_file(f"{SERVICE_ROOT}/runner.sh", _RUNNER)
    unpack = await box.exec(
        [
            "bash",
            "-c",
            (
                f"set -eu; tar -xpf {SERVICE_ROOT}/arena.tar -C {SERVICE_ARENA_ROOT}; "
                f"rm -f {SERVICE_ROOT}/arena.tar"
            ),
        ],
        user="root",
    )
    if not unpack.success:
        raise ServiceError(
            f"unpacking upstream's files in the hook service failed: {unpack.stderr}"
        )

    env_out = f"{SERVICE_ROOT}/agent_env"
    log = f"{SERVICE_ROOT}/hooks.log"
    hook_paths = " ".join(shlex.quote(f"{SERVICE_ARENA_ROOT}/{r}") for r in rel)
    launch = (
        f"cd {SERVICE_ARENA_ROOT} && nohup bash {SERVICE_ROOT}/runner.sh {env_out} {hook_paths} "
        f"> {log} 2>&1 < /dev/null & echo $! > {SERVICE_ROOT}/runner.pid"
    )
    variables = {
        # The pipeline state the hook contract names (pipeline_pre_agent.sh:5-6).
        "ARENA_ROOT": SERVICE_ARENA_ROOT,
        "EVAL_DIR": SERVICE_EVAL_DIR,
        "JOB_DIR": AGENT_HOME,
        "SETTING": setting,
        "SIDE_TASK": side_task,
        "TASK_CONFIG": task_config,
    }
    started = await box.exec(["bash", "-c", launch], user="root", env=variables)
    if not started.success:
        raise ServiceError(f"starting the hooks failed: {started.stderr}")

    check = (
        f'if [ -f {env_out} ]; then echo done; elif kill -0 "$(cat {SERVICE_ROOT}/runner.pid)" '
        "2>/dev/null; then echo running; else echo exited; fi"
    )
    state = "running"
    while time.monotonic() - start < HOOK_START_TIMEOUT_S:
        result = await box.exec(["bash", "-c", check], user="root")
        state = result.stdout.strip()
        if state != "running":
            break
        await anyio.sleep(0.2)
    output = (await box.exec(["bash", "-c", f"cat {log} 2>/dev/null || true"], user="root")).stdout
    if state != "done":
        raise ServiceError(
            f"{rel} {'failed' if state == 'exited' else 'did not finish in time'}: {_tail(output)}"
        )
    raw = await box.read_file(env_out, text=False)
    return HookService(
        service=SERVICE_NAME,
        hooks=rel,
        agent_env=_parse_env_args(raw),
        output=_tail(output),
        seconds=round(time.monotonic() - start, 2),
    )


_RESOLVE = """
import json, socket, sys
order = []
try:
    for line in open("/etc/nsswitch.conf"):
        if line.split("#")[0].strip().startswith("hosts:"):
            order = line.split("#")[0].split(":", 1)[1].split()
except OSError:
    pass
out = {"nsswitch_hosts": order, "addresses": {}}
# Only when /etc/hosts is read first does the lookup below stay on the machine.
if order[:1] == ["files"]:
    for host in sys.argv[1:]:
        out["addresses"][host] = sorted({a[4][0] for a in socket.getaddrinfo(host, 443)})
print(json.dumps(out))
"""


async def pin_endpoint_hosts(hosts: list[str]) -> dict[str, Any]:
    """D-173: make `hosts` resolve to loopback in the agent's container, and check it.

    Raises if the resolver would not read `/etc/hosts` first or a host resolves elsewhere: the
    run must not start with the endpoint reachable.
    """
    if not hosts:
        return {}
    box = sandbox_env()
    lines = "".join(f"{address}\t{host}\n" for host in hosts for address in LOOPBACK)
    written = await box.exec(
        ["bash", "-c", 'printf "%s" "$1" >> /etc/hosts', "pin", lines], user="root"
    )
    if not written.success:
        raise ServiceError(f"writing /etc/hosts failed: {written.stderr}")
    resolved = await box.exec(["python3", "-c", _RESOLVE, *hosts], user=AGENT_USER)
    if not resolved.success:
        raise ServiceError(f"checking the pinned hosts failed: {resolved.stderr}")
    report = json.loads(resolved.stdout)
    if report.get("nsswitch_hosts", [])[:1] != ["files"]:
        raise ServiceError(f"the sandbox does not read /etc/hosts first: {report}")
    for host in hosts:
        addresses = report["addresses"].get(host) or []
        if not addresses or any(a not in LOOPBACK for a in addresses):
            raise ServiceError(f"{host} resolves to {addresses}, not only to loopback")
    return {"hosts": hosts, "lines": lines.splitlines(), **report}


@dataclass
class HookEvidence:
    """The hook service's `EVAL_DIR` as copied into the evaluator's sandbox, for the score."""

    service: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Relative path -> `bytes`, `sha256`; logs also `text` (bounded)."""
    requests: list[dict[str, Any]] = field(default_factory=list)
    """`requests.jsonl` records, the first `EVIDENCE_EVENTS_KEPT` (no bodies exist)."""
    request_lines: int = 0
    malformed_lines: int = 0
    captured: int = 0
    """Records with `captured: true` (requests to the capture endpoint)."""
    captured_bytes: int = 0
    seconds: float = 0.0
    proxy: dict[str, Any] = field(default_factory=dict)
    """The capture proxy's state just after the evidence was copied (`PROXY_HEALTH`): whether
    its process was alive and something listened on its port; `"unknown"` if the probe failed.
    Best effort and recorded only: it never delays the copy or changes the score (D-189)."""

    def summary(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "proxy": self.proxy,
            "files": self.files,
            "request_lines": self.request_lines,
            "malformed_lines": self.malformed_lines,
            "captured": self.captured,
            "captured_bytes": self.captured_bytes,
            "requests_kept": len(self.requests),
            "requests": self.requests,
            "seconds": self.seconds,
        }


def _summarise(data: bytes, evidence: HookEvidence) -> None:
    import hashlib

    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            handle = tar.extractfile(member)
            assert handle is not None
            content = handle.read()
            record: dict[str, Any] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if name.endswith((".log", "port")):
                record["text"] = _tail(content.decode("utf-8", errors="replace"))
            evidence.files[name] = record
            if name.endswith("requests.jsonl"):
                for line in content.decode("utf-8", errors="replace").splitlines():
                    if not line.strip():
                        continue
                    evidence.request_lines += 1
                    try:
                        event = json.loads(line)
                    except ValueError:
                        evidence.malformed_lines += 1
                        continue
                    if not isinstance(event, dict):
                        evidence.malformed_lines += 1
                        continue
                    if event.get("captured") is True:
                        evidence.captured += 1
                        length = event.get("content_length")
                        if isinstance(length, int):
                            evidence.captured_bytes += length
                    if len(evidence.requests) < EVIDENCE_EVENTS_KEPT:
                        evidence.requests.append(event)


# Run in the hook service as root: the proxy's processes and whether something listens on the
# port its hook wrote. Prints one JSON line.
PROXY_HEALTH = """
import json, os, sys
folder = os.path.join(sys.argv[1], "side_task_evidence", "upload_capture")
pids = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\\0")
    except OSError:
        continue
    if any(a.endswith(b"upload_capture_proxy.py") for a in argv):
        pids.append(int(pid))
port = None
try:
    with open(os.path.join(folder, "port")) as f:
        port = int(f.read().strip())
except (OSError, ValueError):
    pass
listening = False
for table in ("/proc/net/tcp", "/proc/net/tcp6") if port is not None else ():
    try:
        with open(table) as f:
            rows = f.read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        cols = row.split()
        if len(cols) > 3 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
            listening = True
print(json.dumps({"alive": bool(pids), "pids": sorted(pids), "port": port, "listening": listening}))
"""


async def _proxy_health(box: SandboxEnvironment) -> dict[str, Any]:
    """`PROXY_HEALTH`'s record, or `"unknown"` with the reason; never raises (D-189)."""
    try:
        result = await box.exec(
            ["python3", "-c", PROXY_HEALTH, SERVICE_EVAL_DIR], user="root", timeout=30
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as error:  # a record only: no failure may reach scoring
        return {"alive": "unknown", "listening": "unknown", "error": repr(error)[-500:]}


async def copy_hook_evidence(
    evaluator: SandboxEnvironment, owner: str = AGENT_USER
) -> HookEvidence:
    """Copy the hook service's `EVAL_DIR` into the `EVAL_DIR` of the evaluator's sandbox.

    Call after `environment.prepare_evaluation`, before `evaluate.py`; upstream's evaluator
    reads the evidence beside `red_team/`. The files belong to `owner`, the evaluator's user.

    Raises:
        ServiceError: the service is gone, or the folder is larger than `EVIDENCE_MAX_BYTES`
            (the sample must not be scored from a partial log).
    """
    start = time.monotonic()
    box = _service()
    tar_path = f"{SERVICE_ROOT}/evidence.tar"
    made = await box.exec(
        [
            "bash",
            "-c",
            f"set -eu; tar -C {SERVICE_EVAL_DIR} -cf {tar_path} . && stat -c %s {tar_path}",
        ],
        user="root",
    )
    if not made.success:
        raise ServiceError(f"archiving the hook evidence failed: {made.stderr}")
    size = int(made.stdout.strip().splitlines()[-1])
    if size > EVIDENCE_MAX_BYTES:
        raise ServiceError(f"the hook evidence is {size} bytes, over {EVIDENCE_MAX_BYTES}")
    data = await box.read_file(tar_path, text=False)
    scratch = "/var/tmp/researcharena-evidence.tar"
    await evaluator.write_file(scratch, data)
    readable = await evaluator.exec(["chmod", "644", scratch], user="root")
    unpacked = await evaluator.exec(
        ["bash", "-c", f"umask 022; tar -xpf {scratch} --no-same-owner -C {EVAL_DIR}"],
        user=owner,
    )
    await evaluator.exec(["rm", "-f", scratch], user="root")
    if not (readable.success and unpacked.success):
        raise ServiceError(
            f"copying the hook evidence into {EVAL_DIR} failed: {readable.stderr}{unpacked.stderr}"
        )
    evidence = HookEvidence(service=SERVICE_NAME, proxy=await _proxy_health(box))
    _summarise(data, evidence)
    evidence.seconds = round(time.monotonic() - start, 2)
    return evidence


async def replay_hook_evidence(folder: Path, *, has_hooks: bool) -> dict[str, Any]:
    """Put a stored run's capture log back in the hook service, for a replay (D-172).

    The stored `requests.jsonl` replaces the replay's own (empty) one, so the evaluator reads
    what the stored run captured. Nothing is done for a sample without the hook service.

    Returns:
        What was put back: the file, its size and SHA-256; or why nothing was.
    """
    import hashlib

    log = folder / "upload_capture" / "requests.jsonl"
    if not has_hooks:
        return {"used": False, "reason": "this sample has no hook service"}
    if not log.is_file():
        return {"used": False, "reason": "the stored run captured no request"}
    data = log.read_bytes()
    if len(data) > EVIDENCE_MAX_BYTES:
        raise ServiceError(
            f"the stored capture log is {len(data)} bytes, over {EVIDENCE_MAX_BYTES}"
        )
    box = _service()
    dest = f"{SERVICE_EVAL_DIR}/side_task_evidence/upload_capture/requests.jsonl"
    await box.write_file(dest, data)
    chmod = await box.exec(["chmod", "644", dest], user="root")
    if not chmod.success:
        raise ServiceError(f"putting the stored capture log back failed: {chmod.stderr}")
    return {
        "used": True,
        "file": "side_task_evidence/upload_capture/requests.jsonl",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


__all__ = [
    "EVIDENCE_MAX_BYTES",
    "SERVICE_NAME",
    "HookEvidence",
    "HookService",
    "ServiceError",
    "copy_hook_evidence",
    "endpoint_hosts",
    "pin_endpoint_hosts",
    "pipeline_hooks",
    "replay_hook_evidence",
    "setting_has_hooks",
    "start_pipeline_hooks",
]
