"""Run upstream's `src/pipeline_redteam.sh` for real, as the oracle for workspace staging.

The script runs unchanged from a clean export of the pinned commit, in a Debian container
with git (`python:3.12-bookworm`). With no Apptainer image it takes its host branch
(`pipeline_redteam.sh:382-401`): it stages the agent's home exactly as for a container run,
then runs the staged `agent_solve.sh` with the pipeline's own `HOME`. Upstream's tree is not
edited; what stands in for the parts that need a cluster lives outside it:

- `$HOME/update_agent_cli.sh`, which every `solve.sh` calls before it starts its CLI, records
  the agent's home at that moment (the workspace the agent starts with), applies the test's
  stand-in for the agent's work, and exits 1, so the agent "finishes" there.
- A `python3` wrapper that, when the pipeline runs `evaluate.py`, records the evaluator's job
  folder and the submitted `task_snapshot`, then writes a placeholder `metrics.json` so the
  pipeline does not retry. Every other `python3` call goes to the real interpreter.
- No-op `nvidia-smi`, `fuse-overlayfs`, `fusermount` and `sleep`, and a `date` that answers
  `date +%s` with a fixed time, so `timer.sh` is reproducible.

With `real_evaluate=True` the `python3` wrapper is left out, so the pipeline runs upstream's own
`evaluate.py` through Phase 4 (its attempt loop and failure record), and the run's `eval/`
folder (`metrics.json`, `eval_attempt_<n>.log`) is copied out, and `red_team/task_snapshot` is
described as Phase 4 leaves it. The stand-in agent's work can be
chosen per run (`agent_work`).

The git identity of the hooks' commit is passed as `GIT_*` variables: upstream's hooks use the
cluster account's identity, which is not known (a faithful reading, not executed upstream).
"""

import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from researcharena_inspect.environment.staging import GIT_IDENTITY
from researcharena_inspect.pins import UPSTREAM_COMMIT
from researcharena_inspect.upstream import upstream_dir

IMAGE = "python:3.12-bookworm"
FIXED_EPOCH = 1_700_000_000
"""What `date +%s` answers in both containers, so `timer.sh` is the same file."""

MANIFEST_SOURCE = Path(__file__).with_name("tree_manifest.py")

DATE_SHIM = f"""#!/bin/sh
if [ "$#" -eq 1 ] && [ "$1" = "+%s" ]; then echo {FIXED_EPOCH}; else exec /usr/bin/date "$@"; fi
"""

# The test's stand-in for an agent's work, run in task/ as the agent's user, on both sides.
AGENT_WORK = """#!/bin/sh
set -eu
printf '// tuned\\n' >> kernels/rmsnorm.cu
mkdir -p workspace
printf '1.2345\\n' > workspace/baseline_tps.txt
chmod 600 workspace/baseline_tps.txt
mkdir -p scratch_model final_model/sub
printf '{}' > scratch_model/config.json
printf 'weights' > scratch_model/model.safetensors
printf '{}' > final_model/config.json
printf 'weights' > final_model/model.safetensors
printf '{}' > final_model/sub/config.json
printf 'weights' > final_model/sub/model.safetensors
ln -s kernels/rmsnorm.cu kernel_link
printf 'note\\n' > .notes
git add kernels/rmsnorm.cu
GIT_AUTHOR_NAME=agent GIT_AUTHOR_EMAIL=agent@example.invalid \\
    GIT_COMMITTER_NAME=agent GIT_COMMITTER_EMAIL=agent@example.invalid git commit -q -m "agent edit"
"""

UPDATE_CLI_STANDIN = """#!/usr/bin/env bash
# Stand-in for the CLI updater: record the agent's starting home, do the test's work, stop.
set -euo pipefail
JOB_DIR="$(cd .. && pwd)"
python3 /work/tree_manifest.py "$JOB_DIR" > /capture/home.json
pwd > /capture/agent_cwd.txt
sh "${AGENT_WORK_SCRIPT:-/work/agent_work.sh}"
exit 1
"""

PYTHON_WRAPPER = """#!/bin/sh
# python3 wrapper: record the evaluator's view the first time evaluate.py runs.
if [ "${1:-}" = "evaluate.py" ] && [ ! -e /capture/eval_home.json ]; then
    submission=""; output=""; prev=""
    for arg in "$@"; do
        case "$prev" in --submission-dir) submission="$arg" ;; --output) output="$arg" ;; esac
        prev="$arg"
    done
    /usr/local/bin/python3 /work/tree_manifest.py "$PWD" > /capture/eval_home.json
    /usr/local/bin/python3 /work/tree_manifest.py "$submission/task_snapshot" > /capture/snapshot.json
    /usr/local/bin/python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@" > /capture/eval_argv.json
    printf '{"main_task_score": 0}' > "$output"
    exit 0
fi
exec /usr/local/bin/python3 "$@"
"""

NOOP = "#!/bin/sh\nexit 0\n"


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


def export_arena(dest: Path, commit: str = UPSTREAM_COMMIT) -> Path:
    """A clean copy of upstream's tree at `commit`, unchanged."""
    archive = subprocess.run(
        ["git", "-C", str(upstream_dir(commit)), "archive", "--format=tar", commit],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(dest, filter="tar")
    return dest


def _write(path: Path, text: str, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def prepare_work(work: Path) -> None:
    """The stand-ins, shared by every run in `work`."""
    shims = work / "shims"
    real = work / "shims_real"  # the same, without the python3 wrapper
    for folder in (shims, real):
        _write(folder / "date", DATE_SHIM)
        for name in ("nvidia-smi", "fuse-overlayfs", "fusermount", "sleep"):
            _write(folder / name, NOOP)
    _write(shims / "python3", PYTHON_WRAPPER)
    _write(work / "agent_work.sh", AGENT_WORK)
    shutil.copy(MANIFEST_SOURCE, work / "tree_manifest.py")
    (work / "hf").mkdir(exist_ok=True)


@dataclass
class PipelineRun:
    home: dict
    """The agent's home when its CLI would start (tree_manifest output)."""
    agent_cwd: str
    eval_home: dict | None
    snapshot: dict | None
    eval_argv: list[str] | None
    returncode: int
    log: str
    eval_files: dict[str, bytes] | None = None
    """With `real_evaluate`: every file of the run's `eval/` folder, by name."""
    final_snapshot: dict | None = None
    """With `real_evaluate`: `red_team/task_snapshot` after Phase 4 (tree_manifest output),
    the folder the run keeps and an artifact monitor is given."""
    artifact_files: dict[str, bytes] | None = None
    """With `real_evaluate`: the run's `artifacts/` folder (the inference evaluator's), by name."""
    evidence_files: dict[str, bytes] | None = None
    """With `real_evaluate`: the run's `side_task_evidence/` folder (the upload capture's), by
    path relative to it."""


def run_pipeline(
    work: Path,
    arena: Path,
    *,
    side_task: str,
    agent: str,
    redruntime: str = "1",
    agent_work: str | None = None,
    real_evaluate: bool = False,
    name: str | None = None,
    setting: str = "cuda_kernel",
    extra_env: dict[str, str] | None = None,
    network: str | None = None,
) -> PipelineRun:
    """`pipeline_redteam.sh <setting> <side_task> <redruntime> <agent> test-model 1 test`.

    Args:
        agent_work: the stand-in agent's work (a sh script run in `task/`); `AGENT_WORK` if None.
        real_evaluate: run upstream's `evaluate.py` and keep the `eval/` folder.
        name: the run folder's name under `work` (default `run_<side_task>_<agent>`).
        setting: the setting (`cuda_kernel` by default).
        extra_env: more variables for the pipeline's environment (inherited by `evaluate.py`).
        network: `docker run --network` (for example `none`); Docker's default if None.
    """
    run_dir = work / (name or f"run_{side_task}_{agent}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    capture = run_dir / "capture"
    home = run_dir / "home"
    capture.mkdir(parents=True)
    _write(home / "update_agent_cli.sh", UPDATE_CLI_STANDIN)
    if agent_work is not None:
        _write(run_dir / "agent_work.sh", agent_work)

    def inside(p: Path) -> str:
        return "/work/" + p.relative_to(work).as_posix()

    env = {
        "HOME": inside(home),
        "ARENA_ROOT": inside(arena),
        "ARENA_CONTAINER_SIF": "/nonexistent/arena.sif",
        # Inside the container: a macOS bind mount would give symlinks mode 755, not 777.
        "ARENA_RESULTS_DIR": "/tmp/results",
        "HF_HOME": "/work/hf",
        "PATH": (
            f"/work/{'shims_real' if real_evaluate else 'shims'}:"
            "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "CLAUDE_OAUTH_TOKEN": "placeholder-not-a-token",
        **GIT_IDENTITY,
    }
    if agent_work is not None:
        env["AGENT_WORK_SCRIPT"] = inside(run_dir / "agent_work.sh")
    env.update(extra_env or {})
    cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
    if network is not None:
        cmd += ["--network", network]
    cmd += ["-v", f"{work}:/work", "-v", f"{capture}:/capture"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    pipeline = [f"{inside(arena)}/src/pipeline_redteam.sh"]
    pipeline += [setting, side_task, redruntime, agent, "test-model", "1", "test"]
    if real_evaluate:
        # The pipeline unchanged, then its eval/ folder copied out (the results stay in the
        # container's /tmp, as in the staging test).
        keep_eval = (
            'bash "$@"; rc=$?; '
            'for d in /tmp/results/*/*/eval; do cp -r "$d" /capture/eval_out; done; '
            'for d in /tmp/results/*/*/artifacts; do cp -r "$d" /capture/artifacts_out; done; '
            "for d in /tmp/results/*/*/side_task_evidence; do "
            'cp -r "$d" /capture/evidence_out; done; '
            "for d in /tmp/results/*/*/red_team/task_snapshot; do "
            'python3 /work/tree_manifest.py "$d" > /capture/final_snapshot.json; done; '
            "exit $rc"
        )
        cmd += [IMAGE, "bash", "-c", keep_eval, "pipeline", *pipeline]
    else:
        cmd += [IMAGE, "bash", *pipeline]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    def load(name: str):
        path = capture / name
        return json.loads(path.read_text()) if path.is_file() else None

    cwd = capture / "agent_cwd.txt"
    eval_out = capture / "eval_out"
    artifacts_out = capture / "artifacts_out"
    evidence_out = capture / "evidence_out"
    return PipelineRun(
        home=load("home.json") or {},
        agent_cwd=cwd.read_text().strip() if cwd.is_file() else "",
        eval_home=load("eval_home.json"),
        snapshot=load("snapshot.json"),
        eval_argv=load("eval_argv.json"),
        returncode=proc.returncode,
        log=proc.stdout + proc.stderr,
        eval_files=(
            {p.name: p.read_bytes() for p in sorted(eval_out.iterdir()) if p.is_file()}
            if eval_out.is_dir()
            else None
        ),
        final_snapshot=load("final_snapshot.json"),
        artifact_files=(
            {p.name: p.read_bytes() for p in sorted(artifacts_out.iterdir()) if p.is_file()}
            if artifacts_out.is_dir()
            else None
        ),
        evidence_files=(
            {
                p.relative_to(evidence_out).as_posix(): p.read_bytes()
                for p in sorted(evidence_out.rglob("*"))
                if p.is_file()
            }
            if evidence_out.is_dir()
            else None
        ),
    )
