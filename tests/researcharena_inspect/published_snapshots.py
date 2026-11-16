"""Compare the workspace the port stages with published CUDA runs (pass rule 3).

A published run keeps `red_team/task_snapshot/`, the agent's `task/` folder after the run
(cleaned of model weights). The fair comparison is file by file: a file the agent did not
touch should be byte-identical to what the port stages at the start; every other difference
is classified, not forced to match. Hugging Face keeps no modes, links or empty folders, so
only bytes are compared.

    uv run python -m tests.researcharena_inspect.published_snapshots   # prints the table
"""

import hashlib
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from huggingface_hub import HfApi

from researcharena_inspect.environment.staging import home_entries
from researcharena_inspect.pins import HF_DATASET, HF_REVISION, PAPER_COMMIT, UPSTREAM_COMMIT
from researcharena_inspect.published import fetch_run_file, load_index
from researcharena_inspect.upstream import upstream_dir

SNAPSHOT = "red_team/task_snapshot"
CATEGORIES = ("baseline", "embedded", "embedded2", "embedded3", "independent")
PROVIDERS = ("claude", "codex", "opencode")
_CREATION = re.compile(rb"^CREATION_DATE=(\d+)$", re.MULTILINE)

# Files the agent is asked to change (program.md: "Only edit files in kernels/ and
# bindings.cpp"); a difference there is the agent's work.
AGENT_EDITABLE = re.compile(r"^(kernels/[^/]+\.cu|bindings\.cpp)$")

STAGED_EPOCH = 1_700_000_000
"""`date +%s` when the port's timer.sh is made here; each run's own time shows as a difference."""


def select_runs(per_provider: int = 1) -> list[dict]:
    """The first published CUDA run(s) of each agent family, for every category, by id."""
    runs = [r for r in load_index()["runs"] if r["setting"] == "cuda_kernel"]
    chosen = []
    for category in CATEGORIES:
        for provider in PROVIDERS:
            matches = sorted(
                (r for r in runs if r["side_task"] == category and r["red_provider"] == provider),
                key=lambda r: r["id"],
            )
            chosen += matches[:per_provider]
    return chosen


def published_snapshot(run_id: str) -> dict[str, bytes]:
    """Every file under the run's `task_snapshot/`, by path relative to it."""
    prefix = f"{run_id}/{SNAPSHOT}/"
    tree = HfApi().list_repo_tree(
        HF_DATASET,
        repo_type="dataset",
        revision=HF_REVISION,
        path_in_repo=f"{run_id}/{SNAPSHOT}",
        recursive=True,
    )
    files = {}
    for entry in tree:
        if getattr(entry, "size", None) is None:  # a folder
            continue
        rel = entry.path.removeprefix(prefix)
        files[rel] = fetch_run_file(run_id, f"{SNAPSHOT}/{rel}").read_bytes()
    return files


def timer_for(hours: str, epoch: int) -> bytes:
    """`create_timer.sh <hours>` as upstream's script writes it, with `date +%s` = epoch."""
    script = upstream_dir() / "src" / "utils" / "create_timer.sh"
    shim = f'date() {{ if [ "$1" = "+%s" ]; then echo {epoch}; else command date "$@"; fi; }}'
    text = script.read_text()
    out = subprocess.run(
        ["bash", "-c", f"{shim}\n{text}", "create_timer.sh", hours],
        capture_output=True,
        check=True,
    ).stdout
    return out


def staged_task(side_task: str, hours: str, epoch: int) -> dict[str, bytes]:
    """The port's `task/` at the agent's start, without `.git` (Hugging Face keeps none)."""
    files = {
        e.path.removeprefix("task/"): e.content
        for e in home_entries("cuda_kernel", side_task, agent=None, prompt=None)
        if e.path.startswith("task/") and e.content is not None
    }
    files["timer.sh"] = timer_for(hours, epoch)
    if side_task == "independent":  # dropped by the side task's pre_agent.sh hook
        files["requirements.txt"] = (
            upstream_dir() / "settings/cuda_kernel/side_tasks/independent/requirements.txt"
        ).read_bytes()
    return files


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def upstream_versions(rel: str) -> dict[str, str]:
    """Blob id -> the public upstream commits whose task_context has it at `rel` (first..last)."""
    repo = upstream_dir()
    path = f"settings/cuda_kernel/task_context/{rel}"
    commits = subprocess.run(
        ["git", "-C", str(repo), "log", "--all", "--reverse", "--format=%H %as", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    spans: dict[str, list[str]] = {}
    for line in filter(None, commits):
        commit, date = line.split()
        blob = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{commit}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if blob:
            spans.setdefault(blob, []).append(f"{commit[:7]} ({date})")
    return {blob: " to ".join(dict.fromkeys([seen[0], seen[-1]])) for blob, seen in spans.items()}


@dataclass
class Difference:
    path: str
    kind: str
    detail: str = ""


# Written into task/ by the agent's launcher, not by staging: `cat > opencode.json` in
# agents/opencode/solve.sh:55, run with task/ as the working directory.
LAUNCHER_FILES = {"opencode.json": "agents/opencode/solve.sh:55"}


def _new_file(
    path: str, data: bytes, staged: dict[str, bytes], published: dict[str, bytes]
) -> Difference:
    """A file in the published snapshot that staging does not make."""
    top = path.split("/")[0]
    if top in ("workspace", "__pycache__"):
        return Difference(path, f"made during the run ({top}/)")
    if path in LAUNCHER_FILES:
        return Difference(
            path, "made during the run (written by the agent's launcher)", LAUNCHER_FILES[path]
        )
    inner = path.split("/", 1)[1] if "/" in path else ""
    if top == "task" and inner and data in (staged.get(inner), published.get(inner)):
        return Difference(path, "made during the run (the agent's copy of the task folder)")
    if top == "task" and inner in staged:
        return Difference(
            path, "made during the run (the agent's copy of the task folder, mid-edit)"
        )
    return Difference(path, "made during the run (another file the agent wrote)")


def classify(
    side_task: str, staged: dict[str, bytes], published: dict[str, bytes]
) -> list[Difference]:
    """Every path where the staged start and the published end differ, with its class."""
    out: list[Difference] = []
    for path in sorted(set(staged) | set(published)):
        mine, theirs = staged.get(path), published.get(path)
        if mine == theirs:
            continue
        if theirs is None:
            out.append(Difference(path, "staged, not in the published snapshot"))
            continue
        if mine is None:
            out.append(_new_file(path, theirs, staged, published))
            continue
        if path == "timer.sh":
            match = _CREATION.search(theirs)
            normal = _CREATION.sub(b"CREATION_DATE=X", theirs)
            if match and normal == _CREATION.sub(b"CREATION_DATE=X", mine):
                when = datetime.fromtimestamp(int(match.group(1)), UTC).isoformat()
                out.append(Difference(path, "timer.sh: only CREATION_DATE differs", when))
            else:
                out.append(Difference(path, "timer.sh: differs beyond CREATION_DATE"))
            continue
        if AGENT_EDITABLE.match(path):
            out.append(Difference(path, "edited by the agent (a file it is told to edit)"))
            continue
        if path == "requirements.txt" and side_task == "independent":
            if theirs.startswith(mine.rstrip(b"\n")):
                added = theirs[len(mine.rstrip(b"\n")) :].decode(errors="replace").strip()
                out.append(Difference(path, "side task's file, lines appended", added))
            else:
                out.append(Difference(path, "side task's file, changed otherwise"))
            continue
        versions = upstream_versions(path)
        origin = versions.get(_blob_id(theirs))
        if origin is not None:
            out.append(Difference(path, "harness file: an earlier public upstream version", origin))
        else:
            out.append(
                Difference(
                    path,
                    "harness file: matches no public upstream version",
                    f"{len(mine)} -> {len(theirs)} bytes",
                )
            )
    return out


def compare_runs(runs: list[dict]) -> list[tuple[dict, list[Difference], int]]:
    """For each run: its differences and the number of identical staged files."""
    results = []
    for run in runs:
        published = published_snapshot(run["id"])
        side_task = "none" if run["side_task"] == "baseline" else run["side_task"]
        staged = staged_task(side_task, str(run["hours"]), STAGED_EPOCH)
        diffs = classify(run["side_task"], staged, published)
        identical = sum(1 for p in staged if staged[p] == published.get(p))
        results.append((run, diffs, identical))
    return results


def main() -> None:
    print(f"upstream {UPSTREAM_COMMIT[:7]} (paper commit {PAPER_COMMIT[:7]}), HF {HF_REVISION[:7]}")
    totals: dict[str, int] = defaultdict(int)
    for run, diffs, identical in compare_runs(select_runs()):
        print(f"\n## {run['id']}  (hours={run['hours']}, ts={run['ts']})")
        print(f"   identical staged files: {identical}")
        for d in diffs:
            totals[d.kind] += 1
            print(f"   - {d.path}: {d.kind}" + (f" [{d.detail}]" if d.detail else ""))
    print("\n## Totals by class")
    for kind, count in sorted(totals.items()):
        print(f"   {count:4d}  {kind}")


if __name__ == "__main__":
    main()
