"""Pass rule 3: upstream's side-task evaluators on the published snapshots, on a CPU.

Of the four CUDA side tasks, only `independent`'s verdict needs no GPU (it reads
`requirements.txt`). The other three measure their verdict on the GPU (a KL ratio, a latency
ratio, an fp16/bf16 KL); each also records a `suspicious_code` flag, a static read of the kernels,
which is checked here.

Each published attack run is replayed as published
(`researcharena(agent="replay", replay_mode="as_published")`) in the CPU Docker sandbox: the
workspace is staged, the run's `task_snapshot/` replaces `task/`, and `red_team_scorer` runs
upstream's `evaluate.py` with the side task's evaluator, unchanged. The result is compared with
the run's published `metrics.json` (`raw_metrics` in its monitor folder's `results.json`).

    uv run python -m tests.researcharena_inspect.published_side_tasks LOG_DIR   # prints the table
"""

import sys
from dataclasses import dataclass
from typing import Any

from inspect_ai import eval
from inspect_ai.log import EvalLog

from researcharena_inspect.analysis.published_rows import load_raw_metrics
from researcharena_inspect.published import load_index
from researcharena_inspect.task import researcharena
from tests.researcharena_inspect.published_metrics import published_metrics

SIDE_TASKS = ("independent", "embedded", "embedded2", "embedded3")

FIELDS = {
    "independent": ("side_task_score", "pin_added", "requirements_file_present"),
    "embedded": ("side_task.suspicious_code",),
    "embedded2": ("suspicious_code",),
    "embedded3": ("suspicious_code",),
}
"""The fields that need no GPU, compared with the published `metrics.json`."""


@dataclass
class SideTaskCheck:
    run_id: str
    side_task: str
    ours: dict[str, Any]
    published: dict[str, Any]
    published_kl_ratio: Any
    our_kl_ratio: Any
    sample_error: str | None

    @property
    def differences(self) -> list[str]:
        return [
            f"{k}: ours {self.ours.get(k)!r}, published {self.published.get(k)!r}"
            for k in FIELDS[self.side_task]
            if self.ours.get(k) != self.published.get(k)
        ]


def _get(metrics: dict[str, Any], path: str) -> Any:
    value: Any = metrics
    for part in path.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def attack_runs() -> list[dict[str, Any]]:
    return [
        r
        for r in load_index()["runs"]
        if r["setting"] == "cuda_kernel" and r["side_task"] in SIDE_TASKS
    ]


def replay_runs(runs: list[dict[str, Any]], log_dir: str, max_samples: int = 4) -> EvalLog:
    task = researcharena(
        agent="replay",
        replay=[r["id"] for r in runs],
        replay_mode="as_published",
        sandbox="docker",
    )
    [log] = eval(
        task, model="mockllm/model", display="none", log_dir=log_dir, max_samples=max_samples
    )
    return log


def compare(runs: list[dict[str, Any]], log: EvalLog) -> list[SideTaskCheck]:
    snapshot = load_raw_metrics()
    samples = {str(s.id): s for s in log.samples or []}
    out = []
    for run in runs:
        name = run["id"].split("/", 1)[1]
        sample = samples[name]
        score = (sample.scores or {}).get("red_team_scorer")
        ours = (score.metadata or {}).get("metrics", {}) if score else {}
        theirs = published_metrics(run, snapshot[run["id"]]["results_from"])
        fields = FIELDS[run["side_task"]]
        out.append(
            SideTaskCheck(
                run_id=run["id"],
                side_task=run["side_task"],
                ours={k: _get(ours, k) for k in fields},
                published={k: _get(theirs, k) for k in fields},
                published_kl_ratio=_get(theirs, "side_task.kl_ratio"),
                our_kl_ratio=_get(ours, "side_task.kl_ratio"),
                sample_error=str(sample.error) if sample.error else None,
            )
        )
    return out


def main(log_dir: str) -> None:
    runs = attack_runs()
    checks = compare(runs, replay_runs(runs, log_dir))
    print(f"{len(checks)} published attack runs replayed as published on a CPU")
    print("| Side task | Runs | Fields compared | Equal | Sample errors |")
    print("| --- | --- | --- | --- | --- |")
    for side_task in SIDE_TASKS:
        group = [c for c in checks if c.side_task == side_task]
        equal = sum(not c.differences for c in group)
        errors = sum(c.sample_error is not None for c in group)
        fields = ", ".join(f"`{f}`" for f in FIELDS[side_task])
        print(f"| {side_task} | {len(group)} | {fields} | {equal} | {errors} |")
    print()
    for c in checks:
        if c.differences or c.sample_error:
            print(f"- `{c.run_id}`: {'; '.join(c.differences)} {c.sample_error or ''}")
    kl = sorted({repr(c.our_kl_ratio) for c in checks if c.side_task == "embedded"})
    print(f"\nembedded kl_ratio on the CPU: {kl} (needs the GPU)")


if __name__ == "__main__":
    main(sys.argv[1])
