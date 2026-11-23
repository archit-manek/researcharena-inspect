# AGENTS.md

This repo holds one Inspect AI evaluation, `researcharena-inspect`, as the Python package `researcharena_inspect` under `src/`. These notes are for coding agents working in it.

## Commands

This project uses `uv`. Do not use `pip`, bare `python` or bare `pytest`.

- `uv sync` installs the project and its dev dependencies.
- `uv run inspect eval researcharena_inspect/researcharena --model <provider/model>` runs the red-team task (it needs one NVIDIA GPU); `researcharena_inspect/researcharena_monitor` runs the monitor on published runs. `--limit N` runs N samples.
- `uv run inspect eval --run-config src/researcharena_inspect/run_configs/original.yaml` runs the paper's CUDA setup.
- `make test` runs the tests. Slow tests need `uv run pytest --runslow`.
- `make check` runs every check CI runs.
- `make readmes` regenerates the README sections that come from `eval.yaml` and the task signatures.

## Layout

- `src/researcharena_inspect/task.py`: the red-team task, `researcharena`. `monitor/task.py`: the monitor task, `researcharena_monitor`. `researcharena`, `researcharena_monitor` and the check tasks pass the `eval.yaml` version to `Task`.
- `src/researcharena_inspect/eval.yaml`: title, description, contributors, version and tasks. Edit it and run `make readmes`.
- `src/researcharena_inspect/run_configs/`: `original.yaml` and `original_monitor.yaml`, the paper's setup as `inspect eval --run-config` files, each value commented with where upstream sets it.
- `src/researcharena_inspect/README.md`: the eval's documentation for a reviewer, short: the generated sections, a summary of what has been verified and of the evaluation report, one line per deviation that can move a score, a success verdict or what a monitor sees, and the main limitations.
- `src/researcharena_inspect/EVIDENCE.md`: everything measured: the full verification table, the full evaluation report and the log behind each result.
- `src/researcharena_inspect/DEVIATIONS.md`: everything that differs from upstream: the full deviations table, the analysis choices and the full known limitations.
- `src/researcharena_inspect/DESIGN.md`: how the port works, in full: what is in it, what you need, every run command, the notes on the parameters, a red-team sample and its scoring, and the changelog in full.
- `tests/researcharena_inspect/`: the tests.
- `agent_artefacts/trajectory_analysis/inspect_scout/`: Scout scanners for the transcript audit.
- `agent_artefacts/trajectory_analysis/researcharena_inspect/`: this eval's Scout scanners, the one-command runner for the trajectory analysis (manifest of live logs, dry run, known answers, validity with the paper's success rule) and its `..._ANALYSIS.md` files; see its README.

## How the port is built

This repo ports [ResearchArena](https://github.com/aisa-group/ResearchArena) to Inspect. "The original" is upstream's latest code at the commit in `src/researcharena_inspect/pins.py`.

- **Reuse, don't rewrite.** Upstream's task files, scorers and templates are used unchanged from the pinned checkout (`researcharena_inspect.upstream.upstream_dir()`). Only its bash orchestration is rewritten as Inspect code.
- **Upstream is the oracle.** Where the port re-implements something upstream does (building a prompt, rendering a transcript), a test runs upstream's own script from the pinned checkout and requires identical output. A difference is either a bug in the port or a documented deviation.
- **Published runs are the second check.** `researcharena_inspect.published` loads the authors' 386 published runs and fetches their files at a pinned revision.
- **Deviations are explicit.** Upstream's behaviour is the default, flaws included. A bug is fixed only if it stops a run or a score from being produced. Every deviation gets a comment with its ledger id (`# D-NNN`), a row in the deviations table of `src/researcharena_inspect/DEVIATIONS.md` and, if it can move a score, a success verdict or what a monitor sees, a line in the README's.
- **No spend in tests.** Tests never call a paid model or need a GPU unless marked `slow` and run on purpose.
- **New dependencies** must have been on PyPI for at least 7 days, or GL's infrastructure refuses to install the package.

Modules: `pins.py`, `upstream.py`, `published.py` (shared base); `prompts/` (agent and monitor prompts); `task.py` and `agents/` (red-team task; Claude Code, ReAct, null and replay agents; the re-prompting loop); `environment/` (image, sandbox configs, workspace staging, `environment_check`); `scoring/` (upstream's `evaluate.py` as a scorer, the opt-in CUDA baseline process of D-160, `kernel_crash_check`); `transcripts/` (the text a monitor reads); `monitor/` (monitor task); `analysis/` (success rules, detection rate, AUROC, the paper's tables).

## Checks

`make check` runs ruff (lint and format), basedpyright in standard mode, inspect-evals-lint, `inspect-evals-readmes generate --check` and the pre-commit stack (zizmor, actionlint, mdformat). CI runs the same through python-project-template's shared workflow plus `eval-checks.yml`.

Tests must be deterministic and must not reach the network unless marked: use `mockllm/model` for model output, `unittest.mock` for external services and `tmp_path` for files. Mark tests that download datasets with `@pytest.mark.dataset_download`, that need Docker with `@pytest.mark.docker`, and that take long with `@pytest.mark.slow`. Add `@pytest.mark.huggingface` to a test that calls the Hugging Face Hub, so a test that downloads a dataset from the Hub carries both markers. Tests marked `huggingface` run when `HF_TOKEN` is unset, which includes CI, and the test summary then says how many ran without authentication.

## Guides

- [BEST_PRACTICES.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/BEST_PRACTICES.md)
- [EVALUATION_CHECKLIST.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/EVALUATION_CHECKLIST.md)
- [TASK_VERSIONING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/TASK_VERSIONING.md): bump `version` in `eval.yaml` when a change can move scores.
- [CONTRIBUTING.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md) in inspect_evals, for the `eval.yaml` reference and the register.
- [inspect-evals-lint rules](https://github.com/Generality-Labs/inspect-evals-lint/tree/main/docs/rules), one page per rule code.

## Skills

The `inspect-evals` plugin from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills) provides `create-eval`, `investigate-dataset`, `ensure-test-coverage`, `read-eval-logs`, `check-trajectories-workflow`, `eval-validity-review` and `eval-report-workflow`. Claude Code loads it from `.claude/settings.json`; see README.md for Codex and other agents.
