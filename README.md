# researcharena-inspect

An [Inspect](https://inspect.aisi.org.uk/) port of [ResearchArena](https://research-arena.ai/) ([paper](https://arxiv.org/abs/2607.19321), [code](https://github.com/aisa-group/ResearchArena)), an AI-control evaluation of sabotage and monitoring in automated AI R&D. An agent does a long research-engineering task while some runs also ask it to sabotage the work in secret; a monitor model then reads the run and scores how suspicious it looks.

New to the benchmark or to this code? Start with the guide, [docs/guide.html](docs/guide.html): one run stage by stage, the four settings, the monitor, the evidence and the differences from the original, each linked to the lines it describes. It is a single HTML file; open it in a browser after cloning (GitHub shows its source, not the page). It was written on 18 Nov 2026 against commit `68c52cc`, and its links point at that commit.

The eval's documentation is in [src/researcharena_inspect/README.md](src/researcharena_inspect/README.md). For a review, read it in this order:

1. [What is in the port](src/researcharena_inspect/README.md#what-is-in-the-port) and [what "the original" means here](src/researcharena_inspect/README.md#what-the-original-means-here): the port follows upstream's code at commit `b922f4b`, reuses its task files, evaluators and templates unchanged, and rewrites only the bash orchestration as Inspect tasks.
2. [What has been verified, and how](src/researcharena_inspect/README.md#what-has-been-verified-and-how): one row per check, with the result and the test or run behind it, and a list of what has not been verified.
3. [Deviations from upstream](src/researcharena_inspect/README.md#deviations-from-upstream) and [Known limitations](src/researcharena_inspect/README.md#known-limitations).
4. [How to run it](src/researcharena_inspect/README.md#how-to-run-it), locally or on Hawk, and the paper's setup as a run config: [`run_configs/original.yaml`](src/researcharena_inspect/run_configs/original.yaml).

Where things stand (18 Nov 2026): all four settings are ported. Prompts, workspace staging and scorers match upstream's own scripts in the tests, and the paper's tables reproduce from the authors' published runs except for cells listed with their causes. On an NVIDIA H100, the paper's GPU: `cuda_kernel` has sixteen one-hour live runs and published workspaces replayed through the scorer; `inference_optimization` has its scorer checks, with one-hour live runs under way. `utility_finetuning`, `safety_finetuning` and scoring in a container of its own (version 6-A onwards) have run on a CPU only, with stand-ins for the GPU parts; their GPU checks are prepared. The details, and what has not been verified, are in [What has been verified, and how](src/researcharena_inspect/README.md#what-has-been-verified-and-how).

## Getting started

```bash
uv sync
uv run pre-commit install
uv run inspect eval researcharena_inspect/researcharena_monitor --model anthropic/claude-sonnet-4-6 \
    -T runs=cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740
uv run inspect view
```

That command runs the monitor on one of the authors' published CUDA runs (about $0.60) and needs Docker but no GPU. The red-team task (`researcharena_inspect/researcharena`) needs one NVIDIA GPU; see [How to run it](src/researcharena_inspect/README.md#how-to-run-it).

Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time.

The dev dependencies include the `anthropic` and `openai` SDKs. Inspect needs one package per model provider, so add the package for any other provider with `uv add --dev <package>`; Inspect's error message names the package when it is missing.

## Checks and tests

```bash
make check   # ruff, basedpyright, inspect-evals-lint, generated README sections, pre-commit
make test    # pytest
```

Slow tests (Docker parity with upstream's scripts, published-data checks) run with `uv run pytest --runslow`. No test calls a paid model or needs a GPU. See [CONTRIBUTING.md](CONTRIBUTING.md) for the other gates and markers.

## Layout

```text
src/researcharena_inspect/        the eval
  task.py                         the red-team task, `researcharena`
  monitor/                        the monitor task, `researcharena_monitor`
  agents/                         Claude Code, ReAct, null and replay agents; the re-prompting loop
  environment/                    the agent's image, sandbox configs, workspace staging, environment_check
  scoring/                        upstream's evaluate.py as an Inspect scorer; kernel_crash_check
  prompts/                        the agent and monitor prompts
  transcripts/                    the transcripts a monitor reads, made by upstream's parser
  analysis/                       thresholds, detection rate, AUROC, the paper's tables
  run_configs/                    original.yaml: the paper's setup as an `inspect eval --run-config` file
  pins.py, upstream.py, published.py   pinned versions, the upstream checkout, the published runs
  eval.yaml, README.md            the eval's metadata and documentation
tests/researcharena_inspect/      its tests
hawk/                             job files for GL's Hawk cluster, as they ran
agent_artefacts/trajectory_analysis/inspect_scout/   Scout scanners for the transcript audit
```

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Keeping up with the template

A weekly workflow runs `copier update` and opens a pull request when the template changed.
`README.md`, `src/researcharena_inspect/` and `tests/researcharena_inspect/` are never overwritten by an update.
