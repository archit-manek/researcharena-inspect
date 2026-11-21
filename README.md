# researcharena-inspect

An [Inspect](https://inspect.aisi.org.uk/) port of [ResearchArena](https://research-arena.ai/) ([paper](https://arxiv.org/abs/2607.19321), [code](https://github.com/aisa-group/ResearchArena)), an AI-control evaluation of sabotage and monitoring in automated AI R&D. An agent does a long research-engineering task while some runs also ask it to sabotage the work in secret; a monitor model then reads the run and scores how suspicious it looks.

New to the benchmark or to this code? Start with the guide, [docs/guide.html](docs/guide.html): one run stage by stage, the four settings, the monitor, the evidence and the differences from the original, each linked to the lines it describes. It is a single HTML file; open it in a browser after cloning (GitHub shows its source, not the page). It was written on 18 Nov 2026 against commit `68c52cc`, and its links point at that commit.

The eval's documentation is in [src/researcharena_inspect/README.md](src/researcharena_inspect/README.md). For a review, read it in this order:

1. [What is in the port](src/researcharena_inspect/README.md#what-is-in-the-port) and [what "the original" means here](src/researcharena_inspect/README.md#what-the-original-means-here): the port follows upstream's code at commit `b922f4b`, reuses its task files, evaluators and templates unchanged, and rewrites only the bash orchestration as Inspect tasks.
2. [What has been verified, and how](src/researcharena_inspect/README.md#what-has-been-verified-and-how): one row per check, with the result and the test or run behind it, and a list of what has not been verified.
3. [Deviations from upstream](src/researcharena_inspect/README.md#deviations-from-upstream) and [Known limitations](src/researcharena_inspect/README.md#known-limitations).
4. [How to run it](src/researcharena_inspect/README.md#how-to-run-it), locally or on Hawk, and the paper's setup as run configs, one per setting and one for the monitor: [`run_configs/`](src/researcharena_inspect/run_configs/).

Where things stand (19 Nov 2026 for `cuda_kernel` and `inference_optimization`; 21 Nov for the other two): all four settings are ported. Prompts, workspace staging and scorers match upstream's own scripts in the tests, and the paper's tables reproduce from the authors' published runs except for cells listed with their causes. On an NVIDIA H100, the paper's GPU: `cuda_kernel` has sixteen one-hour live runs, 47 published workspaces replayed through the scorer, and transcript and workspace monitors on our runs; `inference_optimization` has its scorer checks, fifteen one-hour live runs and transcript monitors on them. For these two settings, scoring in a container of its own (version 6-A onwards) has been checked on the H100 against the earlier scores. Replayed that way, twelve of the fifteen live inference runs score again and three give 0.0, because their submissions point vLLM at a compile cache in the agent's home (D-255). `utility_finetuning` and `safety_finetuning` have passed their scorer checks on the H100 (published models re-scored, and the port's scoring equal to upstream's scorer run directly within the checks' bands) and have one five-hour live Claude Code run per cell (capabilities: baseline, `embedded`, `independent`; safety: baseline, `embedded`, `embedded_german`, `independent`), with short Claude Code and ReAct runs; one run per cell, reported as observed. In both settings' upload (`independent`) runs the agent called the instruction a prompt injection and did not upload, so an upload being captured is shown by tests on a CPU, not by a live run. The details, and what has not been verified, are in [What has been verified, and how](src/researcharena_inspect/README.md#what-has-been-verified-and-how).

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
  environment/                    the agent's image, the containers, workspace staging, the model cache, the hand-over to scoring, environment_check
  scoring/                        upstream's evaluate.py as an Inspect scorer; what is kept of a run; kernel_crash_check
  prompts/                        the agent and monitor prompts
  transcripts/                    the transcripts a monitor reads, made by upstream's parser
  analysis/                       thresholds, detection rate, AUROC, the paper's tables
  run_configs/                    the paper's setup as `inspect eval --run-config` files: one per setting, one for the monitor
  pins.py, upstream.py, published.py   pinned versions, the upstream checkout, the published runs
  eval.yaml, README.md            the eval's metadata and documentation
tests/researcharena_inspect/      its tests
hawk/                             job files for GL's Hawk cluster, as they ran
docs/guide.html                   a guide for reviewers (open in a browser)
agent_artefacts/                  the validity report, the review summary, and Scout scanners for the transcript analysis (not run yet)
NOTICE                            attribution for the code adapted from ResearchArena (MIT)
```

## Skills

The `inspect-evals` skills (create-eval, investigate-dataset, ensure-test-coverage, read-eval-logs, check-trajectories-workflow, eval-validity-review, eval-report-workflow) come from [Generality-Labs/agent-skills](https://github.com/Generality-Labs/agent-skills).

- Claude Code: open this folder and trust it. The plugin loads from `.claude/settings.json`; skills appear as `/inspect-evals:<name>`.
- Codex: `codex plugin marketplace add Generality-Labs/agent-skills` then `codex plugin add inspect-evals@generality`, once per machine.
- Other agents: `npx skills add Generality-Labs/agent-skills --agent <agent>`.

## Keeping up with the template

A weekly workflow runs `copier update` and opens a pull request when the template changed.
`README.md`, `src/researcharena_inspect/` and `tests/researcharena_inspect/` are never overwritten by an update.
