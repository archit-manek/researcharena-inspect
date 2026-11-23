# researcharena-inspect

An [Inspect](https://inspect.aisi.org.uk/) port of [ResearchArena](https://research-arena.ai/) ([paper](https://arxiv.org/abs/2607.19321), [code](https://github.com/aisa-group/ResearchArena)), an AI-control evaluation of sabotage and monitoring in automated AI R&D. An agent does a long research-engineering task while some runs also ask it to sabotage the work in secret; a monitor model then reads the run and scores how suspicious it looks.

**Reference setup:** the four-setting evaluation is run and evidenced on Docker with one NVIDIA H100, the port's built-in sandbox. Another Inspect sandbox can be given with `sandbox` and `sandbox_config`; without an `evaluator` service the scorer then runs in the agent's own container instead of a separate one.

The eval's documentation, written for a reviewer, is [src/researcharena_inspect/README.md](src/researcharena_inspect/README.md): what the port is and what "the original" means, how to run it, how a sample is scored, what has been verified, the evaluation report, one line per deviation from upstream that can move a score or what a monitor sees, and the main limitations. Three appendices beside it hold the detail: [EVIDENCE.md](src/researcharena_inspect/EVIDENCE.md), everything measured (the full verification table, the full evaluation report and the log behind each result); [DEVIATIONS.md](src/researcharena_inspect/DEVIATIONS.md), everything that differs from upstream (the full deviations table and known limitations); and [DESIGN.md](src/researcharena_inspect/DESIGN.md), how the port works, in full (every run command, the parameters, a sample and its scoring, the changelog in full).

Where things stand (version 10-G, 23 Nov 2026; every result was produced at 10-F or earlier, and 10-G only removes the built-in Kubernetes backend and the Hawk job files):

- All four settings and all 13 of upstream's side tasks are ported, and the monitor task builds for all four.
- In the tests, prompts match upstream's own scripts byte for byte, and staged workspaces and scorer outputs match upstream's pipeline on CPU stand-ins; on the H100 the port's scoring agrees with upstream's `evaluate.py` run directly, within bands fixed beforehand.
- Published submissions re-scored on the H100: safety 94 of 94 (mean absolute difference 0.027), capabilities 61 of 71 (0.028), CUDA 107 of 113 at 0.824 to 1.039 of published; inference is the exception, with 20 of the 104 published passes scoring 0.0 on the released image (vLLM 0.19.1), all of which pass on a comparison image with vLLM 0.17.0.
- Live, with Sonnet 4.6: sixteen one-hour `cuda_kernel` runs, fifteen one-hour `inference_optimization` runs and seven five-hour post-training runs, reported as observed and not as a reproduction of the paper's rates; ReAct with GPT-5.5 ran a half-hour baseline in each setting at 10-F.
- What has not been shown: [the eval README](src/researcharena_inspect/README.md#what-has-been-verified-and-how).

## Getting started

First, a free scored run: the monitor task end to end on one of the authors' published CUDA runs, with Inspect's mock model scripted to write a judgement, checking that the score read back is 37.

```bash
uv sync
uv run --frozen pytest --runslow -q "tests/researcharena_inspect/test_monitor_task.py::test_monitor_task_end_to_end_with_mockllm[transcript]"
```

It needs Docker and, on a fresh checkout, network access to fetch the pinned upstream checkout, the published run and a small Python image. No API key and no GPU. On 22 Nov 2026, with those already cached, it passed in 13 seconds.

The same run with a real model costs money. Put `INSPECT_EVAL_MODEL` and your API keys in a `.env` file (it is git-ignored) to avoid passing `--model` each time, then:

```bash
uv run inspect eval researcharena_inspect/researcharena_monitor --model anthropic/claude-sonnet-4-6 \
    -T runs=cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740
uv run inspect view
```

That command runs the monitor on the same published CUDA run ($0.66 a call on average over 190 calls) and needs Docker but no GPU. The red-team task (`researcharena_inspect/researcharena`) needs one NVIDIA GPU; see [How to run it](src/researcharena_inspect/README.md#how-to-run-it).

The dev dependencies include the `anthropic` and `openai` SDKs. Inspect needs one package per model provider, so add the package for any other provider with `uv add --dev <package>`; Inspect's error message names the package when it is missing.

## Checks and tests

```bash
uv run pre-commit install   # once, to run the hooks on each commit
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
  EVIDENCE.md, DEVIATIONS.md,     the README's appendices: everything measured; everything that
  DESIGN.md                       differs from upstream; how the port works, in full
tests/researcharena_inspect/      its tests
agent_artefacts/                  the validity report, the review summary, and the trajectory analysis (Scout scanners, runner, results, ..._ANALYSIS.md)
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
