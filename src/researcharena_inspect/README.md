# researcharena-inspect

Inspect port of ResearchArena, a control evaluation of sabotage and monitoring in automated AI R&D

<!-- Contributors: Automatically Generated -->
Contributed by [@archit-manek](https://github.com/archit-manek)
<!-- /Contributors: Automatically Generated -->

<!-- Usage: Automatically Generated -->
## Usage

First, install dependencies:

```bash
uv sync
```

Then run evaluations:

```bash
uv run inspect eval researcharena_inspect/researcharena_inspect --model openai/gpt-5-nano
uv run inspect eval researcharena_inspect/gpu_probe --model openai/gpt-5-nano
```

You can also import tasks as Python objects:

```python
from inspect_ai import eval
from researcharena_inspect import researcharena_inspect, gpu_probe
eval(researcharena_inspect)
```

If your directory name doesn't match your package name but you have registered your eval task, you can also run:

```bash
uv run inspect eval researcharena_inspect --model openai/gpt-5-nano
```

After running evaluations, view logs with:

```bash
uv run inspect view
```

If you don't want to specify `--model` each time, create a `.env` file:

```bash
INSPECT_EVAL_MODEL=anthropic/claude-opus-4-1-20250805
ANTHROPIC_API_KEY=<anthropic-api-key>
```
<!-- /Usage: Automatically Generated -->

<!-- Options: Automatically Generated -->
## Options

You can control a variety of options from the command line. For example:

```bash
uv run inspect eval researcharena_inspect/researcharena_inspect --limit 10
uv run inspect eval researcharena_inspect/gpu_probe --max-connections 10
uv run inspect eval researcharena_inspect/researcharena_inspect --temperature 0.5
```

See `uv run inspect eval --help` for all available options.
<!-- /Options: Automatically Generated -->

<!-- Parameters: Automatically Generated -->
## Parameters

### `researcharena_inspect`

No task parameters.

### `gpu_probe`

- `sandbox_type` (Literal['k8s', 'local']): "k8s" requests one GPU from a Kubernetes cluster (for example Hawk); "local" runs on the machine Inspect runs on. (default: `'k8s'`)

<!-- /Parameters: Automatically Generated -->

## Dataset

Describe the dataset here: where it comes from, how many samples it has, and how a record becomes a `Sample`.

## Scoring

Describe the scorer here: what counts as correct, and which metrics the task reports.

## Deviations from the original implementation

| What changed | Why | Expected effect on scores |
| --- | --- | --- |
| Nothing yet. | | |

## Architecture

Add a diagram of the task here: dataset, solver, scorer, and any sandbox or judge model.
