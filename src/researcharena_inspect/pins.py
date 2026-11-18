"""Pinned upstream versions. Every part of the port is checked against these.

Change a pin only on purpose: it changes what "the original" means for every test.
"""

from pathlib import Path

# Code: https://github.com/aisa-group/ResearchArena (MIT).
UPSTREAM_REPO_URL = "https://github.com/aisa-group/ResearchArena.git"

# Latest upstream main (5 Nov 2026). The port replicates this version.
UPSTREAM_COMMIT = "b922f4b94c9a31a8124e2ed32752f073ab583366"

# First public commit (28 Jul 2026), the release that goes with the paper. Only for comparisons.
PAPER_COMMIT = "58776e09339c1714d0524130590abaf6eac7879d"

# Published runs: https://huggingface.co/datasets/aisa-group/ResearchArena-Trajectories
HF_DATASET = "aisa-group/ResearchArena-Trajectories"
HF_REVISION = "8cd737c58bdac876fbf7b137796508c99c6e9588"

# The authors' fine-tuned models from the two post-training settings, one `final_model/` per run
# at `<setting>/<run>/red_team/task_snapshot/final_model/` (the trajectories leave it out):
# https://huggingface.co/datasets/aisa-group/ResearchArena-output-models. Revision of 17 Nov 2026;
# 71 capabilities runs (every published one but the run that saved no model) and 94 safety runs.
OUTPUT_MODELS_DATASET = "aisa-group/ResearchArena-output-models"
OUTPUT_MODELS_REVISION = "28acb91599e8f2f6a9363a71bf854f9b5e6dd42e"
# The capabilities runs' files in that dataset at that revision (path, size, SHA-256 or git blob
# id), listed from the Hub on 18 Nov 2026, so a replay checks a model without calling the Hub. The
# hash is of the uncompressed file.
OUTPUT_MODELS_LISTING = Path(__file__).parent / "data" / "output_models_utility_finetuning.json.gz"
OUTPUT_MODELS_LISTING_SHA256 = "092d151c4551274370def67d5589298b4fce63ef57411867d401de863c07f62d"

# Website run index: https://research-arena.ai/traces/data/index.json, fetched 30 Oct 2026.
# The site doesn't version it, so a gzipped snapshot is in the package. The hash is of the
# uncompressed file.
INDEX_URL = "https://research-arena.ai/traces/data/index.json"
INDEX_PATH = Path(__file__).parent / "data" / "researcharena_index.json.gz"
INDEX_SHA256 = "1f7dd4d846a186546fee2c4aa09335621a4135826d33f948a0e6cd6f3bbd752c"
INDEX_BUILD_TS = "2026-07-20T03:55:29Z"

# The model Claude Code used for its own small background calls in the published runs: the
# `modelUsage` of their `result` events lists it next to the main model.
CLAUDE_CODE_SMALL_MODEL = "anthropic/claude-haiku-4-5-20251001"
