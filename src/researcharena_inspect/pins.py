"""Pinned upstream versions, which every part of the port is checked against.

Change a pin only on purpose: it changes what "the original" means for every test.
"""

from pathlib import Path

# Code: https://github.com/aisa-group/ResearchArena (MIT).
UPSTREAM_REPO_URL = "https://github.com/aisa-group/ResearchArena.git"

# The upstream commit the port replicates.
UPSTREAM_COMMIT = "b922f4b94c9a31a8124e2ed32752f073ab583366"

# The paper's release (first public commit). Only for comparisons.
PAPER_COMMIT = "58776e09339c1714d0524130590abaf6eac7879d"

# Published runs: https://huggingface.co/datasets/aisa-group/ResearchArena-Trajectories
HF_DATASET = "aisa-group/ResearchArena-Trajectories"
HF_REVISION = "8cd737c58bdac876fbf7b137796508c99c6e9588"

# The authors' post-training models, one `final_model/` per run at
# `<setting>/<run>/red_team/task_snapshot/final_model/` (the trajectories leave it out):
# https://huggingface.co/datasets/aisa-group/ResearchArena-output-models.
OUTPUT_MODELS_DATASET = "aisa-group/ResearchArena-output-models"
OUTPUT_MODELS_REVISION = "28acb91599e8f2f6a9363a71bf854f9b5e6dd42e"
# The capabilities runs' model files at that revision (path, size, SHA-256 or git blob id), so a
# replay checks a model without calling the Hub. The hash is of the uncompressed file.
OUTPUT_MODELS_LISTING = Path(__file__).parent / "data" / "output_models_utility_finetuning.json.gz"
OUTPUT_MODELS_LISTING_SHA256 = "092d151c4551274370def67d5589298b4fce63ef57411867d401de863c07f62d"
# D-212: the safety runs' listing leaves out one run's second copy at `task/final_model/`.
OUTPUT_MODELS_LISTINGS = {
    "utility_finetuning": (OUTPUT_MODELS_LISTING, OUTPUT_MODELS_LISTING_SHA256),
    "safety_finetuning": (
        Path(__file__).parent / "data" / "output_models_safety_finetuning.json.gz",
        "f50d344b13edf0949ef6b9cb81e2c995682c6dda846b5348e1698b2c347b91e2",
    ),
}

# Website run index, fetched 30 Oct 2026. The site does not version it, so a gzipped snapshot is
# in the package. The hash is of the uncompressed file.
INDEX_URL = "https://research-arena.ai/traces/data/index.json"
INDEX_PATH = Path(__file__).parent / "data" / "researcharena_index.json.gz"
INDEX_SHA256 = "1f7dd4d846a186546fee2c4aa09335621a4135826d33f948a0e6cd6f3bbd752c"
INDEX_BUILD_TS = "2026-07-20T03:55:29Z"

# Claude Code's background model in the published runs (their `result` events' `modelUsage`).
CLAUDE_CODE_SMALL_MODEL = "anthropic/claude-haiku-4-5-20251001"
