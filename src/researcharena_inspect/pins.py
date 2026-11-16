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

# Website run index: https://research-arena.ai/traces/data/index.json, fetched 30 Oct 2026.
# The site doesn't version it, so a gzipped snapshot is in the package. The hash is of the
# uncompressed file.
INDEX_URL = "https://research-arena.ai/traces/data/index.json"
INDEX_PATH = Path(__file__).parent / "data" / "researcharena_index.json.gz"
INDEX_SHA256 = "1f7dd4d846a186546fee2c4aa09335621a4135826d33f948a0e6cd6f3bbd752c"
INDEX_BUILD_TS = "2026-07-20T03:55:29Z"
