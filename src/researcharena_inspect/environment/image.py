"""The agent's image: upstream's `containers/arena.def` built from `Dockerfile` beside this file.

Referred to by digest only, so a re-pushed tag cannot change what runs. `compose.yaml` carries
the same reference; a test checks they agree.
"""

from typing import Final

IMAGE_REPOSITORY: Final = "ghcr.io/archit-manek/researcharena-inspect/arena"

IMAGE_DIGEST: Final = "sha256:e1cebf2cfc3a3868b272d91413275d884095b930f97475a2180ea1f95ca68d5d"
"""Built by run 43935849324 of `environment-image.yml` on 16 Nov 2026 (tags `upstream-b922f4b`
and `dockerfile-<hash>`); 13.8 GB compressed in 19 layers."""

IMAGE: Final = f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"

# D-133: vLLM 0.19.1 (upstream's containers/requirements.txt); the published inference runs
# ran 0.17.0.
