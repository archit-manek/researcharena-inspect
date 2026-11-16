"""The agent's image: upstream's `containers/arena.def` built from `Dockerfile` beside this file.

`.github/workflows/environment-image.yml` builds it for linux/amd64 and pushes it; the eval
refers to it by digest only, so a re-pushed tag cannot change what runs. The sandbox configs
(`compose.yaml`, `k8s_values_*.yaml`) carry the same reference; a test checks they agree.
"""

from typing import Final

IMAGE_REPOSITORY: Final = "ghcr.io/archit-manek/researcharena-inspect/arena"

IMAGE_DIGEST: Final = "sha256:e1cebf2cfc3a3868b272d91413275d884095b930f97475a2180ea1f95ca68d5d"
"""Built by run 43935849324 of `environment-image.yml` on 16 Nov 2026 (tags `upstream-b922f4b`
and `dockerfile-<hash>`); 13.8 GB compressed in 19 layers."""

IMAGE: Final = f"{IMAGE_REPOSITORY}@{IMAGE_DIGEST}"
