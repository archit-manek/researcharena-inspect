# syntax=docker/dockerfile:1.7
#
# The agent's image: upstream's containers/arena.def (ResearchArena, MIT,
# https://github.com/aisa-group/ResearchArena) translated line by line to a Dockerfile, at the
# pinned commit b922f4b94c9a31a8124e2ed32752f073ab583366. Each block below names the arena.def
# section it comes from; what is not reproduced, and why, is listed at the end of this file.
#
# Built for linux/amd64 by .github/workflows/environment-image.yml and pushed to
# ghcr.io/archit-manek/researcharena-inspect/arena; the eval refers to it by digest
# (researcharena_inspect.environment.image). No task files, data or secrets are baked in:
# the workspace is staged into the running sandbox from the pinned checkout.

# arena.def `From: nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04`, pinned to the linux/amd64
# image that tag named on 16 Nov 2026 (index digest
# sha256:827e01ba745d5488e8f199bb9331ca006da97542909af4076b89754cbf5a5a55).
FROM --platform=linux/amd64 nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04@sha256:0447c3b139d17436d399cc9d5070062c5013bb1b1a35187c5020a644f642a1ef

# %files `containers/requirements.txt /opt/requirements.txt`. Fetched from the pinned commit
# and checked against its sha256, so the file is upstream's byte for byte.
ADD --chmod=644 \
    --checksum=sha256:c5b69dd84fc4849797992e9bdf942ba2d86ec87618b0dd6f3b2f3596a2057d45 \
    https://raw.githubusercontent.com/aisa-group/ResearchArena/b922f4b94c9a31a8124e2ed32752f073ab583366/containers/requirements.txt \
    /opt/requirements.txt

# %labels
LABEL Author="ResearchArena" \
      Description="Unified container for AI R&D control evaluations" \
      Version="v1.3" \
      org.opencontainers.image.source="https://github.com/archit-manek/researcharena-inspect" \
      org.opencontainers.image.description="ResearchArena's arena.def (b922f4b) as a Docker image" \
      org.opencontainers.image.licenses="MIT"

# %post. Its exports (DEBIAN_FRONTEND, TZ, CUDA_HOME, TMPDIR, PATH) last only for the build,
# as in Apptainer, so they are set per RUN rather than with ENV.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN set -eux; \
    chmod 1777 /tmp; \
    export DEBIAN_FRONTEND=noninteractive TZ=UTC CUDA_HOME=/usr/local/cuda; \
    mkdir -p /var/tmp; \
    export TMPDIR=/var/tmp; \
    apt-get update && apt-get install -y \
        software-properties-common \
        git wget curl build-essential \
        ca-certificates gnupg; \
    add-apt-repository -y ppa:deadsnakes/ppa; \
    apt-get update && apt-get install -y \
        python3.11 python3.11-dev \
        && rm -rf /var/lib/apt/lists/*; \
    ln -sf /usr/bin/python3.11 /usr/bin/python3; \
    ln -sf /usr/bin/python3.11 /usr/bin/python; \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -; \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*; \
    curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh

# arena.def installs every Python package in one command. One layer of that size is past what
# a registry accepts in one upload, so the same packages go in over three layers: the PyTorch
# stack, then vLLM, each constrained to requirements.txt's pins (flash-attn's direct URL is
# left out of the constraints), then arena.def's own command, which adds the rest. Every
# package is pinned, so the result is the same set of versions; the image check
# (environment/image_check.py) compares the installed versions with requirements.txt.
RUN set -eux; \
    export TMPDIR=/var/tmp PATH="/root/.local/bin:$PATH"; \
    grep -v '^flash-attn @' /opt/requirements.txt > /var/tmp/constraints.txt; \
    uv pip install --system --no-cache --torch-backend=cu129 -c /var/tmp/constraints.txt \
        torch torchvision torchaudio; \
    rm -f /var/tmp/constraints.txt

RUN set -eux; \
    export TMPDIR=/var/tmp PATH="/root/.local/bin:$PATH"; \
    grep -v '^flash-attn @' /opt/requirements.txt > /var/tmp/constraints.txt; \
    uv pip install --system --no-cache --torch-backend=cu129 -c /var/tmp/constraints.txt \
        vllm flashinfer-python flashinfer-cubin; \
    rm -f /var/tmp/constraints.txt

# ===== Python dependencies ===== (arena.def's command, unchanged)
RUN set -eux; \
    export TMPDIR=/var/tmp PATH="/root/.local/bin:$PATH"; \
    uv pip install --system --no-cache --torch-backend=cu129 -r /opt/requirements.txt; \
    uv pip check --system

# ===== Agent CLI tools (npm) =====
RUN set -eux; \
    export TMPDIR=/var/tmp; \
    npm install -g \
        @anthropic-ai/claude-code@2.1.34 \
        @openai/codex@0.166.0 \
        @google/gemini-cli@0.18.4 \
        opencode-ai@1.14.29

# ===== Verify installations =====
RUN set -eux; \
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"; \
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"; \
    python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"; \
    python3 -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')"; \
    claude --version; \
    codex --version; \
    gemini --version; \
    opencode --version

# %environment: set when the container runs. `$PATH` is the base image's PATH, as it is when
# Apptainer sources %environment over the converted Docker environment.
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:/usr/local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    NO_PROXY=localhost,127.0.0.1 \
    no_proxy=localhost,127.0.0.1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# Not reproduced from arena.def:
# - `Bootstrap: docker`: Apptainer's way of naming the base image; FROM above.
# - %help ("Use the --nv flag ..."): Docker has no help section. The sandbox configs ask for
#   the GPU instead of `--nv`.
# - %runscript `exec python3 "$@"`: upstream only uses `apptainer exec`, never `run`, and a
#   sandbox container must stay up (the configs run `tail -f /dev/null`), so the base image's
#   entrypoint is kept.
# Added: the three org.opencontainers.image.* labels, which link the package to this repo.
