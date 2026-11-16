"""A read-only checkout of upstream ResearchArena at a pinned commit.

The port reuses upstream's task files and scorers unchanged, and its tests compare the
port's output with what upstream's own scripts produce. Both need the exact pinned tree.
"""

import os
import subprocess
from pathlib import Path

from platformdirs import user_cache_dir

from researcharena_inspect.pins import UPSTREAM_COMMIT, UPSTREAM_REPO_URL

# Set this to keep the checkouts somewhere else (for example inside a container image).
CACHE_ENV_VAR = "RESEARCHARENA_UPSTREAM_CACHE"


class UpstreamCheckoutError(RuntimeError):
    """The checkout is missing, at another commit, or has been edited."""


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def cache_root() -> Path:
    """Folder that holds one checkout per pinned commit."""
    override = os.environ.get(CACHE_ENV_VAR)
    return (
        Path(override) if override else Path(user_cache_dir("researcharena_inspect")) / "upstream"
    )


def upstream_dir(commit: str = UPSTREAM_COMMIT) -> Path:
    """Return upstream's tree at `commit`, cloning it on first use.

    Raises:
        UpstreamCheckoutError: the existing checkout is at another commit or has local edits.
    """
    checkout = cache_root() / commit[:12]
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--no-checkout", UPSTREAM_REPO_URL, str(checkout))
        _git("checkout", "--quiet", "--detach", commit, cwd=checkout)
    head = _git("rev-parse", "HEAD", cwd=checkout)
    if head != commit:
        raise UpstreamCheckoutError(f"{checkout} is at {head}, expected {commit}")
    if _git("status", "--porcelain", "--untracked-files=no", cwd=checkout):
        raise UpstreamCheckoutError(f"{checkout} has local edits; delete it and fetch again")
    return checkout
