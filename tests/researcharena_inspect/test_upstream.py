import subprocess
from pathlib import Path

import pytest

from researcharena_inspect.pins import PAPER_COMMIT, UPSTREAM_COMMIT
from researcharena_inspect.upstream import (
    CACHE_ENV_VAR,
    UpstreamCheckoutError,
    cache_root,
    upstream_dir,
)

pytestmark = pytest.mark.dataset_download


@pytest.mark.parametrize("commit", [UPSTREAM_COMMIT, PAPER_COMMIT])
def test_checkout_is_at_the_pinned_commit(commit: str) -> None:
    checkout = upstream_dir(commit)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert head.stdout.strip() == commit
    assert (checkout / "src" / "prompt_builder.py").is_file()


def test_edited_checkout_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pristine = upstream_dir()
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
    assert cache_root() == tmp_path
    edited = tmp_path / UPSTREAM_COMMIT[:12]
    subprocess.run(["git", "clone", "--quiet", str(pristine), str(edited)], check=True)
    subprocess.run(
        ["git", "-C", str(edited), "checkout", "--quiet", "--detach", UPSTREAM_COMMIT], check=True
    )
    (edited / "run.sh").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(UpstreamCheckoutError, match="local edits"):
        upstream_dir()
