"""Put one pinned Hugging Face model into a Hugging Face cache, in the sandbox.

This file is copied into the sandbox and run there with the image's Python; it is not imported by
the package (see `environment.model_cache`). Standard library only, apart from `huggingface_hub`
for a download.

    python3 model_cache_script.py --spec JSON --hf-home DIR [--seed DIR] --mode auto|seed|download

`--spec` is the pinned repository (`PinnedModel.to_json()`; its `repo_type` is `model` or
`dataset`, D-170). It ends up in the layout `huggingface_hub` uses,
`<hf-home>/hub/models--<org>--<name>/{blobs,snapshots/<rev>,refs/main}` (`datasets--` for a
dataset),
with `refs/main` naming the pinned revision, so that `from_pretrained("<org>/<name>")` and vLLM
find it with `HF_HUB_OFFLINE=1`. Sources:

- `seed`: a read-only cache on the host, mounted into the sandbox, in `HF_HOME` layout
  (`<seed>/hub/models--...`) or hub layout (`<seed>/models--...`). The pinned snapshot's files
  are copied, hashed on the way, into blobs named by their SHA-256, with the snapshot's links
  rebuilt.
- `download`: `huggingface_hub.snapshot_download` at the pinned revision (needs the network).
- `auto`: `seed` if the seed holds the pinned snapshot, else `download`.

Every pinned file is then checked: present in the snapshot, the pinned size, and the pinned
SHA-256 (from the blob's name where it is one, hashed otherwise). The last line printed is
`MODEL_CACHE_RESULT=<json>`; the exit status is 0 only when the check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

RESULT_PREFIX = "MODEL_CACHE_RESULT="


def model_folder(repo_id: str, repo_type: str = "model") -> str:
    """`models--<org>--<name>`, or `datasets--...` for a dataset repository."""
    return f"{repo_type}s--" + repo_id.replace("/", "--")


def seed_snapshot(seed: Path, repo_id: str, revision: str, repo_type: str = "model") -> Path | None:
    """The pinned snapshot folder in the seed, in either layout; None if it is not there."""
    for hub in (seed / "hub", seed):
        snapshot = hub / model_folder(repo_id, repo_type) / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_hashed(source: Path, blobs: Path) -> tuple[Path, int]:
    """Copy `source` into `blobs`, named by the SHA-256 of what was copied."""
    digest = hashlib.sha256()
    tmp = blobs / f".incoming-{os.getpid()}"
    size = 0
    with source.open("rb") as src, tmp.open("wb") as dst:
        for chunk in iter(lambda: src.read(1 << 24), b""):
            digest.update(chunk)
            dst.write(chunk)
            size += len(chunk)
    blob = blobs / digest.hexdigest()
    os.replace(tmp, blob)
    return blob, size


def copy_from_seed(snapshot: Path, dest_model: Path, files: list[dict[str, Any]]) -> int:
    """Copy the pinned files from a seed snapshot into `dest_model`'s blobs and snapshot.

    Each file becomes `blobs/<sha256 of its contents>` and `snapshots/<rev>/<path>` a relative
    link to it, as a download leaves an LFS file. A blob already named by the pinned hash is
    not copied again. Returns the bytes copied.
    """
    revision = snapshot.name
    blobs = dest_model / "blobs"
    target = dest_model / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    copied = 0
    for entry in files:
        rel = entry["path"]
        blob = blobs / entry["sha256"]
        if not blob.is_file() or blob.stat().st_size != entry["size"]:
            blob, size = copy_hashed(snapshot / rel, blobs)
            copied += size
        link = target / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
    return copied


def download(repo_id: str, revision: str, hub: Path, repo_type: str = "model") -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, cache_dir=str(hub), repo_type=repo_type)


def verify(snapshot: Path, files: list[dict[str, Any]]) -> list[str]:
    """Problems with the snapshot against the pinned files; empty when it matches."""
    problems = []
    for entry in files:
        path = snapshot / entry["path"]
        if not path.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        size = path.stat().st_size
        if size != entry["size"]:
            problems.append(f"{entry['path']}: {size} bytes, pinned {entry['size']}")
            continue
        named = path.resolve().name
        actual = named if named == entry["sha256"] else sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 {actual}, pinned {entry['sha256']}")
    return problems


def provision(spec: dict[str, Any], hf_home: Path, seed: Path | None, mode: str) -> dict[str, Any]:
    repo_id, revision, files = spec["repo_id"], spec["revision"], spec["files"]
    repo_type = spec.get("repo_type", "model")
    hub = hf_home / "hub"
    dest_model = hub / model_folder(repo_id, repo_type)
    result: dict[str, Any] = {"repo_id": repo_id, "revision": revision, "mode": mode}
    if repo_type != "model":
        result["repo_type"] = repo_type
    start = time.monotonic()
    found = seed_snapshot(seed, repo_id, revision, repo_type) if seed is not None else None
    if mode == "seed" and found is None:
        result["source"] = None
        result["problems"] = [f"the seed {seed} has no snapshot {revision} of {repo_id}"]
        return result
    if found is not None and mode in ("auto", "seed"):
        result["source"] = "seed"
        result["seed_snapshot"] = str(found)
        result["bytes_copied"] = copy_from_seed(found, dest_model, files)
    else:
        result["source"] = "download"
        download(repo_id, revision, hub, repo_type)
    # `from_pretrained("<repo>")` offline resolves "main" through refs/main; a download by
    # commit hash does not write it.
    refs = dest_model / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    result["seconds"] = round(time.monotonic() - start, 1)
    result["problems"] = verify(dest_model / "snapshots" / revision, files)
    result["hub_entries"] = sorted(p.name for p in hub.iterdir())
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--seed", default="")
    parser.add_argument("--mode", choices=("auto", "seed", "download"), default="auto")
    args = parser.parse_args()
    seed = Path(args.seed) if args.seed and Path(args.seed).is_dir() else None
    try:
        result = provision(json.loads(args.spec), Path(args.hf_home), seed, args.mode)
    except Exception as error:  # reported, then a non-zero exit
        traceback.print_exc()
        result = {"problems": [f"{type(error).__name__}: {error}"]}
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if not result.get("problems") else 1


if __name__ == "__main__":
    sys.exit(main())
