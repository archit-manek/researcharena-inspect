"""Put one pinned Hugging Face model into a Hugging Face cache, in the sandbox.

This file is copied into the sandbox and run there with the image's Python; it is not imported by
the package (see `environment.model_cache`). Standard library only, apart from `huggingface_hub`
for a download and the image's `datasets` for a dataset's prepared builds (D-187).

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

A dataset's `builds` (config, split, records) are then put in `<hf-home>/datasets/`: copied
from `<seed>/datasets/` or made by `load_dataset`, and each split loaded by name with no network
must give its record count (`prepare_builds`).

Every pinned file is then checked: present in the snapshot, the pinned size, and the pinned
SHA-256 of its bytes (always hashed: a blob's name is not trusted, round A of the review). A file
pinned by its git object id instead (`git_oid`: the Hub hides a gated repository's SHA-256s from
anonymous callers, D-211) must have that id, either as its bytes' git blob or as the git LFS
pointer that names its SHA-256 and size; its SHA-256 is then reported (`hashed`). The last line
printed is `MODEL_CACHE_RESULT=<json>`; the exit status is 0 only when the check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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
        blob = blobs / (entry.get("sha256") or ".none")
        if not blob.is_file() or blob.stat().st_size != entry["size"]:
            blob, size = copy_hashed(snapshot / rel, blobs)
            copied += size
        link = target / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
    return copied


def download(
    repo_id: str, revision: str, hub: Path, repo_type: str = "model", paths: list[str] | None = None
) -> None:
    """Download the pinned revision's pinned files (`paths`).

    Only those: a repository's other files (a GGUF copy, an original checkpoint) are not fetched
    (D-211).
    """
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(hub),
        repo_type=repo_type,
        allow_patterns=paths,
    )


def git_blob_id(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfs_pointer_id(sha256: str, size: int) -> str:
    """The git object id of the LFS pointer git keeps for a file of this SHA-256 and size."""
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256}\nsize {size}\n"
    data = pointer.encode()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify(
    snapshot: Path, files: list[dict[str, Any]], hashed: dict[str, str] | None = None
) -> list[str]:
    """Problems with the snapshot against the pinned files; empty when it matches.

    `hashed`, if given, receives the SHA-256 of each file pinned without one (D-211).
    """
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
        actual = sha256_of(path)
        pinned, oid = entry.get("sha256"), entry.get("git_oid")
        if not (pinned or oid):
            problems.append(f"{entry['path']}: no hash pinned")
        if pinned and actual != pinned:
            problems.append(f"{entry['path']}: sha256 {actual}, pinned {pinned}")
        # The pointer first: a weight file is not read a second time when it matches.
        if oid and oid != lfs_pointer_id(actual, size) and oid != git_blob_id(path):
            problems.append(f"{entry['path']}: not the git object {oid}")
        if not pinned and hashed is not None:
            hashed[entry["path"]] = actual
    return problems


# Loads each [repo, config, split, revision] with the image's `datasets` and prints the record
# counts (or the error) as one JSON line.
LOAD = """
import json, sys
import datasets
out = []
for repo, config, split, revision in json.loads(sys.argv[1]):
    try:
        out.append(len(datasets.load_dataset(repo, config, split=split, revision=revision)))
    except Exception as error:
        out.append(f"{type(error).__name__}: {error}"[:300])
print(json.dumps(out))
"""


def build_folder(repo_id: str, revision: str, config: str | None) -> Path:
    """Where `datasets` keeps a prepared build: `datasets/<org>___<name>/<config>/0.0.0/<rev>`."""
    return Path("datasets", repo_id.replace("/", "___"), config or "default", "0.0.0", revision)


def load_builds(hf_home: Path, loads: list[list[Any]], offline: bool) -> list[Any]:
    env = {**os.environ, "HF_HOME": str(hf_home)}
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        env.pop(name, None)
        if offline:
            env[name] = "1"
    run = subprocess.run(
        [sys.executable, "-c", LOAD, json.dumps(loads)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(run.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return [f"loading failed: {run.stderr[-300:]}"] * len(loads)


def prepare_builds(
    spec: dict[str, Any], hf_home: Path, seed: Path | None, source: str
) -> list[str]:
    """Put a dataset's prepared builds in place, then load each by name with no network (D-187).

    From the seed (`<seed>/datasets/...`, copied) or, for a download, by `load_dataset` at the
    pinned revision, as upstream's `download_resources.py` does. Checked by the record count of
    each split loaded offline and by the build's folder, which names the pinned revision; the
    files `datasets` writes are not pinned by hash.
    """
    repo, revision, builds = spec["repo_id"], spec["revision"], spec.get("builds") or []
    if source == "seed" and seed is not None:
        for config in {b[0] for b in builds}:
            found = seed / build_folder(repo, revision, config)
            if not found.is_dir():
                command = "python -m researcharena_inspect.environment.model_cache <seed>"
                return [f"the seed has no prepared build {found}: run `{command}` (D-187)"]
            shutil.copytree(
                found, hf_home / build_folder(repo, revision, config), dirs_exist_ok=True
            )
    elif builds:
        made = load_builds(hf_home, [[repo, c, s, revision] for c, s, _ in builds], offline=False)
        if errors := [str(n) for n in made if not isinstance(n, int)]:
            return [f"preparing {repo}: {e}" for e in errors]
    counts = load_builds(hf_home, [[repo, c, s, None] for c, s, _ in builds], offline=True)
    problems = []
    for (config, split, rows), got in zip(builds, counts, strict=True):
        if got != rows or not (hf_home / build_folder(repo, revision, config)).is_dir():
            problems.append(
                f"{repo} {config or 'default'}/{split} at {revision}: {got}, pinned {rows}"
            )
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
        download(repo_id, revision, hub, repo_type, [f["path"] for f in files])
    # `from_pretrained("<repo>")` offline resolves "main" through refs/main; a download by
    # commit hash does not write it.
    refs = dest_model / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    result["seconds"] = round(time.monotonic() - start, 1)
    hashed: dict[str, str] = {}
    result["problems"] = verify(dest_model / "snapshots" / revision, files, hashed)
    if hashed:  # D-211
        result["hashed"] = hashed
    if spec.get("builds") and not result["problems"]:  # D-187
        result["problems"] = prepare_builds(spec, hf_home, seed, result["source"])
        result["builds"] = [list(b) for b in spec["builds"]]
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
