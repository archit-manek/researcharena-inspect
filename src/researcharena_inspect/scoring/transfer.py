"""Copying large files between the sample's sandbox and the host, in bounded pieces.

Inspect's `read_file` and `write_file` move a whole file at once, and `read_file` refuses one
over 100 MiB (`INSPECT_SANDBOX_MAX_READ_FILE_SIZE`). A post-training submission's weights are
about 1 GB. These helpers move a file in pieces of at most `CHUNK_BYTES`, so memory on both sides
stays bounded, every exec has a time limit, and every file is checked by its SHA-256 at the end:
the sandbox's own hash (`list_tree`) against the bytes that arrived. Nothing here follows a
symbolic link.

Used by the submission store (`submission_store`, sandbox to host) and the replay agent
(`agents.replay`, host to sandbox).
"""

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Final

from inspect_ai.util import SandboxEnvironment

CHUNK_BYTES: Final = 64 * 1024 * 1024
"""Largest piece read or written at once; under Inspect's default 100 MiB read limit."""

EXEC_TIMEOUT_S: Final = 600
"""Limit on one exec (one piece, or hashing a listing)."""

LIST_TIMEOUT_S: Final = 3600
"""Limit on listing and hashing a whole folder (a few GB at about 1 GB/s)."""


class TransferError(RuntimeError):
    """A copy failed, or the bytes that arrived do not match the source's SHA-256."""


# Lists ROOT without following links: one JSON line per entry, sorted, folders first. Files are
# hashed here, in the sandbox, so the copy can be checked end to end. Stops after MAX entries.
_LIST = """
import hashlib, json, os, stat, sys
root, limit = sys.argv[1], int(sys.argv[2])
first = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
out = []
class Full(Exception):
    pass
def kind(mode):
    if stat.S_ISREG(mode): return "file"
    if stat.S_ISDIR(mode): return "dir"
    if stat.S_ISLNK(mode): return "link"
    return "other"
def visit(folder, rel):
    with os.scandir(folder) as it:
        names = sorted(e.name for e in it)
    for name in names:
        full = os.path.join(folder, name)
        child = name if rel == "" else rel + "/" + name
        info = os.lstat(full)
        k = kind(info.st_mode)
        entry = {"path": child, "kind": k, "mode": stat.S_IMODE(info.st_mode)}
        if k == "file":
            entry["size"] = info.st_size
        elif k == "link":
            entry["target"] = os.readlink(full)
        out.append(entry)
        if len(out) > limit:
            raise Full
        if k == "dir":
            visit(full, child)
truncated = False
try:
    if not os.path.isdir(root) or os.path.islink(root):
        print(json.dumps({"error": "not a folder: " + root})); sys.exit(0)
    visit(root, "")
except Full:
    truncated = True
    out = out[:limit]
def order(e):
    top = e["path"].split("/")[0]
    return (0 if top in first else 1, e["path"])
out.sort(key=order)
for e in out:
    if e["kind"] == "file":
        h = hashlib.sha256()
        with open(os.path.join(root, e["path"]), "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        e["sha256"] = h.hexdigest()
print(json.dumps({"entries": out, "truncated": truncated}))
"""

# Copies bytes [offset, offset + length) of SRC to DEST.
_PIECE = """
import sys
src, dest, offset, length = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "rb") as f, open(dest, "wb") as g:
    f.seek(offset)
    left = length
    while left > 0:
        block = f.read(min(left, 1 << 24))
        if not block:
            break
        g.write(block)
        left -= len(block)
"""


async def list_tree(
    box: SandboxEnvironment,
    root: str,
    *,
    max_entries: int,
    first: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], bool]:
    """Every entry under `root` (no links followed), files with size and SHA-256.

    Args:
        box: the sandbox.
        root: the folder in the sandbox.
        max_entries: stop after this many entries.
        first: top-level names to list before the others (for example `final_model`).

    Returns:
        The entries, and whether the listing stopped at `max_entries`.

    Raises:
        TransferError: the listing failed or `root` is not a folder.
    """
    result = await box.exec(
        ["python3", "-c", _LIST, root, str(max_entries), json.dumps(list(first))],
        user="root",
        timeout=LIST_TIMEOUT_S,
    )
    lines = result.stdout.strip().splitlines()
    if not result.success or not lines:
        raise TransferError(f"listing {root} failed ({result.returncode}): {result.stderr[-2000:]}")
    report = json.loads(lines[-1])
    if "error" in report:
        raise TransferError(report["error"])
    return report["entries"], bool(report["truncated"])


async def copy_out(
    box: SandboxEnvironment,
    source: str,
    dest: Path,
    *,
    size: int,
    sha256: str,
    scratch: str,
    chunk: int | None = None,
) -> None:
    """Copy one sandbox file to `dest` on the host, piece by piece, and check its SHA-256.

    `dest` is written as `<dest>.part` and renamed once the hash matches.

    Raises:
        TransferError: a piece failed or the bytes do not match `sha256`.
    """
    chunk = chunk or CHUNK_BYTES
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with part.open("wb") as handle:
            if size <= chunk:
                data = await box.read_file(source, text=False)
                digest.update(data)
                handle.write(data)
                written = len(data)
            else:
                for offset in range(0, size, chunk):
                    length = min(chunk, size - offset)
                    piece = await box.exec(
                        ["python3", "-c", _PIECE, source, scratch, str(offset), str(length)],
                        user="root",
                        timeout=EXEC_TIMEOUT_S,
                    )
                    if not piece.success:
                        raise TransferError(f"reading {source} at {offset} failed: {piece.stderr}")
                    data = await box.read_file(scratch, text=False)
                    if len(data) != length:
                        raise TransferError(
                            f"{source}: {len(data)} bytes at {offset}, expected {length}"
                        )
                    digest.update(data)
                    handle.write(data)
                    written += len(data)
                    del data
        if written != size or digest.hexdigest() != sha256:
            raise TransferError(
                f"{source}: {written} bytes, sha256 {digest.hexdigest()}; "
                f"the sandbox listed {size} bytes, {sha256}"
            )
        part.replace(dest)
    finally:
        if part.exists():
            part.unlink()


async def copy_in(
    box: SandboxEnvironment,
    source: Path,
    dest: str,
    *,
    user: str,
    scratch: str,
    chunk: int | None = None,
) -> str:
    """Copy one host file into the sandbox as `user`, piece by piece; return its SHA-256.

    The pieces are appended to `<dest>.part` as `user`, which is renamed to `dest` once the
    sandbox's hash of the whole file equals the host's. `dest`'s folder must exist and be
    writable by `user`.

    Raises:
        TransferError: a piece failed or the file in the sandbox does not match.
    """
    chunk = chunk or CHUNK_BYTES
    digest = hashlib.sha256()
    part = f"{dest}.part"
    q_part, q_scratch = shlex.quote(part), shlex.quote(scratch)
    start = await box.exec(["bash", "-c", f": > {q_part}"], user=user, timeout=EXEC_TIMEOUT_S)
    if not start.success:
        raise TransferError(f"creating {part} failed: {start.stderr}")
    with source.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
            await box.write_file(scratch, block)
            appended = await box.exec(
                ["bash", "-c", f"chmod 644 {q_scratch} && cat {q_scratch} >> {q_part}"],
                user="root",
                timeout=EXEC_TIMEOUT_S,
            )
            if not appended.success:
                raise TransferError(f"appending to {part} failed: {appended.stderr}")
            del block
    expected = digest.hexdigest()
    check = await box.exec(
        [
            "bash",
            "-c",
            f"rm -f {q_scratch}; sha256sum {q_part} | cut -d' ' -f1",
        ],
        user="root",
        timeout=EXEC_TIMEOUT_S,
    )
    actual = check.stdout.strip()
    if not check.success or actual != expected:
        await box.exec(["rm", "-f", part], user="root")
        raise TransferError(
            f"{dest}: sha256 {actual or check.stderr} in the sandbox, {expected} sent"
        )
    moved = await box.exec(["mv", "-f", part, dest], user=user, timeout=EXEC_TIMEOUT_S)
    if not moved.success:
        raise TransferError(f"renaming {part} failed: {moved.stderr}")
    return expected


__all__ = [
    "CHUNK_BYTES",
    "TransferError",
    "copy_in",
    "copy_out",
    "list_tree",
]
