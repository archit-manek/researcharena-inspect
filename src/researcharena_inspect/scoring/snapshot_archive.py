"""Archive a folder as a gzipped tar, giving up once the archive passes a size limit.

Stdlib only, Python 3.8+: the scorer copies this file into the sample's sandbox and runs it
there as root, so it can read every file whatever its mode.

    python3 snapshot_archive.py ROOT ARCNAME OUTPUT MAX_BYTES [EXCLUDE]

prints a JSON report on stdout (see `archive`). EXCLUDE, if given, is a comma-separated list of
top-level names under ROOT left out with everything below them (the report counts them). Nothing under ROOT is changed or followed:
symbolic links are archived as links, and FIFOs, sockets and device files are listed in the
report instead of archived.
"""

import gzip
import json
import os
import stat
import sys
import tarfile

LARGEST_KEPT = 20
"""How many of the largest files the report names."""


class TooLargeError(Exception):
    """The archive passed the size limit."""


class _LimitedWriter:
    """A file wrapper that raises `TooLargeError` as soon as more than `limit` bytes are written."""

    def __init__(self, handle, limit):
        self.handle = handle
        self.limit = limit
        self.written = 0

    def write(self, data):
        self.written += len(data)
        if self.written > self.limit:
            raise TooLargeError
        return self.handle.write(data)

    def flush(self):
        self.handle.flush()


def walk(root, exclude=()):
    """Every path under `root` as (relative POSIX path, lstat), sorted, each folder first.

    The root itself comes first, as "". Symbolic links to folders are not descended into.
    Top-level names in `exclude` are left out with everything below them.
    """
    out = [("", os.lstat(root))]

    def visit(folder, rel):
        with os.scandir(folder) as it:
            names = sorted(entry.name for entry in it)
        for name in names:
            if rel == "" and name in exclude:
                continue
            full = os.path.join(folder, name)
            child = name if rel == "" else rel + "/" + name
            info = os.lstat(full)
            out.append((child, info))
            if stat.S_ISDIR(info.st_mode):
                visit(full, child)

    visit(root, "")
    return out


def _kind(mode):
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISLNK(mode):
        return "link"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "char_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other"


def _add(tar, root, arcname, rel, info):
    kind = _kind(info.st_mode)
    if kind not in ("file", "dir", "link"):
        return
    full = root if rel == "" else os.path.join(root, rel)
    member = tarfile.TarInfo(arcname if rel == "" else arcname + "/" + rel)
    member.mode = stat.S_IMODE(info.st_mode)
    member.mtime = int(info.st_mtime)
    member.uid = info.st_uid
    member.gid = info.st_gid
    if kind == "dir":
        member.type = tarfile.DIRTYPE
        tar.addfile(member)
    elif kind == "link":
        member.type = tarfile.SYMTYPE
        member.linkname = os.readlink(full)
        tar.addfile(member)
    else:
        member.type = tarfile.REGTYPE
        member.size = info.st_size
        with open(full, "rb") as source:
            tar.addfile(member, source)


def _excluded(root, exclude):
    """[name, files, bytes] for each excluded top-level name that exists."""
    out = []
    for name in sorted(exclude):
        full = os.path.join(root, name)
        if not os.path.lexists(full):
            continue
        files = size = 0
        if os.path.isdir(full) and not os.path.islink(full):
            for _rel, info in walk(full):
                if stat.S_ISREG(info.st_mode):
                    files += 1
                    size += info.st_size
        elif stat.S_ISREG(os.lstat(full).st_mode):
            files, size = 1, os.lstat(full).st_size
        out.append([name, files, size])
    return out


def archive(root, arcname, output, max_bytes, exclude=()):
    """Write `root` to `output` as `arcname/...` in a gzipped PAX tar of at most `max_bytes`.

    Returns:
        The report: `status` ("kept" or "too_large"), `archive_bytes` (None when too large),
        counts of `files`, `dirs` and `links`, `raw_bytes` (the regular files' total size),
        `largest` (the largest files as [path, size]) and `skipped` (entries not archived,
        as [path, kind]), and with `exclude`, `excluded` (as [name, files, bytes]). When too
        large, `output` is removed.
    """
    entries = walk(root, exclude)
    report = {
        "status": None,
        "archive_bytes": None,
        "max_bytes": max_bytes,
        "files": 0,
        "dirs": 0,
        "links": 0,
        "raw_bytes": 0,
        "largest": [],
        "skipped": [],
    }
    if exclude:
        report["excluded"] = _excluded(root, exclude)
    sizes = []
    for rel, info in entries:
        kind = _kind(info.st_mode)
        if kind == "file":
            report["files"] += 1
            report["raw_bytes"] += info.st_size
            sizes.append((info.st_size, rel))
        elif kind == "dir":
            report["dirs"] += 1
        elif kind == "link":
            report["links"] += 1
        else:
            report["skipped"].append([rel, kind])
    sizes.sort(key=lambda item: (-item[0], item[1]))
    report["largest"] = [[rel, size] for size, rel in sizes[:LARGEST_KEPT]]

    try:
        with open(output, "wb") as handle:
            limited = _LimitedWriter(handle, max_bytes)
            # Nested, not parenthesised: the sandbox's Python may be older than 3.10.
            with gzip.GzipFile(fileobj=limited, mode="wb", compresslevel=6, mtime=0) as gz:  # noqa: SIM117
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for rel, info in entries:
                        _add(tar, root, arcname, rel, info)
    except TooLargeError:
        os.remove(output)
        report["status"] = "too_large"
        return report
    report["status"] = "kept"
    report["archive_bytes"] = os.path.getsize(output)
    return report


def main(argv):
    root, arcname, output, max_bytes = argv[1], argv[2], argv[3], int(argv[4])
    exclude = tuple(n for n in argv[5].split(",") if n) if len(argv) > 5 else ()
    if not os.path.isdir(root) or os.path.islink(root):
        print(json.dumps({"status": "error", "error": "not a folder: " + root}))
        return 1
    print(json.dumps(archive(root, arcname, output, max_bytes, exclude)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
