"""Describe a folder: every path with its type, mode, size and SHA-256, and any git repository.

Stdlib only, Python 3.8+: the tests copy this file into containers (upstream's pipeline in
Docker, and the port's sandbox) and run it there, so both sides are described by one program.

    python3 tree_manifest.py ROOT [--owner]

prints JSON: {"entries": {relpath: [...]}, "git": {repo relpath: {...}}, "pyc_code": {...}}.
Paths inside a `.git` folder are left out of "entries"; the repository is described under "git" instead, by
what does not depend on when or on which machine the commit was made.
"""

import hashlib
import json
import marshal
import os
import stat
import subprocess
import sys
import types

# Files in .git whose bytes hold a commit id, a timestamp or file stat data.
VOLATILE_GIT_FILES = ("index", "ORIG_HEAD", "FETCH_HEAD")
VOLATILE_GIT_PREFIXES = ("logs/", "refs/heads/", "objects/")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo, *args):
    result = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout
    return f"ERROR {result.returncode}: {result.stderr.strip()}"


def git_state(repo):
    gitdir = os.path.join(repo, ".git")
    files = {}
    for dirpath, dirnames, filenames in os.walk(gitdir):
        dirnames.sort()
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, gitdir).replace(os.sep, "/")
            if rel.startswith("objects/"):
                continue
            volatile = rel in VOLATILE_GIT_FILES or rel.startswith(VOLATILE_GIT_PREFIXES)
            mode = stat.S_IMODE(os.lstat(full).st_mode)
            files[rel] = [oct(mode), None if volatile else sha256(full)]
    objects = git(
        repo,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
    )
    return {
        "branch": git(repo, "symbolic-ref", "--short", "HEAD").strip(),
        "tree": git(repo, "rev-parse", "HEAD^{tree}").strip(),
        "log": git(repo, "log", "--format=%s|%an|%ae").strip().splitlines(),
        "ls_files": git(repo, "ls-files", "-s").splitlines(),
        "status": git(repo, "status", "--porcelain", "--untracked-files=all").splitlines(),
        "non_commit_objects": sorted(
            line for line in objects.splitlines() if " commit " not in line
        ),
        "gitdir_files": files,
    }


def describe(root, owner=False):
    entries = {}
    repos = {}
    oldest = None  # the oldest modification time outside .git, links aside
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if ".git" in dirnames:
            repos[rel_dir] = git_state(dirpath)
            dirnames.remove(".git")
        names = [(n, True) for n in dirnames] + [(n, False) for n in filenames]
        if rel_dir == ".":
            names = [(".", True), *names]
        for name, _is_dir in names:
            full = dirpath if name == "." else os.path.join(dirpath, name)
            rel = "." if name == "." else os.path.relpath(full, root).replace(os.sep, "/")
            info = os.lstat(full)
            if not stat.S_ISLNK(info.st_mode):
                oldest = info.st_mtime if oldest is None else min(oldest, info.st_mtime)
            mode = oct(stat.S_IMODE(info.st_mode))
            entry: list[object]
            if stat.S_ISLNK(info.st_mode):
                entry = ["link", mode, os.readlink(full)]
            elif stat.S_ISDIR(info.st_mode):
                entry = ["dir", mode]
            else:
                entry = ["file", mode, info.st_size, sha256(full)]
            if owner:
                entry.append(info.st_uid)
            entries[rel] = entry
    pyc_code = {
        rel: pyc_digest(os.path.join(root, rel))
        for rel, entry in entries.items()
        if entry[0] == "file" and rel.endswith(".pyc")
    }
    return {"entries": entries, "git": repos, "oldest_mtime": oldest, "pyc_code": pyc_code}


def _code_key(value):
    if isinstance(value, types.CodeType):
        fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_stacksize")
        fields += ("co_flags", "co_code", "co_names", "co_varnames", "co_freevars")
        fields += ("co_cellvars", "co_name", "co_firstlineno")
        key = [repr(getattr(value, name, None)) for name in fields]
        return "code(" + ",".join(key) + "," + _code_key(value.co_consts) + ")"
    if isinstance(value, tuple):
        return "(" + ",".join(_code_key(v) for v in value) + ")"
    if isinstance(value, frozenset):
        return "frozenset(" + ",".join(sorted(_code_key(v) for v in value)) + ")"
    return repr(value)


def pyc_digest(path):
    """SHA-256 of a .pyc's compiled code, without what depends on where and when it was made.

    The header holds the source's modification time, and every code object the source's
    absolute path (`co_filename`), which is the run folder's path when an evaluator imports the
    snapshot's modules. Needs the Python version that wrote the file; None otherwise.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    try:
        code = marshal.loads(data[16:])
    except (ValueError, EOFError, TypeError):
        return None
    return hashlib.sha256(_code_key(code).encode("utf-8", "surrogateescape")).hexdigest()


if __name__ == "__main__":
    print(
        json.dumps(describe(sys.argv[1], owner="--owner" in sys.argv[2:]), indent=1, sort_keys=True)
    )
