"""Check a built image against upstream's `containers/requirements.txt` (pass rule 1).

Runs inside the image with its own Python and the standard library only (the image workflows
mount it with `docker run`). Prints one JSON report; exits 1 if a pin is missing or installed
at another version.
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

REQUIREMENTS = Path("/opt/requirements.txt")
CUDA_VERSION_JSON = Path("/usr/local/cuda/version.json")
# The base image's own record of its CUDA and cuDNN versions, and arena.def's %environment.
IMAGE_ENV = (
    "CUDA_VERSION",
    "NV_CUDNN_VERSION",
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "UV_LINK_MODE",
    "NO_PROXY",
    "no_proxy",
    "DEBIAN_FRONTEND",
    "PATH",
)

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
_URL = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)")
# flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl -> 2.8.3+cu128torch2.10
_WHEEL_VERSION = re.compile(r"/[^/]+?-([0-9][^-]*)-[^-]+-[^-]+-[^-]+\.whl$")


def normalize(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> dict[str, str]:
    """Pinned name -> version. A direct wheel URL pins the version in its file name.

    URL pins are marked by a leading "@": `compare` accepts them with or without the local tag
    that a wheel's file name may carry and its metadata leaves out (flash-attn's "+cu128...").
    """
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if match := _PIN.match(line):
            pins[normalize(match.group(1))] = match.group(2)
        elif match := _URL.match(line):
            wheel = _WHEEL_VERSION.search(match.group(2))
            pins[normalize(match.group(1))] = "@" + (wheel.group(1) if wheel else match.group(2))
        else:
            raise ValueError(f"unrecognised requirement line: {raw!r}")
    return pins


def compare(pins: dict[str, str], installed: dict[str, str]) -> dict[str, Any]:
    """Pins that match, differ or are missing, and installed packages nobody pinned."""
    rows = []
    for name, pin in sorted(pins.items()):
        have = installed.get(name)
        wanted = pin.removeprefix("@")
        accepted = {wanted, wanted.split("+")[0]} if pin.startswith("@") else {wanted}
        status = "missing" if have is None else ("ok" if have in accepted else "mismatch")
        rows.append({"package": name, "pinned": wanted, "installed": have, "status": status})
    unpinned = {name: v for name, v in sorted(installed.items()) if name not in pins}
    return {
        "pins": rows,
        "mismatches": [r for r in rows if r["status"] != "ok"],
        "unpinned_installed": unpinned,
    }


def installed_distributions() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Every distribution the image's Python finds, by normalised name.

    Returns the version `import` resolves and, separately, the ones it shadows: Ubuntu's own
    packages in /usr/lib/python3/dist-packages come after uv's on the path.
    """
    first: dict[str, str] = {}
    shadowed: dict[str, list[str]] = {}
    for dist in metadata.distributions():
        name = normalize(dist.metadata["Name"])
        if name in first:
            shadowed.setdefault(name, []).append(f"{dist.version} ({dist.locate_file('')})")
        else:
            first[name] = dist.version
    return first, shadowed


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError as error:
        return f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an image against requirements.txt.")
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    args = parser.parse_args()

    pins = parse_requirements(args.requirements.read_text(encoding="utf-8"))
    installed, shadowed = installed_distributions()
    result = compare(pins, installed)
    try:
        torch = importlib.import_module("torch")  # only inside the image

        torch_cuda = {"torch": torch.__version__, "torch.version.cuda": torch.version.cuda}
    except Exception as error:
        torch_cuda = {"error": f"{type(error).__name__}: {error}"}
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "nvcc": _run(["/usr/local/cuda/bin/nvcc", "--version"]),
        "cuda_version_json": json.loads(CUDA_VERSION_JSON.read_text())
        if CUDA_VERSION_JSON.is_file()
        else None,
        "torch_cuda": torch_cuda,
        "image_env": {k: os.environ.get(k) for k in IMAGE_ENV},
        "shadowed_distributions": shadowed,
        "pinned_packages": len(pins),
        **result,
    }
    print(json.dumps(report, indent=2))
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
