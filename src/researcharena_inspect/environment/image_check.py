"""Check a built image against upstream's `containers/requirements.txt` (pass rule 1).

Runs inside the image with the image's own Python and the standard library only:

    docker run --rm --entrypoint python3 -v "$PWD/image_check.py:/check.py:ro" IMAGE /check.py

It prints one JSON report: the Python version, the CUDA toolkit version (`nvcc` and
`/usr/local/cuda/version.json`), PyTorch's own CUDA version, every pin in
`/opt/requirements.txt` with the version installed, and the packages installed that the file
does not pin. It exits 1 if a pin is missing or installed at another version.
"""

import argparse
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REQUIREMENTS = Path("/opt/requirements.txt")
CUDA_VERSION_JSON = Path("/usr/local/cuda/version.json")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;#]+)")
_URL = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*@\s*(\S+)")
# flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl -> 2.8.3+cu128torch2.10
_WHEEL_VERSION = re.compile(r"/[^/]+?-([0-9][^-]*)-[^-]+-[^-]+-[^-]+\.whl$")


def normalize(name: str) -> str:
    """PEP 503 name normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(text: str) -> dict[str, str]:
    """Pinned name -> version. A direct wheel URL pins the version in its file name.

    A wheel's file name may carry a local tag its metadata leaves out (flash-attn's
    "2.8.3+cu128torch2.10"); `compare` accepts either form for URL pins, which are marked by
    a leading "@".
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


def compare(pins: dict[str, str], installed: dict[str, str]) -> dict[str, object]:
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


def installed_distributions() -> dict[str, str]:
    """Every distribution the image's Python sees, by normalised name."""
    return {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError as error:
        return f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requirements", type=Path, default=REQUIREMENTS)
    args = parser.parse_args()

    pins = parse_requirements(args.requirements.read_text(encoding="utf-8"))
    result = compare(pins, installed_distributions())
    try:
        import torch

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
        "pinned_packages": len(pins),
        **result,
    }
    print(json.dumps(report, indent=2))
    return 1 if result["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
