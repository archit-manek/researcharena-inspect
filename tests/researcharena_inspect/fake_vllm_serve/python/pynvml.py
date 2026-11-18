"""A stand-in for `pynvml` (tests only): one device whose memory is free, so that upstream's
`_wait_for_gpu_memory_release` returns at once on a CPU. `FAKE_NVML_USED_MIB` makes that much of
it used, for a server whose memory is never released."""

import os


class _Memory:
    used = int(os.environ.get("FAKE_NVML_USED_MIB", "0")) * 1024**2
    total = 80 * 1024**3


def nvmlInit() -> None:  # noqa: N802 (pynvml's names)
    pass


def nvmlShutdown() -> None:  # noqa: N802 (pynvml's names)
    pass


def nvmlDeviceGetHandleByIndex(index: int) -> int:  # noqa: N802 (pynvml's names)
    return index


def nvmlDeviceGetMemoryInfo(handle: int) -> _Memory:  # noqa: N802 (pynvml's names)
    return _Memory()
