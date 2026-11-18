"""A stand-in for the two `torch` calls upstream's capabilities evaluator makes when it stops its
server (tests only): no CUDA device, nothing to empty."""


class cuda:  # noqa: N801 (torch's name)
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        pass
