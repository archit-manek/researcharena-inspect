# Side branch for the Hawk GPU probe only: the main task needs inspect-evals-utils 0.3.0,
# which Hawk refuses until it is 7 days old (18 Nov 2026).
from researcharena_inspect.gpu_probe import gpu_probe

__all__ = ["gpu_probe"]
