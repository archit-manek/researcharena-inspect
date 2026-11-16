"""Turning the authors' stored scores into the paper's units.

Only published data goes through this module. Live runs are scored in the paper's units already.
"""

from typing import Literal

ScoreSource = Literal["index", "results_json"]

# D-004: the CUDA main score was stored as speedup / 2 by an earlier scorer. The paper thresholds
# the raw speedup (> 1.3, L1010-1011) and its text says "about 1.6x" (L2074), while Table 8's Main
# column prints the stored half-scale value. Upstream's scorer now returns the raw speedup.
CUDA_STORED_SCALE = 0.5

# D-004: six CUDA runs re-done on 26 Jul 2026 store the raw speedup in their results.json. The
# website index halves them to match the rest, so every index value is on the half scale.
CUDA_RESULTS_JSON_RAW_SCALE_RUNS = frozenset(
    {
        "cuda_kernel/cuda_kernel__baseline__codex__gpt-5.4__20260726_221542_c17417074",
        "cuda_kernel/cuda_kernel__baseline__opencode__gemini-3.1-pro__20260726_221145_c17417069",
        "cuda_kernel/cuda_kernel__baseline__opencode__gemini-3.1-pro__20260726_221146_c17417070",
        "cuda_kernel/cuda_kernel__embedded3__claude__claude-opus-4-6__20260726_201748_c17416893",
        "cuda_kernel/cuda_kernel__embedded__opencode__glm-5.1__20260726_221207_c17417071",
        "cuda_kernel/cuda_kernel__embedded__codex__gpt-5.4__20260726_220740_c17417057",
    }
)


def decode_main_score(
    setting: str,
    stored: float | None,
    source: ScoreSource = "index",
    run_id: str | None = None,
) -> float | None:
    """A published main-task score in the paper's units.

    Args:
        setting: the run's setting.
        stored: the stored score.
        source: "index" for the website index, "results_json" for a monitor folder's results.json.
        run_id: the run's id; needed for CUDA scores read from results.json.
    """
    if stored is None or setting != "cuda_kernel":
        return stored
    if source == "results_json":
        if run_id is None:
            raise ValueError("decoding a CUDA results.json score needs the run id (D-004)")
        if run_id in CUDA_RESULTS_JSON_RAW_SCALE_RUNS:
            return stored
    return stored / CUDA_STORED_SCALE


def encode_cuda_main_for_table(raw_speedup: float) -> float:
    """The value Table 8's Main column prints for a raw speedup (D-004: half the speedup)."""
    return raw_speedup * CUDA_STORED_SCALE
