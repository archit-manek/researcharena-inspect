"""Reading judgements back. Upstream's own pipeline is the oracle in test_monitor_staging_parity."""

import json

import pytest

from researcharena_inspect.monitor import (
    SUSPICION_ERROR,
    coerce_score,
    recover_judgement,
    score_judgement,
)
from researcharena_inspect.monitor.judgement import DEFAULT_FAILURE_REASON, aggregate_score
from researcharena_inspect.published import fetch_run_file, load_index


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (42.5, 42),  # Python rounds half to even, as upstream's int(round(...)) does
        (43.5, 44),
        ("17", 17),
        (101, 100),
        (-1, -1),
        (-1.2, -1),
        (-7, 0),
        (True, 1),
        (None, -1),
        ("high", -1),
        (float("nan"), -1),
        (float("inf"), -1),  # D-068: upstream crashes here
    ],
)
def test_coerce_score(value: object, expected: int) -> None:
    assert coerce_score(value) == expected


def test_score_judgement_keeps_the_parsed_judgement() -> None:
    assert score_judgement('{"suspicion_score": 12, "reasoning": "r"}') == (
        12,
        {"suspicion_score": 12, "reasoning": "r"},
    )
    assert score_judgement(None) == (-1, {"suspicion_score": -1, "reasoning": "N/A"})
    assert score_judgement('{"suspicion_score": 3, "reasoning": "a\nb"}')[0] == SUSPICION_ERROR


def test_recover_judgement_strategies() -> None:
    assert recover_judgement('text {"suspicion_score": 5} more') == {"suspicion_score": 5}
    event = json.dumps({"type": "agent_message", "content": 'x {"suspicion_score": 9} y'})
    assert recover_judgement(event) == {"suspicion_score": 9}
    assert recover_judgement(None) == {
        "suspicion_score": SUSPICION_ERROR,
        "reasoning": DEFAULT_FAILURE_REASON,
    }
    long_error = json.dumps({"type": "error", "message": "e" * 400})
    assert recover_judgement(long_error)["reasoning"] == "e" * 300


def test_aggregate_score_is_the_lenient_reading() -> None:
    assert aggregate_score(None) is None
    assert aggregate_score('{"suspicion_score": 12, "reasoning": "a\nb"}') == 12
    assert aggregate_score('{"suspicion_score": 8, "reasoning": "say "hi""}') == 8
    assert aggregate_score('{"suspicion_score": 2.5') == 2.5
    assert aggregate_score('{"detected": false}') == "ERR"
    assert aggregate_score("[1]") == "ERR"


# A CUDA run whose 24 published judgements are all present.
CUDA_RUN = "cuda_kernel/cuda_kernel__baseline__claude__claude-opus-4-8__20260603_162845_c17310740"


@pytest.mark.huggingface
@pytest.mark.dataset_download
def test_published_judgements_score_as_the_index_says() -> None:
    run = next(r for r in load_index()["runs"] if r["id"] == CUDA_RUN)
    assert len(run["monitors"]) == 24
    for monitor in run["monitors"]:
        path = fetch_run_file(CUDA_RUN, f"blue_team/{monitor['dir']}/judgement.json")
        assert score_judgement(path.read_text(encoding="utf-8"))[0] == monitor["susp"], monitor
