"""Unit tests for the repeat-run variance analysis of the checks slice (#357).

The pre-registration fixes the statistics before the result is read. These
tests prove each rule on small synthetic receipts: a run with a different
code-path digest leaves the statistics, one disagreeing case lowers the
agreement share, one perturbed probability sets the largest difference, and
the bootstrap interval is deterministic under seed 0.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from typevet_evals.check_match import (
    REPEAT_RESAMPLES,
    REPEAT_SEED,
    STABILITY_RANGE_LIMIT,
    case_agreement,
    digest_outliers,
    max_probability_difference,
    pooled_accuracy_interval,
    repeat_summary,
    stability_verdict,
)
from typevet_evals.wording.metrics import percentile_interval, resample_indices

pytestmark = pytest.mark.unit

_DIGESTS = {"runner": "aaaa", "metrics": "bbbb"}
_VERDICTS = ("consistent", "payee_mismatch", "unsigned", "consistent")
_CORRECT = (True, True, False, True)


def _case(index: int, verdict: str, correct: bool) -> dict[str, Any]:
    return {
        "case_id": f"r{index:02d}:v",
        "verdict": verdict,
        "correct": correct,
        "verdict_probabilities": {"consistent": 0.7, "unsigned": 0.3},
        "legibility_probabilities": {"0": 0.1, "4": 0.9},
        "payee_confidence": 0.9,
        "amounts_confidence": 0.8,
    }


def _receipt(
    *,
    digests: Mapping[str, str] = _DIGESTS,
    verdicts: Sequence[str] = _VERDICTS,
    correct: Sequence[bool] = _CORRECT,
    ece: float = 0.02,
) -> dict[str, Any]:
    cases = [
        _case(i, v, c) for i, (v, c) in enumerate(zip(verdicts, correct, strict=True))
    ]
    return {
        "identity": {"baseline_commit": "c0ffee", "code_path_digests": dict(digests)},
        "wall_seconds": 10.0,
        "metrics": {
            "accuracy": sum(correct) / len(correct),
            "false_clear_rate": 0.0,
            "nouls": {"payee_matches": {"ece": ece}, "amounts_match": {"ece": 0.0}},
        },
        "cases": cases,
    }


def _write(tmp_path: Path, receipts: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths = []
    for k, receipt in enumerate(receipts):
        path = tmp_path / f"run{k}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        paths.append(path)
    return paths


def test_a_run_with_a_different_code_path_digest_is_excluded(
    tmp_path: Path,
) -> None:
    odd = _receipt(digests={"runner": "zzzz", "metrics": "bbbb"}, ece=0.5)
    paths = _write(tmp_path, [odd, _receipt(), _receipt(), _receipt()])

    summary = repeat_summary(paths)

    assert summary["included"] == ["run1", "run2", "run3"]
    assert summary["excluded"] == [{"run": "run0", "differing_digests": ["runner"]}]
    assert "zzzz" not in json.dumps(summary["excluded"])
    assert summary["statistics"]["ece"]["payee_matches"]["max"] == 0.02
    assert summary["runs"][0]["included"] is False
    assert summary["runs"][0]["accuracy"] == 0.75


def test_digest_outliers_names_only_the_odd_runs_and_keys() -> None:
    receipts = [
        _receipt(),
        _receipt(digests={"runner": "aaaa", "metrics": "cccc"}),
        _receipt(),
    ]

    assert digest_outliers(receipts) == {1: ["metrics"]}
    assert digest_outliers([_receipt(), _receipt()]) == {}


@pytest.mark.parametrize("index", [0, 2, 3])
def test_one_disagreeing_case_lowers_agreement_by_one_case(index: int) -> None:
    other = list(_VERDICTS)
    other[index] = "cannot_tell"
    receipts = [_receipt(), _receipt(verdicts=other), _receipt()]

    assert case_agreement(receipts) == 0.75
    assert case_agreement([_receipt(), _receipt()]) == 1.0


def test_first_and_last_disagreeing_cases_halve_agreement() -> None:
    other = ("cannot_tell", "payee_mismatch", "unsigned", "cannot_tell")

    assert case_agreement([_receipt(), _receipt(verdicts=other)]) == 0.5


def test_one_perturbed_probability_sets_the_largest_difference() -> None:
    moved = _receipt()
    moved["cases"][2]["legibility_probabilities"]["4"] = 0.875

    assert max_probability_difference([_receipt(), _receipt()]) == 0.0
    assert max_probability_difference([_receipt(), moved]) == pytest.approx(0.025)


def test_cases_must_share_their_ids_across_runs() -> None:
    short = _receipt()
    short["cases"] = short["cases"][:3]

    with pytest.raises(ValueError, match="case ids differ"):
        case_agreement([_receipt(), short])


def test_bootstrap_interval_is_deterministic_under_seed_zero() -> None:
    receipts = [_receipt(), _receipt(correct=(True, False, False, True))]

    first = pooled_accuracy_interval(receipts)
    second = pooled_accuracy_interval(receipts)

    assert first == second
    assert first["resamples"] == REPEAT_RESAMPLES == 10_000
    assert first["seed"] == REPEAT_SEED == 0
    assert first["rows"] == 8
    assert 0.0 <= first["low"] < 0.625 < first["high"] <= 1.0
    rows = [1, 1, 0, 1, 1, 0, 0, 1]
    shares = [
        math.fsum(rows[i] for i in resample_indices(8, seed=0, resample=b)) / 8
        for b in range(REPEAT_RESAMPLES)
    ]
    expected = percentile_interval(shares, level=0.95)
    assert (first["low"], first["high"]) == (expected.low, expected.high)


def test_bootstrap_interval_collapses_when_every_row_is_correct() -> None:
    receipts = [_receipt(correct=(True,) * 4)] * 2

    interval = pooled_accuracy_interval(receipts)

    assert (interval["low"], interval["high"]) == (1.0, 1.0)


@pytest.mark.parametrize(
    ("accuracy_range", "ece_ranges", "stable"),
    [
        (0.0, {"a": 0.0}, True),
        (0.0099, {"a": 0.0099}, True),
        (0.01, {"a": 0.0}, False),
        (0.0, {"a": 0.0, "b": 0.01}, False),
    ],
)
def test_stability_rule_needs_every_range_below_the_limit(
    accuracy_range: float, ece_ranges: dict[str, float], stable: bool
) -> None:
    assert STABILITY_RANGE_LIMIT == 0.01
    assert stability_verdict(accuracy_range, ece_ranges) is stable


def test_summary_gives_spread_rows_and_the_verdict(tmp_path: Path) -> None:
    worse = _receipt(correct=(True, False, False, True), ece=0.04)
    paths = _write(tmp_path, [_receipt(), worse])

    summary = repeat_summary(paths)

    accuracy = summary["statistics"]["accuracy"]
    assert accuracy["mean"] == pytest.approx(0.625)
    assert accuracy["sd"] == pytest.approx(math.sqrt(0.125**2 * 2))
    assert (accuracy["min"], accuracy["max"]) == (0.5, 0.75)
    assert accuracy["range"] == pytest.approx(0.25)
    assert summary["stable"] is False
    assert summary["agreement"] == 1.0
    run = summary["runs"][1]
    assert (run["run"], run["commit"], run["wall_seconds"]) == ("run1", "c0ffee", 10.0)
    assert run["ece"] == {"payee_matches": 0.04, "amounts_match": 0.0}
    assert (run["agreement"], run["max_abs_difference"]) == (1.0, 0.0)


def test_summary_refuses_an_empty_path_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one receipt"):
        repeat_summary([])
