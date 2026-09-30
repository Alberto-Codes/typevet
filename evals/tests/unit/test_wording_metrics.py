"""Unit checks for the held-out wording metrics, bootstrap and pass rule (#309).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_metrics.py
    ```

See Also:
    - [typevet_evals.wording.metrics][]: the metrics module
"""

from __future__ import annotations

import hashlib
import math

import pytest

from typevet_evals.wording.metrics import (
    Interval,
    WordingMetrics,
    paired_bootstrap,
    pass_verdict,
    percentile_interval,
    resample_indices,
    wording_metrics,
)

pytestmark = pytest.mark.unit


def _metrics(*, accuracy: float, brier: float, ece: float) -> WordingMetrics:
    """Return metrics over 158 rows with the given values."""
    return WordingMetrics(
        rows=158, positives=36, accuracy=accuracy, brier=brier, ece=ece
    )


SEED = _metrics(accuracy=0.90, brier=0.12, ece=0.158)


def test_metrics_match_hand_computed_values() -> None:
    probabilities = [0.9, 0.6, 0.2, 0.5]
    labels = [1, 0, 0, 1]

    metrics = wording_metrics(probabilities, labels)

    assert metrics.rows == 4
    assert metrics.positives == 2
    # 0.5 reads scam, so rows 0, 2 and 3 are right and row 1 is wrong.
    assert metrics.accuracy == pytest.approx(0.75)
    brier = (0.01 + 0.36 + 0.04 + 0.25) / 4
    assert metrics.brier == pytest.approx(brier)
    # Bins: 0.2 -> [0.2, 0.3), 0.5 -> [0.5, 0.6), 0.6 -> [0.6, 0.7), 0.9 -> last.
    ece = (abs(0.2 - 0) + abs(0.5 - 1) + abs(0.6 - 0) + abs(0.9 - 1)) / 4
    assert metrics.ece == pytest.approx(ece)


def test_metrics_share_one_bin_for_close_probabilities() -> None:
    metrics = wording_metrics([0.81, 0.89], [1, 0])

    assert metrics.ece == pytest.approx(abs(0.85 - 0.5))


def test_metrics_map_to_a_json_mapping() -> None:
    metrics = wording_metrics([1.0], [1])

    assert metrics.to_mapping() == {
        "rows": 1,
        "positives": 1,
        "accuracy": 1.0,
        "brier": 0.0,
        "ece": 0.0,
    }


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([], [], "empty"),
        ([0.5], [1, 0], "length"),
        ([0.5], [2], "label"),
        ([1.5], [1], "between 0 and 1"),
        ([math.nan], [1], "between 0 and 1"),
    ],
)
def test_metrics_refuse_bad_input(
    probabilities: list[float], labels: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        wording_metrics(probabilities, labels)


def test_resample_indices_follow_the_documented_hash() -> None:
    expected = [
        int.from_bytes(hashlib.sha256(f"0:3:{i}".encode()).digest()[:8], "big") % 7
        for i in range(7)
    ]

    assert resample_indices(7, seed=0, resample=3) == expected


def test_resample_indices_change_with_the_seed() -> None:
    assert resample_indices(50, seed=0, resample=0) != resample_indices(
        50, seed=1, resample=0
    )


def test_percentile_interval_takes_the_documented_ranks() -> None:
    assert percentile_interval([float(v) for v in range(2000)], level=0.95) == Interval(
        50.0, 1949.0
    )
    assert percentile_interval(
        [float(v) for v in reversed(range(20))], level=0.9
    ) == Interval(1.0, 18.0)


def test_percentile_interval_refuses_no_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile_interval([], level=0.95)


def test_bootstrap_of_a_constant_difference_is_that_difference() -> None:
    boot = paired_bootstrap([0.5, 0.5, 0.5], [1.0, 0.0, 1.0], [1, 0, 1])

    assert boot.resamples == 2000
    assert boot.seed == 0
    assert boot.brier_difference == Interval(-0.25, -0.25)


def test_bootstrap_matches_a_hand_counted_small_case() -> None:
    # Seed wording says 0.5 for both rows; evolved is exact. The Brier
    # difference is -0.25 on every resample. The seed ECE is 0.5 when a
    # resample holds one row twice and 0 when it holds both rows.
    resamples = 20
    ece_differences = sorted(
        -0.5 if len(set(resample_indices(2, seed=0, resample=b))) == 1 else 0.0
        for b in range(resamples)
    )
    expected = percentile_interval(ece_differences, level=0.9)

    boot = paired_bootstrap(
        [0.5, 0.5], [1.0, 0.0], [1, 0], resamples=resamples, level=0.9
    )

    assert boot.ece_difference == expected
    assert boot.brier_difference == Interval(-0.25, -0.25)
    assert boot.to_mapping()["ece_difference"] == {
        "low": expected.low,
        "high": expected.high,
    }


def test_bootstrap_is_seeded() -> None:
    seed = [0.1, 0.4, 0.8, 0.3, 0.9, 0.2]
    evolved = [0.2, 0.3, 0.9, 0.1, 0.7, 0.4]
    labels = [0, 0, 1, 0, 1, 1]

    first = paired_bootstrap(seed, evolved, labels, resamples=200)
    again = paired_bootstrap(seed, evolved, labels, resamples=200)
    other = paired_bootstrap(seed, evolved, labels, resamples=200, seed=1)

    assert first == again
    assert first != other


def test_bootstrap_refuses_unpaired_rows() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap([0.5], [0.5, 0.5], [1])


def test_verdict_passes_when_every_clause_holds() -> None:
    evolved = _metrics(accuracy=0.90, brier=0.10, ece=0.08)

    verdict = pass_verdict(SEED, evolved)

    assert verdict.passed
    assert verdict.ece_drop == pytest.approx(0.078)
    assert verdict.to_mapping()["passed"] is True


def test_verdict_accepts_every_boundary_value() -> None:
    seed = _metrics(accuracy=0.90, brier=0.12, ece=0.13)
    evolved = _metrics(accuracy=0.89, brier=0.1199, ece=0.10)

    verdict = pass_verdict(seed, evolved)

    assert verdict.ece_drop_ok
    assert verdict.evolved_ece_ok
    assert verdict.brier_drops
    assert verdict.accuracy_ok
    assert verdict.passed


@pytest.mark.parametrize(
    ("seed_ece", "evolved", "failed_clause"),
    [
        (0.12, _metrics(accuracy=0.90, brier=0.10, ece=0.0901), "ece_drop_ok"),
        (0.20, _metrics(accuracy=0.90, brier=0.10, ece=0.1001), "evolved_ece_ok"),
        (0.158, _metrics(accuracy=0.90, brier=0.12, ece=0.08), "brier_drops"),
        (0.158, _metrics(accuracy=0.8899, brier=0.10, ece=0.08), "accuracy_ok"),
    ],
)
def test_verdict_fails_when_one_clause_fails(
    seed_ece: float, evolved: WordingMetrics, failed_clause: str
) -> None:
    seed = _metrics(accuracy=0.90, brier=0.12, ece=seed_ece)

    verdict = pass_verdict(seed, evolved)

    clauses = {
        name: getattr(verdict, name)
        for name in ("ece_drop_ok", "evolved_ece_ok", "brier_drops", "accuracy_ok")
    }
    assert clauses.pop(failed_clause) is False
    assert all(clauses.values())
    assert not verdict.passed


def test_verdict_counts_an_accuracy_gain_as_no_drop() -> None:
    evolved = _metrics(accuracy=0.95, brier=0.10, ece=0.08)

    verdict = pass_verdict(SEED, evolved)

    assert verdict.accuracy_drop == pytest.approx(-0.05)
    assert verdict.accuracy_ok
