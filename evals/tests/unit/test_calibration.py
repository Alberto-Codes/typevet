"""Offline tests for the post-hoc calibration helpers (#343).

The synthetic series come from sha256 draws, so they are fixed without a
``random`` generator. The overconfident series draws labels from ``p_true``
and reports
``sigmoid(3 * logit(p_true))``, so a correct temperature fit finds T near 3
and a correct Platt fit finds a near 1/3 and b near 0. No test calls a model.
"""

from __future__ import annotations

import hashlib
import itertools
import math

import pytest

from typevet_evals.calibration import (
    TemperatureScaler,
    accuracy_at_half,
    bootstrap_ece_change,
    brier_score,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    in_calibration_half,
    rule_met,
    split_indices,
)
from typevet_evals.face_match.metrics import expected_calibration_error

pytestmark = pytest.mark.unit


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _uniform(seed: int, stream: str, index: int) -> float:
    digest = hashlib.sha256(f"{seed}:{stream}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _overconfident(n: int, seed: int) -> tuple[list[float], list[bool]]:
    probs: list[float] = []
    labels: list[bool] = []
    for i in range(n):
        p_true = 0.02 + 0.96 * _uniform(seed, "p", i)
        labels.append(_uniform(seed, "y", i) < p_true)
        probs.append(_sigmoid(3.0 * _logit(p_true)))
    return probs, labels


def test_split_is_deterministic_and_roughly_balanced() -> None:
    ids = [f"row-{i}" for i in range(1000)]
    first = split_indices(ids)
    assert first == split_indices(ids)
    calibration, evaluation = first
    assert sorted(calibration + evaluation) == list(range(1000))
    assert 450 <= len(calibration) <= 550
    expected = int(hashlib.sha256(b"row-7").hexdigest(), 16) % 2 == 0
    assert in_calibration_half("row-7") is expected
    assert (7 in calibration) is expected


def test_temperature_recovers_three_and_lowers_ece() -> None:
    probs, labels = _overconfident(6000, seed=1)
    scaler = fit_temperature(probs, labels)
    assert isinstance(scaler, TemperatureScaler)
    assert scaler.temperature == pytest.approx(3.0, abs=0.3)
    test_probs, test_labels = _overconfident(3000, seed=2)
    before = expected_calibration_error(test_probs, test_labels)
    after = expected_calibration_error([scaler(p) for p in test_probs], test_labels)
    assert before > 0.1
    assert after < before - 0.05


def test_platt_recovers_one_third_and_zero() -> None:
    probs, labels = _overconfident(6000, seed=3)
    scaler = fit_platt(probs, labels)
    assert scaler.slope == pytest.approx(1.0 / 3.0, abs=0.05)
    assert scaler.intercept == pytest.approx(0.0, abs=0.1)


def test_platt_stays_finite_on_separable_data() -> None:
    scaler = fit_platt([0.1, 0.2, 0.8, 0.9], [False, False, True, True])
    assert math.isfinite(scaler.slope)
    assert scaler(0.9) > 0.99
    assert scaler(0.1) < 0.01


def test_isotonic_pools_violators_and_steps_between_knots() -> None:
    mapping = fit_isotonic([0.1, 0.2, 0.3, 0.4], [False, True, False, True])
    # PAV: 0, 1, 0, 1 -> pool (1, 0) to 0.5 -> 0, 0.5, 0.5, 1.
    assert [mapping(p) for p in (0.1, 0.2, 0.3, 0.4)] == [0.0, 0.5, 0.5, 1.0]
    assert mapping(0.05) == 0.0  # below the first knot: the first knot value
    assert mapping(0.25) == 0.5  # nearest lower knot is 0.2
    assert mapping(0.95) == 1.0


def test_isotonic_is_monotone_and_matches_calibration_frequencies() -> None:
    probs, labels = _overconfident(2000, seed=4)
    mapping = fit_isotonic(probs, labels)
    grid = [i / 1000 for i in range(1001)]
    values = [mapping(p) for p in grid]
    assert all(a <= b for a, b in itertools.pairwise(values))
    fitted = [mapping(p) for p in probs]
    assert sum(fitted) == pytest.approx(sum(labels))
    assert expected_calibration_error(fitted, labels) < 0.02


def test_ties_share_one_isotonic_value() -> None:
    mapping = fit_isotonic([0.5, 0.5, 0.5, 0.9], [True, False, False, True])
    assert mapping(0.5) == pytest.approx(1.0 / 3.0)
    assert mapping(0.9) == 1.0


def test_ece_is_small_on_a_calibrated_series() -> None:
    probs = [_uniform(5, "p", i) for i in range(20000)]
    labels = [_uniform(5, "y", i) < p for i, p in enumerate(probs)]
    assert expected_calibration_error(probs, labels) < 0.02


def test_brier_and_accuracy_on_hand_values() -> None:
    probs = [0.9, 0.2, 0.5, 0.4]
    labels = [True, False, False, True]
    # (0.01 + 0.04 + 0.25 + 0.36) / 4 = 0.165
    assert brier_score(probs, labels) == pytest.approx(0.165)
    # 0.5 reads as positive: right, right, wrong, wrong.
    assert accuracy_at_half(probs, labels) == pytest.approx(0.5)


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    probs, labels = _overconfident(400, seed=6)
    scaler = fit_temperature(probs, labels)
    low, high = bootstrap_ece_change(probs, labels, scaler, resamples=200)
    point = expected_calibration_error(
        [scaler(p) for p in probs], labels
    ) - expected_calibration_error(probs, labels)
    assert low <= point <= high
    assert high < 0.0
    assert (low, high) == bootstrap_ece_change(probs, labels, scaler, resamples=200)


@pytest.mark.parametrize(
    ("change", "met"),
    [
        ({"ece": -0.03, "brier": -0.001, "accuracy": -0.01}, True),
        ({"ece": -0.029, "brier": -0.01, "accuracy": 0.0}, False),
        ({"ece": -0.1, "brier": 0.0, "accuracy": 0.0}, False),
        ({"ece": -0.1, "brier": -0.01, "accuracy": -0.02}, False),
    ],
)
def test_rule_met_follows_the_preregistered_thresholds(
    change: dict[str, float], met: bool
) -> None:
    before = {"ece": 0.2, "brier": 0.2, "accuracy": 0.8}
    after = {key: before[key] + change[key] for key in before}
    assert rule_met(before, after) is met
