"""Unit tests: multi-class calibration from saved distributions ([#296][i296]).

Expected values are hand-computed from equal-width bins (10 by default).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_outcome_replay_calibration.py
    ```

See Also:
    - [typevet_evals.outcome_replay_metrics][]: metric helpers

[i296]: https://github.com/Alberto-Codes/typevet/issues/296
"""

from __future__ import annotations

import pytest

from typevet_evals.outcome_replay_metrics import (
    INSUFFICIENT_N,
    MIN_CLASS_COUNT,
    classwise_ece,
    top_label_ece,
)

pytestmark = pytest.mark.unit

_LABELS16 = tuple(f"l{i:02d}" for i in range(16))


def _peaked(top: str, top_mass: float) -> dict[str, float]:
    rest = (1.0 - top_mass) / 15
    return {label: (top_mass if label == top else rest) for label in _LABELS16}


def test_top_label_ece_on_sixteen_labels_matches_hand_value() -> None:
    """Two bins: conf 0.85 at 6/10 correct, conf 0.25 at 3/10 correct.

    ECE = (10 * |0.85 - 0.6| + 10 * |0.25 - 0.3|) / 20 = 3.0 / 20 = 0.15.
    """
    distributions: list[dict[str, float]] = []
    gold: list[str] = []
    for i in range(10):
        distributions.append(_peaked("l00", 0.85))
        gold.append("l00" if i < 6 else "l05")
    for i in range(10):
        distributions.append(_peaked("l15", 0.25))
        gold.append("l15" if i < 3 else "l07")
    assert top_label_ece(distributions, gold, labels=_LABELS16) == pytest.approx(0.15)


def _split_bin_cases() -> tuple[list[dict[str, float]], list[str]]:
    distributions = [{"a": 0.65, "b": 0.35}] * 30 + [{"a": 0.75, "b": 0.25}] * 30
    return distributions, ["a"] * 30 + ["b"] * 30


def test_default_is_ten_bins_for_top_label_ece() -> None:
    """Conf 0.65 (30 right) and 0.75 (30 wrong) share one bin only at 5 bins.

    10 bins: (30 * 0.35 + 30 * 0.75) / 60 = 0.55.
    5 bins: one bin, mean 0.70, accuracy 0.5, so 0.2.
    """
    distributions, gold = _split_bin_cases()
    labels = ("a", "b")
    assert top_label_ece(distributions, gold, labels=labels) == pytest.approx(0.55)
    five = top_label_ece(distributions, gold, labels=labels, n_bins=5)
    assert five == pytest.approx(0.2)


def test_default_is_ten_bins_for_classwise_ece() -> None:
    """Class a: p 0.65 (30 hits), 0.75 (30 misses); class b: 0.35, 0.25.

    10 bins: each class (30 * 0.35 + 30 * 0.75) / 60 = 0.55.
    5 bins: each class one bin with gap 0.2.
    """
    distributions, gold = _split_bin_cases()
    labels = ("a", "b")
    report = classwise_ece(distributions, gold, labels=labels)
    assert report["n_bins"] == 10
    assert report["classwise_ece"] == pytest.approx(0.55)
    five = classwise_ece(distributions, gold, labels=labels, n_bins=5)
    assert five["classwise_ece"] == pytest.approx(0.2)


def test_top_label_tie_goes_to_first_label_in_labels_order() -> None:
    """Tie b/c at 0.4: labels order picks b (gold), so gap |0.4 - 1| = 0.6.

    Dict order would pick c (wrong), so gap |0.4 - 0| = 0.4.
    """
    distributions = [{"c": 0.4, "b": 0.4, "a": 0.2}]
    ece = top_label_ece(distributions, ["b"], labels=("a", "b", "c"))
    assert ece == pytest.approx(0.6)


def test_top_label_ece_is_zero_when_empty() -> None:
    assert top_label_ece([], [], labels=_LABELS16) == 0.0


def _three_class_cases() -> tuple[list[dict[str, float]], list[str]]:
    rows = {
        "a": ({"a": 0.75, "b": 0.15, "c": 0.10}, 30),
        "b": ({"a": 0.15, "b": 0.75, "c": 0.10}, 30),
        "c": ({"a": 0.45, "b": 0.45, "c": 0.10}, 5),
    }
    distributions: list[dict[str, float]] = []
    gold: list[str] = []
    for label, (probs, count) in rows.items():
        distributions.extend([probs] * count)
        gold.extend([label] * count)
    return distributions, gold


def test_classwise_ece_excludes_class_below_thirty() -> None:
    """Class a: p_a 0.75 x30 (all a), 0.15 x30 (none), 0.45 x5 (none).

    ECE_a = (30*0.25 + 30*0.15 + 5*0.45) / 65 = 14.25 / 65; class b is the
    mirror image. Class c has 5 instances, so the mean is over a and b only.
    With c included the mean would change (ECE_c = 0.1 - 5/65).
    """
    distributions, gold = _three_class_cases()
    report = classwise_ece(distributions, gold, labels=("a", "b", "c"))
    expected = 14.25 / 65
    assert MIN_CLASS_COUNT == 30
    assert report["classwise_ece"] == pytest.approx(expected)
    assert report["insufficient_n"] == ["c"]
    assert report["classes"]["c"] == {
        "count": 5,
        "ece": None,
        "status": INSUFFICIENT_N,
    }
    assert INSUFFICIENT_N == "insufficient N"
    assert report["classes"]["a"]["count"] == 30
    assert report["classes"]["a"]["ece"] == pytest.approx(expected)
    assert report["classes"]["b"]["ece"] == pytest.approx(expected)


def test_classwise_ece_is_none_when_every_class_is_small() -> None:
    distributions, gold = _three_class_cases()
    report = classwise_ece(distributions[-5:], gold[-5:], labels=("a", "b", "c"))
    assert report["classwise_ece"] is None
    assert report["insufficient_n"] == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("distributions", "gold", "labels", "match"),
    [
        ([{"a": 1.0, "b": 0.0}], [], ("a", "b"), "same length"),
        ([{"a": 1.0, "b": 0.0}], ["a"], (), "must not be empty"),
        ([{"a": 1.0, "b": 0.0}], ["a"], ("a", "a"), "duplicate labels"),
        ([{"a": 1.0, "b": 0.0}], ["z"], ("a", "b"), "not in labels"),
        ([{"a": 0.9, "b": 0.9}], ["a"], ("a", "b"), "not a valid distribution"),
    ],
)
def test_calibration_inputs_are_checked(
    distributions: list[dict[str, float]],
    gold: list[str],
    labels: tuple[str, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        top_label_ece(distributions, gold, labels=labels)
    with pytest.raises(ValueError, match=match):
        classwise_ece(distributions, gold, labels=labels)
