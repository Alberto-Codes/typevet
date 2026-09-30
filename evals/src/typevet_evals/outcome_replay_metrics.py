"""Offline metrics from saved complete label distributions ([#132][i132]).

Multi-class calibration ([#296][i296]) uses the equal-width bins of
``typevet_evals.face_match.metrics.expected_calibration_error`` (10 by
default). ``top_label_ece`` bins the top-label confidence against
correctness. ``classwise_ece`` bins the mass of each class against a
one-vs-rest gold and takes the mean over the classes with at least
``MIN_CLASS_COUNT`` gold instances.

Attributes:
    MIN_CLASS_COUNT (int): Fewest gold instances for a class-wise ECE.
    INSUFFICIENT_N (str): Status of a class below ``MIN_CLASS_COUNT``.

Examples:
    ```python
    from typevet_evals.outcome_replay_metrics import (
        SavedPromptOutcome,
        compare_matched_prompt_outcomes,
    )

    seed = {"c1": SavedPromptOutcome(probabilities={"yes": 0.7, "no": 0.3})}
    cand = {"c1": SavedPromptOutcome(probabilities={"yes": 0.9, "no": 0.1})}
    report = compare_matched_prompt_outcomes(
        {"c1": "yes"}, seed, cand, labels=("no", "yes")
    )
    assert report["shared_valid_quality"]["seed"]["accuracy"] == 1.0
    ```

See Also:
    - [typevet_evals.tpjep.outcome][]: ``prob_valid`` tolerance rules
    - [typevet_evals.face_match.metrics][]: shared equal-width binning

[i132]: https://github.com/Alberto-Codes/typevet/issues/132
[i296]: https://github.com/Alberto-Codes/typevet/issues/296
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet_evals.face_match.metrics import DEFAULT_BINS, expected_calibration_error
from typevet_evals.tpjep.outcome import prob_valid

_PROB_SUM_TOLERANCE: Final[float] = 0.001
_LOG_FLOOR: Final[float] = 1e-15
_REPLAY_SCHEMA: Final[str] = "descriptive_v2"
MIN_CLASS_COUNT: Final[int] = 30
INSUFFICIENT_N: Final[str] = "insufficient N"


@dataclass(frozen=True, slots=True)
class SavedPromptOutcome:
    """One saved case outcome for offline replay (no model calls).

    Attributes:
        probabilities (Mapping[str, float]): Label-keyed distribution.
        transport_ok (bool): Whether the upstream call completed.

    Examples:
        ```python
        SavedPromptOutcome(probabilities={"a": 0.5, "b": 0.5})
        ```
    """

    probabilities: Mapping[str, float]
    transport_ok: bool = True


def _predicted_label(probabilities: Mapping[str, float]) -> str:
    return max(probabilities, key=probabilities.__getitem__)


def brier_score(
    probabilities: Mapping[str, float],
    gold: str,
    *,
    labels: tuple[str, ...],
) -> float:
    """Return multi-class Brier score for one gold label.

    Args:
        probabilities: Normalized label masses.
        gold: Gold class label.
        labels: Scheduled label set for the one-hot target.

    Returns:
        Sum of squared errors against a one-hot gold vector.
    """
    return sum(
        (float(probabilities.get(label, 0.0)) - (1.0 if label == gold else 0.0)) ** 2
        for label in labels
    )


def log_loss(probabilities: Mapping[str, float], gold: str) -> float:
    """Return natural log loss for one gold label.

    Args:
        probabilities: Normalized label masses.
        gold: Gold class label.

    Returns:
        ``-log(p_gold)`` with a floor on zero mass.
    """
    mass = float(probabilities.get(gold, 0.0))
    return -math.log(max(mass, _LOG_FLOOR))


def _validate_gold(gold: Mapping[str, str], *, labels: tuple[str, ...]) -> None:
    label_set = set(labels)
    for case_id, gold_label in gold.items():
        if gold_label not in label_set:
            msg = f"gold label {gold_label!r} for {case_id!r} not in scheduled labels"
            raise ValueError(msg)


def _validate_no_unexpected_outcomes(
    gold: Mapping[str, str],
    outcomes: Mapping[str, SavedPromptOutcome],
    *,
    arm: str,
) -> None:
    unexpected = set(outcomes) - set(gold)
    if unexpected:
        msg = f"{arm} has unexpected outcome ids: {sorted(unexpected)}"
        raise ValueError(msg)


def _classify_arm(
    gold: Mapping[str, str],
    outcomes: Mapping[str, SavedPromptOutcome],
    *,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    scheduled_ids = list(gold)
    missing: list[str] = []
    failed_transport: list[str] = []
    invalid_distribution: list[str] = []
    valid: list[str] = []
    for case_id in scheduled_ids:
        if case_id not in outcomes:
            missing.append(case_id)
            continue
        outcome = outcomes[case_id]
        if not outcome.transport_ok:
            failed_transport.append(case_id)
            continue
        if not prob_valid(outcome.probabilities, labels=labels):
            invalid_distribution.append(case_id)
            continue
        valid.append(case_id)
    return {
        "scheduled_case_count": len(scheduled_ids),
        "scheduled_case_ids": scheduled_ids,
        "missing_case_ids": missing,
        "failed_transport_case_ids": failed_transport,
        "invalid_distribution_case_ids": invalid_distribution,
        "valid_distribution_case_ids": valid,
    }


def _quality_on_cases(
    gold: Mapping[str, str],
    outcomes: Mapping[str, SavedPromptOutcome],
    case_ids: list[str],
    *,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    correct: list[str] = []
    briers: list[float] = []
    losses: list[float] = []
    for case_id in case_ids:
        probs = outcomes[case_id].probabilities
        gold_label = gold[case_id]
        if _predicted_label(probs) == gold_label:
            correct.append(case_id)
        briers.append(brier_score(probs, gold_label, labels=labels))
        losses.append(log_loss(probs, gold_label))
    n_valid = len(case_ids)
    return {
        "case_count": n_valid,
        "accuracy": (len(correct) / n_valid) if n_valid else None,
        "mean_brier": (sum(briers) / n_valid) if n_valid else None,
        "mean_log_loss": (sum(losses) / n_valid) if n_valid else None,
    }


def compare_matched_prompt_outcomes(
    gold: Mapping[str, str],
    seed: Mapping[str, SavedPromptOutcome],
    candidate: Mapping[str, SavedPromptOutcome],
    *,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Score seed and candidate saved outcomes on shared scheduled case ids.

    Args:
        gold: Gold label per scheduled case id.
        seed: Baseline prompt outcomes keyed by case id.
        candidate: Candidate prompt outcomes keyed by case id.
        labels: Scheduled label set; distributions must match exactly.

    Returns:
        Descriptive replay report with per-arm scheduling accounting and paired
        deltas on the intersection of valid distributions only. No promotion
        or selection verdicts.

    Raises:
        ValueError: When ``labels`` is empty or contains duplicates, gold labels
            are invalid, or an arm reports ids outside the scheduled gold set.
    """
    if not labels:
        msg = "labels must not be empty"
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = "duplicate labels are not allowed"
        raise ValueError(msg)
    _validate_gold(gold, labels=labels)
    _validate_no_unexpected_outcomes(gold, seed, arm="seed")
    _validate_no_unexpected_outcomes(gold, candidate, arm="candidate")
    seed_arm = _classify_arm(gold, seed, labels=labels)
    candidate_arm = _classify_arm(gold, candidate, labels=labels)
    shared_valid = sorted(
        set(seed_arm["valid_distribution_case_ids"])
        & set(candidate_arm["valid_distribution_case_ids"])
    )
    seed_shared = _quality_on_cases(gold, seed, shared_valid, labels=labels)
    candidate_shared = _quality_on_cases(gold, candidate, shared_valid, labels=labels)
    delta: dict[str, float | None] = {}
    for key in ("mean_brier", "mean_log_loss", "accuracy"):
        s_val = seed_shared.get(key)
        c_val = candidate_shared.get(key)
        if isinstance(s_val, (int, float)) and isinstance(c_val, (int, float)):
            delta[key] = float(c_val) - float(s_val)
        else:
            delta[key] = None
    return {
        "replay_report_schema": _REPLAY_SCHEMA,
        "labels": labels,
        "prob_sum_tolerance": _PROB_SUM_TOLERANCE,
        "scheduled_case_count": len(gold),
        "scheduled_case_ids": list(gold),
        "seed": seed_arm,
        "candidate": candidate_arm,
        "shared_valid_case_ids": shared_valid,
        "shared_valid_quality": {
            "seed": seed_shared,
            "candidate": candidate_shared,
        },
        "delta_candidate_minus_seed": delta,
    }


def replay_identical_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    """Return true when two compare reports are identical (idempotent replay).

    Args:
        first: Report from ``compare_matched_prompt_outcomes``.
        second: Second report on the same inputs.

    Returns:
        ``True`` when mappings are equal.
    """
    return dict(first) == dict(second)


def _check_calibration_inputs(
    distributions: Sequence[Mapping[str, float]],
    gold: Sequence[str],
    labels: tuple[str, ...],
) -> None:
    if len(distributions) != len(gold):
        msg = "distributions and gold must have the same length"
        raise ValueError(msg)
    if not labels:
        msg = "labels must not be empty"
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = "duplicate labels are not allowed"
        raise ValueError(msg)
    for index, (probabilities, gold_label) in enumerate(
        zip(distributions, gold, strict=True)
    ):
        if gold_label not in labels:
            msg = f"gold label {gold_label!r} at row {index} is not in labels"
            raise ValueError(msg)
        if not prob_valid(probabilities, labels=labels):
            msg = f"row {index} is not a valid distribution over labels"
            raise ValueError(msg)


def top_label_ece(
    distributions: Sequence[Mapping[str, float]],
    gold: Sequence[str],
    *,
    labels: tuple[str, ...],
    n_bins: int = DEFAULT_BINS,
) -> float:
    """Return the top-label ECE over any label tuple.

    The confidence of a row is its highest label mass. A row is correct when
    the label with that mass is the gold label. When labels tie on that
    mass, the first of them in ``labels`` order is the predicted label.

    Args:
        distributions: One complete label distribution per row.
        gold: Gold label per row, in the same order.
        labels: Label set; each distribution must cover it exactly.
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]; 0.0 when there are no rows.

    Raises:
        ValueError: When lengths differ, ``labels`` is empty or has
            duplicates, a gold label is not in ``labels`` or a distribution
            is not valid.
    """
    _check_calibration_inputs(distributions, gold, labels)
    confidences = [max(float(v) for v in p.values()) for p in distributions]
    correct = [
        max(labels, key=lambda label: float(p[label])) == gold_label
        for p, gold_label in zip(distributions, gold, strict=True)
    ]
    return expected_calibration_error(confidences, correct, n_bins)


def classwise_ece(
    distributions: Sequence[Mapping[str, float]],
    gold: Sequence[str],
    *,
    labels: tuple[str, ...],
    n_bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Return the class-wise ECE and the value for each class.

    Each class gets a one-vs-rest ECE over all rows: its mass against
    ``gold == class``. A class with fewer than ``MIN_CLASS_COUNT`` gold
    instances gets the status ``INSUFFICIENT_N`` and no value. The
    class-wise ECE is the mean over the other classes.

    Args:
        distributions: One complete label distribution per row.
        gold: Gold label per row, in the same order.
        labels: Label set; each distribution must cover it exactly.
        n_bins: Number of equal-width bins.

    Returns:
        ``classwise_ece`` (``None`` when no class has enough instances),
        ``classes`` with ``count``, ``ece`` and ``status`` for each label,
        ``insufficient_n`` in label order, ``min_class_count`` and
        ``n_bins``.

    Raises:
        ValueError: When lengths differ, ``labels`` is empty or has
            duplicates, a gold label is not in ``labels`` or a distribution
            is not valid.
    """
    _check_calibration_inputs(distributions, gold, labels)
    classes: dict[str, dict[str, Any]] = {}
    insufficient: list[str] = []
    values: list[float] = []
    for label in labels:
        count = sum(1 for gold_label in gold if gold_label == label)
        if count < MIN_CLASS_COUNT:
            insufficient.append(label)
            classes[label] = {"count": count, "ece": None, "status": INSUFFICIENT_N}
            continue
        masses = [float(p[label]) for p in distributions]
        hits = [gold_label == label for gold_label in gold]
        value = expected_calibration_error(masses, hits, n_bins)
        values.append(value)
        classes[label] = {"count": count, "ece": value, "status": "ok"}
    return {
        "classwise_ece": (sum(values) / len(values)) if values else None,
        "classes": classes,
        "insufficient_n": insufficient,
        "min_class_count": MIN_CLASS_COUNT,
        "n_bins": n_bins,
    }
