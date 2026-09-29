"""Offline metrics from saved complete label distributions ([#132][i132]).

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

[i132]: https://github.com/Alberto-Codes/typevet/issues/132
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from typevet_evals.tpjep.outcome import prob_valid

_PROB_SUM_TOLERANCE: Final[float] = 0.001
_LOG_FLOOR: Final[float] = 1e-15
_REPLAY_SCHEMA: Final[str] = "descriptive_v2"


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
