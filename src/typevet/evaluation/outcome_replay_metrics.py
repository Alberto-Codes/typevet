"""Offline metrics from saved complete label distributions ([#132][i132]).

Examples:
    ```python
    from typevet.evaluation.outcome_replay_metrics import (
        SavedPromptOutcome,
        compare_matched_prompt_outcomes,
    )

    seed = {"c1": SavedPromptOutcome(probabilities={"yes": 0.7, "no": 0.3})}
    cand = {"c1": SavedPromptOutcome(probabilities={"yes": 0.9, "no": 0.1})}
    report = compare_matched_prompt_outcomes(
        {"c1": "yes"}, seed, cand, labels=("no", "yes")
    )
    assert report["seed"]["accuracy"] == 1.0
    ```

See Also:
    - [typevet.evaluation.tpjep.outcome][]: ``prob_valid`` tolerance rules

[i132]: https://github.com/Alberto-Codes/typevet/issues/132
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from typevet.evaluation.tpjep.outcome import prob_valid

_PROB_SUM_TOLERANCE: Final[float] = 0.001
_LOG_FLOOR: Final[float] = 1e-15


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


def _quality_block(
    gold: Mapping[str, str],
    outcomes: Mapping[str, SavedPromptOutcome],
    *,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    matched = [case_id for case_id in gold if case_id in outcomes]
    transport_ok = sum(1 for cid in matched if outcomes[cid].transport_ok)
    valid: list[str] = []
    correct: list[str] = []
    briers: list[float] = []
    losses: list[float] = []
    invalid: list[str] = []
    for case_id in matched:
        outcome = outcomes[case_id]
        if not outcome.transport_ok:
            continue
        probs = outcome.probabilities
        if not prob_valid(probs, labels=labels):
            invalid.append(case_id)
            continue
        valid.append(case_id)
        gold_label = gold[case_id]
        if _predicted_label(probs) == gold_label:
            correct.append(case_id)
        briers.append(brier_score(probs, gold_label, labels=labels))
        losses.append(log_loss(probs, gold_label))
    n_valid = len(valid)
    return {
        "matched_cases": len(matched),
        "transport_ok": transport_ok,
        "valid_distribution_cases": n_valid,
        "invalid_distribution_cases": len(invalid),
        "accuracy": (len(correct) / n_valid) if n_valid else None,
        "mean_brier": (sum(briers) / n_valid) if n_valid else None,
        "mean_log_loss": (sum(losses) / n_valid) if n_valid else None,
        "coverage_valid_over_matched": (n_valid / len(matched)) if matched else None,
    }


def compare_matched_prompt_outcomes(
    gold: Mapping[str, str],
    seed: Mapping[str, SavedPromptOutcome],
    candidate: Mapping[str, SavedPromptOutcome],
    *,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Score seed and candidate saved outcomes on the same matched case ids.

    Args:
        gold: Gold label per case id.
        seed: Baseline prompt outcomes keyed by case id.
        candidate: Candidate prompt outcomes keyed by case id.
        labels: Scheduled label set; distributions must match exactly.

    Returns:
        Report with per-arm quality blocks and paired deltas on valid cases.
        ``candidate_improved`` is true only when candidate strictly lowers
        both mean Brier and mean log loss on valid distributions.

    Raises:
        ValueError: When ``labels`` is empty.
    """
    if not labels:
        msg = "labels must not be empty"
        raise ValueError(msg)
    seed_block = _quality_block(gold, seed, labels=labels)
    candidate_block = _quality_block(gold, candidate, labels=labels)
    delta: dict[str, float | None] = {}
    for key in ("mean_brier", "mean_log_loss", "accuracy"):
        s_val = seed_block.get(key)
        c_val = candidate_block.get(key)
        if isinstance(s_val, (int, float)) and isinstance(c_val, (int, float)):
            delta[key] = float(c_val) - float(s_val)
        else:
            delta[key] = None
    improved = False
    if (
        isinstance(seed_block["mean_brier"], (int, float))
        and isinstance(candidate_block["mean_brier"], (int, float))
        and isinstance(seed_block["mean_log_loss"], (int, float))
        and isinstance(candidate_block["mean_log_loss"], (int, float))
    ):
        improved = (
            candidate_block["mean_brier"] < seed_block["mean_brier"]
            and candidate_block["mean_log_loss"] < seed_block["mean_log_loss"]
        )
    return {
        "labels": labels,
        "prob_sum_tolerance": _PROB_SUM_TOLERANCE,
        "seed": seed_block,
        "candidate": candidate_block,
        "delta_candidate_minus_seed": delta,
        "candidate_improved": improved,
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
