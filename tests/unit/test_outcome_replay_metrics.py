"""Unit tests: offline replay metrics ([#132][i132]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_outcome_replay_metrics.py
    ```

See Also:
    - [typevet.evaluation.outcome_replay_metrics][]: metric helpers
"""

from __future__ import annotations

import math

import pytest

from typevet.evaluation.outcome_replay_metrics import (
    SavedPromptOutcome,
    brier_score,
    compare_matched_prompt_outcomes,
    log_loss,
    replay_identical_reports,
)

_LABELS = ("no", "yes")


@pytest.mark.unit
def test_brier_and_log_loss_hand_calculated() -> None:
    """Formulas match pencil-and-paper values for a two-label case."""
    probs = {"no": 0.25, "yes": 0.75}
    # (0.25-1)^2 + (0.75-0)^2 = 0.5625 + 0.5625 = 1.125
    assert brier_score(probs, "no", labels=_LABELS) == pytest.approx(1.125)
    assert log_loss(probs, "no") == pytest.approx(-math.log(0.25))


@pytest.mark.unit
def test_invalid_distribution_excluded_from_quality_denominator() -> None:
    """Malformed distributions do not enter accuracy or Brier means."""
    gold = {"a": "yes", "b": "yes"}
    seed = {
        "a": SavedPromptOutcome(probabilities={"no": 0.1, "yes": 0.9}),
        "b": SavedPromptOutcome(probabilities={"no": 0.6, "yes": 0.3}),
    }
    report = compare_matched_prompt_outcomes(gold, seed, seed, labels=_LABELS)
    assert report["seed"]["valid_distribution_cases"] == 1
    assert report["seed"]["invalid_distribution_cases"] == 1
    assert report["seed"]["accuracy"] == 1.0


@pytest.mark.unit
def test_compare_improvement_requires_both_metrics() -> None:
    """Candidate improvement needs lower Brier and log loss."""
    gold = {"c1": "yes"}
    seed = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.4, "yes": 0.6}),
    }
    better = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.2, "yes": 0.8}),
    }
    log_loss_only = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.41, "yes": 0.59}),
    }
    base = compare_matched_prompt_outcomes(gold, seed, better, labels=_LABELS)
    assert base["candidate_improved"] is True
    tie = compare_matched_prompt_outcomes(gold, seed, seed, labels=_LABELS)
    assert tie["candidate_improved"] is False
    mixed = compare_matched_prompt_outcomes(gold, seed, log_loss_only, labels=_LABELS)
    assert mixed["candidate_improved"] is False


@pytest.mark.unit
def test_replay_identical_reports_is_idempotent() -> None:
    """Recomputing the report on the same inputs yields an identical mapping."""
    gold = {"x": "no"}
    outcomes = {"x": SavedPromptOutcome(probabilities={"no": 0.55, "yes": 0.45})}
    first = compare_matched_prompt_outcomes(gold, outcomes, outcomes, labels=_LABELS)
    second = compare_matched_prompt_outcomes(gold, outcomes, outcomes, labels=_LABELS)
    assert replay_identical_reports(first, second)
