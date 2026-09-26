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
def test_compare_rejects_duplicate_labels() -> None:
    """Repeated label definitions cannot change replay metric weighting."""
    gold = {"a": "yes"}
    outcomes = {"a": SavedPromptOutcome(probabilities={"yes": 0.9, "no": 0.1})}
    with pytest.raises(ValueError, match="duplicate labels"):
        compare_matched_prompt_outcomes(
            gold, outcomes, outcomes, labels=("yes", "yes", "no")
        )


@pytest.mark.unit
def test_brier_and_log_loss_hand_calculated() -> None:
    """Formulas match pencil-and-paper values for a two-label case."""
    probs = {"no": 0.25, "yes": 0.75}
    # (0.25-1)^2 + (0.75-0)^2 = 0.5625 + 0.5625 = 1.125
    assert brier_score(probs, "no", labels=_LABELS) == pytest.approx(1.125)
    assert log_loss(probs, "no") == pytest.approx(-math.log(0.25))


@pytest.mark.unit
def test_invalid_distribution_excluded_from_shared_valid() -> None:
    """Malformed distributions do not enter shared valid pairing."""
    gold = {"a": "yes", "b": "yes"}
    seed = {
        "a": SavedPromptOutcome(probabilities={"no": 0.1, "yes": 0.9}),
        "b": SavedPromptOutcome(probabilities={"no": 0.6, "yes": 0.3}),
    }
    report = compare_matched_prompt_outcomes(gold, seed, seed, labels=_LABELS)
    assert report["seed"]["invalid_distribution_case_ids"] == ["b"]
    assert report["shared_valid_case_ids"] == ["a"]
    assert report["shared_valid_quality"]["seed"]["accuracy"] == 1.0


@pytest.mark.unit
def test_shared_valid_deltas_require_both_arms() -> None:
    """Paired deltas use only cases valid on seed and candidate."""
    gold = {"c1": "yes"}
    seed = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.4, "yes": 0.6}),
    }
    better = {
        "c1": SavedPromptOutcome(probabilities={"no": 0.2, "yes": 0.8}),
    }
    base = compare_matched_prompt_outcomes(gold, seed, better, labels=_LABELS)
    assert "candidate_improved" not in base
    assert base["delta_candidate_minus_seed"]["mean_brier"] < 0.0
    tie = compare_matched_prompt_outcomes(gold, seed, seed, labels=_LABELS)
    assert tie["delta_candidate_minus_seed"]["mean_brier"] == 0.0
    assert tie["delta_candidate_minus_seed"]["mean_log_loss"] == 0.0


@pytest.mark.unit
def test_dropped_hard_case_does_not_improve_via_missing_candidate() -> None:
    """Omitting a hard case cannot inflate shared metrics (optimizer blocker)."""
    gold = {"easy": "yes", "hard": "yes"}
    seed = {
        "easy": SavedPromptOutcome(probabilities={"no": 0.1, "yes": 0.9}),
        "hard": SavedPromptOutcome(probabilities={"no": 0.1, "yes": 0.9}),
    }
    candidate = {
        "easy": SavedPromptOutcome(probabilities={"no": 0.1, "yes": 0.9}),
    }
    report = compare_matched_prompt_outcomes(gold, seed, candidate, labels=_LABELS)
    assert report["candidate"]["missing_case_ids"] == ["hard"]
    assert report["shared_valid_case_ids"] == ["easy"]
    assert report["delta_candidate_minus_seed"]["mean_brier"] == 0.0
    assert report["delta_candidate_minus_seed"]["mean_log_loss"] == 0.0
    assert report["shared_valid_quality"]["seed"]["accuracy"] == 1.0
    assert report["shared_valid_quality"]["candidate"]["accuracy"] == 1.0


@pytest.mark.unit
def test_unexpected_outcome_id_rejected() -> None:
    """Arms cannot report ids outside the scheduled gold set."""
    gold = {"a": "yes"}
    seed = {"a": SavedPromptOutcome(probabilities={"no": 0.5, "yes": 0.5})}
    extra = {
        "a": SavedPromptOutcome(probabilities={"no": 0.5, "yes": 0.5}),
        "ghost": SavedPromptOutcome(probabilities={"no": 0.5, "yes": 0.5}),
    }
    with pytest.raises(ValueError, match="unexpected outcome ids"):
        compare_matched_prompt_outcomes(gold, seed, extra, labels=_LABELS)


@pytest.mark.unit
def test_replay_identical_reports_is_idempotent() -> None:
    """Recomputing the report on the same inputs yields an identical mapping."""
    gold = {"x": "no"}
    outcomes = {"x": SavedPromptOutcome(probabilities={"no": 0.55, "yes": 0.45})}
    first = compare_matched_prompt_outcomes(gold, outcomes, outcomes, labels=_LABELS)
    second = compare_matched_prompt_outcomes(gold, outcomes, outcomes, labels=_LABELS)
    assert replay_identical_reports(first, second)
