"""Offline semantic acceptance over saved CORD combined outcomes (#184)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from typevet_evals.cord.semantic_acceptance import (
    ABSTENTION_CHECK,
    ABSTENTION_FLOOR,
    ACCURACY_CHECK,
    ANSWERABLE_ACCURACY_FLOOR,
    CONTRADICTED_RECALL_CHECK,
    CONTRADICTED_RECALL_FLOOR,
    FALSE_CLEAR_CEILING,
    FALSE_CLEAR_CHECK,
    FROZEN_CLAIM_COUNT,
    LABEL_SHARE_CEILING,
    LABEL_SHARE_CHECK,
    Bound,
    CheckStatus,
    ThresholdCheck,
    accept_combined_receipt,
    accept_semantic_outcome,
)
from typevet_evals.datasets.cord_expense import load_expense_cases

pytestmark = pytest.mark.unit

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "cord"
    / "semantic_acceptance"
)
MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "manifest.json"
)
ZERO_ABSTENTION = "receipt_425e6c9.json"
EVERY_CHECK = (
    ACCURACY_CHECK,
    CONTRADICTED_RECALL_CHECK,
    FALSE_CLEAR_CHECK,
    ABSTENTION_CHECK,
    LABEL_SHARE_CHECK,
)


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def frozen_claim_ids() -> tuple[str, ...]:
    cases = load_expense_cases(MANIFEST.read_text(encoding="utf-8"))
    return tuple(case.claim_id for case in cases)


def test_thresholds_are_the_issue_161_revision_one_values() -> None:
    assert ANSWERABLE_ACCURACY_FLOOR == 0.67
    assert CONTRADICTED_RECALL_FLOOR == 0.5
    assert FALSE_CLEAR_CEILING == 0.25
    assert ABSTENTION_FLOOR == 0.5
    assert LABEL_SHARE_CEILING == 0.8
    assert FROZEN_CLAIM_COUNT == 18


def test_labeled_synthetic_receipt_meets_every_floor() -> None:
    outcome = accept_combined_receipt(load("labeled_synthetic_pass.json"))
    assert outcome.accepted
    assert outcome.claims == FROZEN_CLAIM_COUNT
    assert tuple(check.name for check in outcome.checks) == EVERY_CHECK
    assert all(check.status is CheckStatus.PASS for check in outcome.checks)
    assert outcome.failures == ()


def test_routed_verdicts_score_like_judge_labels() -> None:
    judge = accept_combined_receipt(load("labeled_synthetic_pass.json"))
    routed = accept_combined_receipt(load("routed_verdicts_pass.json"))
    assert routed.accepted
    assert [c.measured for c in routed.checks] == [c.measured for c in judge.checks]


@pytest.mark.parametrize(
    ("fixture", "failed"),
    [
        ("accuracy_below_floor.json", ACCURACY_CHECK),
        ("contradicted_recall_below_floor.json", CONTRADICTED_RECALL_CHECK),
        ("false_clear_above_ceiling.json", FALSE_CLEAR_CHECK),
        ("abstention_below_floor.json", ABSTENTION_CHECK),
        ("label_share_above_ceiling.json", LABEL_SHARE_CHECK),
    ],
)
def test_each_threshold_fail_mode_rejects_the_receipt(
    fixture: str, failed: str
) -> None:
    outcome = accept_combined_receipt(load(fixture))
    assert not outcome.accepted
    assert outcome.check(failed).status is CheckStatus.FAIL
    assert any(failed in reason for reason in outcome.failures)


def test_zero_abstention_receipt_is_rejected_not_soft_warned() -> None:
    outcome = accept_combined_receipt(
        load(ZERO_ABSTENTION), claim_ids=frozen_claim_ids()
    )
    assert not outcome.accepted
    assert outcome.claims == FROZEN_CLAIM_COUNT
    assert outcome.check(ABSTENTION_CHECK).measured == 0.0
    failing = {c.name for c in outcome.checks if c.status is CheckStatus.FAIL}
    assert failing == {ACCURACY_CHECK, FALSE_CLEAR_CHECK, ABSTENTION_CHECK}
    assert outcome.check(CONTRADICTED_RECALL_CHECK).status is CheckStatus.PASS
    assert outcome.check(LABEL_SHARE_CHECK).status is CheckStatus.PASS


def test_zero_abstention_receipt_reports_every_missed_bound() -> None:
    outcome = accept_combined_receipt(load(ZERO_ABSTENTION))
    assert len(outcome.failures) == 3
    assert any(
        ABSTENTION_CHECK in reason and "0.5" in reason for reason in outcome.failures
    )


def test_a_missing_gold_class_cannot_pass_overall_acceptance() -> None:
    outcome = accept_combined_receipt(load("no_insufficient_gold.json"))
    assert outcome.check(ABSTENTION_CHECK).status is CheckStatus.NOT_COMPUTABLE
    assert outcome.check(ABSTENTION_CHECK).measured is None
    others = [c for c in outcome.checks if c.name != ABSTENTION_CHECK]
    assert all(c.status is CheckStatus.PASS for c in others)
    assert not outcome.accepted


@pytest.mark.parametrize(
    ("fixture", "message"),
    [
        ("incomplete_coverage.json", "17"),
        ("extra_claim.json", "19"),
        ("duplicate_claim.json", "duplicate"),
    ],
)
def test_claim_coverage_against_the_frozen_eighteen_is_enforced(
    fixture: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        accept_combined_receipt(load(fixture))


def test_claim_ids_must_match_the_frozen_manifest_when_supplied() -> None:
    receipt = load("labeled_synthetic_pass.json")
    renamed = list(frozen_claim_ids())
    renamed[0] = "R99-C9"
    with pytest.raises(ValueError, match="frozen manifest"):
        accept_combined_receipt(receipt, claim_ids=renamed)


def test_every_case_needs_a_combined_row() -> None:
    receipt = load("labeled_synthetic_pass.json")
    del receipt["combined"]["R01-C1"]
    with pytest.raises(ValueError, match="R01-C1"):
        accept_combined_receipt(receipt)


def test_a_combined_row_outside_the_cases_is_rejected() -> None:
    receipt = load("labeled_synthetic_pass.json")
    receipt["combined"]["R09-C9"] = {"label": "match"}
    with pytest.raises(ValueError, match="R09-C9"):
        accept_combined_receipt(receipt)


def test_a_receipt_without_combined_outcomes_is_rejected() -> None:
    with pytest.raises(ValueError, match="combined"):
        accept_combined_receipt({"cases": []})


def test_a_case_without_gold_is_rejected() -> None:
    receipt = load("labeled_synthetic_pass.json")
    del receipt["cases"][0]["expected_verdict"]
    with pytest.raises(ValueError, match="expected_verdict"):
        accept_combined_receipt(receipt)


@pytest.mark.parametrize("label", ["", "yes", "MATCH", "insufficient_evidences"])
def test_a_predicted_label_outside_the_vocabularies_is_rejected(label: str) -> None:
    receipt = load("labeled_synthetic_pass.json")
    receipt["combined"]["R01-C1"]["label"] = label
    with pytest.raises(ValueError, match="label"):
        accept_combined_receipt(receipt)


def test_gold_verdicts_outside_the_manifest_vocabulary_are_rejected() -> None:
    receipt = load("labeled_synthetic_pass.json")
    receipt["cases"][0]["expected_verdict"] = "match"
    with pytest.raises(ValueError, match="verdict"):
        accept_combined_receipt(receipt)


def test_accept_semantic_outcome_needs_the_same_claim_ids_on_both_sides() -> None:
    gold = {"A": "supported", "B": "contradicted"}
    with pytest.raises(ValueError, match="claim"):
        accept_semantic_outcome(gold, {"A": "supported"})


def test_accept_semantic_outcome_rejects_an_empty_outcome() -> None:
    with pytest.raises(ValueError, match="no claim"):
        accept_semantic_outcome({}, {})


@pytest.mark.parametrize("measured", [float("inf"), float("nan")])
def test_a_non_finite_measure_never_passes_a_floor(measured: float) -> None:
    check = ThresholdCheck(
        name=ACCURACY_CHECK,
        bound=Bound.FLOOR,
        limit=ANSWERABLE_ACCURACY_FLOOR,
        measured=measured,
        denominator=12,
    )
    assert check.status is CheckStatus.FAIL
    assert not check.passed


@pytest.mark.parametrize("measured", [float("-inf"), float("nan")])
def test_a_non_finite_measure_never_passes_a_ceiling(measured: float) -> None:
    check = ThresholdCheck(
        name=FALSE_CLEAR_CHECK,
        bound=Bound.CEILING,
        limit=FALSE_CLEAR_CEILING,
        measured=measured,
        denominator=12,
    )
    assert check.status is CheckStatus.FAIL


def test_check_names_an_unknown_bound() -> None:
    outcome = accept_combined_receipt(load("labeled_synthetic_pass.json"))
    with pytest.raises(KeyError, match="unknown_check"):
        outcome.check("unknown_check")
