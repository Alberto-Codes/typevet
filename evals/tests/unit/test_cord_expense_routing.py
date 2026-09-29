"""Three-label question and verdict routing for the CORD expense smoke (#164)."""

from __future__ import annotations

import pytest

from typevet.domain.judgment_questions import Choice
from typevet_evals.datasets.cord_expense import (
    INSUFFICIENT_EVIDENCE,
    LABEL_ORDER,
    MATCH,
    MISMATCH,
    VERDICTS,
    expense_question,
    route,
)

pytestmark = pytest.mark.unit


def test_label_order_is_pinned() -> None:
    assert LABEL_ORDER == ("insufficient_evidence", "mismatch", "match")
    assert (INSUFFICIENT_EVIDENCE, MISMATCH, MATCH) == LABEL_ORDER


def test_expense_question_is_a_choice_with_labels_in_pinned_order() -> None:
    question = expense_question()
    assert isinstance(question, Choice)
    assert tuple(question.criteria) == LABEL_ORDER
    assert all(isinstance(v, str) and v for v in question.criteria.values())
    assert isinstance(question.instructions, str)
    assert question.instructions


def test_expense_question_is_built_fresh_each_call() -> None:
    first = expense_question()
    first.criteria.pop(MATCH)
    assert tuple(expense_question().criteria) == LABEL_ORDER


@pytest.mark.parametrize(
    ("label", "verdict"),
    [
        (INSUFFICIENT_EVIDENCE, "insufficient"),
        (MISMATCH, "contradicted"),
        (MATCH, "supported"),
    ],
)
def test_route_maps_each_label_to_its_verdict(label: str, verdict: str) -> None:
    assert route(label) == verdict


def test_route_is_exhaustive_and_one_to_one() -> None:
    routed = [route(label) for label in LABEL_ORDER]
    assert sorted(routed) == sorted(VERDICTS)
    assert len(set(routed)) == len(LABEL_ORDER)


@pytest.mark.parametrize("label", ["", "MATCH", "supported", "unknown", " match"])
def test_route_rejects_labels_outside_the_question(label: str) -> None:
    with pytest.raises(ValueError, match="label"):
        route(label)
