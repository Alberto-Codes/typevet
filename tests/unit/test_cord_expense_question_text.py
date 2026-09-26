"""Question wording that keeps ``insufficient_evidence`` reachable (#182).

The #181 decision traced abstention 0.00 on the six gold-insufficient claims to
the wording itself: the masked claims hide digits in the *claim* while the
receipt total stays readable, so "read the receipt total" plus "ignore
thousands separators" let a literal reader compare a masked amount anyway.
These tests lock the repaired intent so the defect cannot return quietly.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_cord_expense_question_text.py
    ```

See Also:
    - [typevet.evaluation.datasets.cord_expense][]: the question under test
"""

from __future__ import annotations

import pytest

from typevet.evaluation.datasets.cord_expense import (
    INSUFFICIENT_EVIDENCE,
    MATCH,
    MISMATCH,
    expense_question,
)

pytestmark = pytest.mark.unit

MASK = "?"
COMPARE_LABELS = (MISMATCH, MATCH)


def _instructions() -> str:
    text = expense_question().instructions
    assert isinstance(text, str)
    return text


def _criterion(label: str) -> str:
    text = expense_question().criteria[label]
    assert isinstance(text, str)
    return text


def test_instructions_name_the_mask_character() -> None:
    """The judge is told what a masked digit looks like, not left to guess."""
    assert MASK in _instructions()


def test_instructions_drop_the_separator_normalization_invitation() -> None:
    """The old separator clause invited normalizing ``?`` away too."""
    assert "ignore thousands separators" not in _instructions().lower()


def test_instructions_forbid_guessing_a_hidden_digit() -> None:
    """Separator tolerance must not extend to a hidden digit."""
    text = _instructions().lower()
    assert "not a digit" in text
    assert "guess" in text


def test_abstain_criterion_names_the_claim_amount_as_unreadable() -> None:
    """Claim-unreadable is the branch the six masked cases need."""
    abstain = _criterion(INSUFFICIENT_EVIDENCE).lower()
    assert "claim" in abstain
    assert MASK in abstain


def test_abstain_criterion_separates_claim_from_receipt() -> None:
    """The old single clause merged both unreadable sources into one test."""
    abstain = _criterion(INSUFFICIENT_EVIDENCE)
    assert abstain != "The claim or the receipt does not show a total that can be read."
    assert "receipt" in abstain.lower()
    assert abstain.lower().index("claim") < abstain.lower().index("receipt")


@pytest.mark.parametrize("label", COMPARE_LABELS)
def test_compare_criteria_require_every_claimed_digit(label: str) -> None:
    """``mismatch`` and ``match`` may not be chosen from a masked amount."""
    criterion = _criterion(label).lower()
    assert "every digit" in criterion


@pytest.mark.parametrize("label", COMPARE_LABELS)
def test_compare_criteria_still_state_their_comparison(label: str) -> None:
    """The legibility precondition must not replace the amount comparison."""
    criterion = _criterion(label).lower()
    assert "receipt total" in criterion
