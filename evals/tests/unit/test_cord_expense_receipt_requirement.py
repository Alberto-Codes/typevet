"""Offline CORD receipt requirement gate before scoring ([#183][i183]).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_cord_expense_receipt_requirement.py
    ```

See Also:
    - [typevet_evals.cord.expense_receipt_requirement][]: mode gate
    - [typevet_evals.datasets.cord_expense][]: three-label question

[i183]: https://github.com/Alberto-Codes/typevet/issues/183
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from typevet.domain.judgment_answers import ChoiceAnswer
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import ImageInput
from typevet_evals.cord.expense_receipt_requirement import (
    DETERMINISTIC_MISSING_RECEIPT,
    MODEL_ROUTING,
    judge_cord_expense_arm,
)
from typevet_evals.datasets.cord_expense import (
    INSUFFICIENT_EVIDENCE,
    MATCH,
    MISMATCH,
    expense_question,
)

pytestmark = pytest.mark.unit

_PNG = ImageInput(data=b"\x89PNG\r\n", mime_type="image/png")
_STATEMENT = "Expense claim: total 80,500."


def test_combined_missing_receipt_abstains_with_zero_port_calls() -> None:
    """Combined with required receipt and no media never calls the port."""
    port = MagicMock()
    row = judge_cord_expense_arm(
        port,
        "fake",
        _STATEMENT,
        (),
        application_mode="combined",
    )
    port.judge.assert_not_called()
    assert row["label"] == INSUFFICIENT_EVIDENCE
    assert row["model_calls"] == 0
    assert row["routing"] == DETERMINISTIC_MISSING_RECEIPT
    assert row["deterministic_abstain"] is True
    assert row["receipt_required"] is True
    assert row["application_mode"] == "combined"


def test_text_only_still_invokes_port_when_receipt_not_required() -> None:
    """Text-only mode may score without a receipt image."""
    answer = ChoiceAnswer(
        choice=MATCH,
        confidence=0.9,
        probabilities={
            INSUFFICIENT_EVIDENCE: 0.05,
            MISMATCH: 0.05,
            MATCH: 0.9,
        },
    )
    port = MagicMock(
        judge=MagicMock(
            return_value=JudgmentResponse(
                model="fake",
                usage=TokenUsage(input_tokens=42, output_tokens=0),
                answers={"expense": answer},
            )
        )
    )
    row = judge_cord_expense_arm(
        port,
        "fake",
        _STATEMENT,
        (),
        application_mode="text_only",
    )
    port.judge.assert_called_once()
    assert row["model_calls"] == 1
    assert row["routing"] == MODEL_ROUTING
    assert row["deterministic_abstain"] is False
    assert row["receipt_required"] is False
    assert row["label"] == MATCH


def test_combined_with_image_uses_direct_three_label_choice() -> None:
    """Combined with media uses one direct ``expense_question()`` scoring call."""
    answer = ChoiceAnswer(
        choice=MATCH,
        confidence=0.8,
        probabilities={
            INSUFFICIENT_EVIDENCE: 0.1,
            MISMATCH: 0.1,
            MATCH: 0.8,
        },
    )
    port = MagicMock(
        judge=MagicMock(
            return_value=JudgmentResponse(
                model="fake",
                usage=TokenUsage(input_tokens=100, output_tokens=0),
                answers={"expense": answer},
            )
        )
    )
    row = judge_cord_expense_arm(
        port,
        "fake",
        _STATEMENT,
        (_PNG,),
        application_mode="combined",
    )
    port.judge.assert_called_once()
    _args, kwargs = port.judge.call_args
    questions = _args[1]
    assert set(questions) == {"expense"}
    expected = expense_question()
    asked = questions["expense"]
    assert asked.criteria == expected.criteria
    assert asked.instructions == expected.instructions
    assert kwargs.get("media") == (_PNG,)
    assert row["model_calls"] == 1
    assert row["routing"] == MODEL_ROUTING
    assert row["deterministic_abstain"] is False
    assert row["label"] == MATCH


class _RecordingPort:
    """Expense port that records each ``off_option_threshold`` it receives.

    Attributes:
        thresholds (list[float | None]): Threshold of each ``judge`` call.
    """

    def __init__(self) -> None:
        self.thresholds: list[float | None] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, object],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Record the threshold and return one ``match`` expense answer.

        Args:
            state: Claim text under evaluation.
            questions: Named native questions.
            model: Model id.
            media: Receipt images.
            off_option_threshold: Threshold to record.

        Returns:
            A response with one ``match`` answer per question.
        """
        self.thresholds.append(off_option_threshold)
        answer = ChoiceAnswer(
            choice=MATCH,
            confidence=1.0,
            probabilities={INSUFFICIENT_EVIDENCE: 0.0, MISMATCH: 0.0, MATCH: 1.0},
        )
        return JudgmentResponse(model=model, answers=dict.fromkeys(questions, answer))


def test_arm_forwards_off_option_threshold_to_the_port() -> None:
    """The arm passes ``off_option_threshold`` to the port, None by default."""
    port = _RecordingPort()
    judge_cord_expense_arm(
        port,
        "fake",
        _STATEMENT,
        (_PNG,),
        application_mode="combined",
        off_option_threshold=0.25,
    )
    judge_cord_expense_arm(
        port, "fake", _STATEMENT, (_PNG,), application_mode="combined"
    )
    assert port.thresholds == [0.25, None]
