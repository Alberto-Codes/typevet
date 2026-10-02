"""Offline proof that the CORD acceptance set records the off-option guard.

The CORD set passes ``CORD_OFF_OPTION_THRESHOLD`` to each scored arm, and
each arm row carries ``off_option_mass`` and ``off_option_flag`` from the
response ([#384][i384]). A recording port stands in for the vLLM port,
because the mock vLLM server cannot report a non-null mass.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_vllm_acceptance_cord_off_option.py
    ```

See Also:
    - [typevet_evals.vllm_acceptance.sets][]: CORD set runner
    - [typevet_evals.cord.expense_receipt_requirement][]: CORD arm rows

[i384]: https://github.com/Alberto-Codes/typevet/issues/384
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from typevet.domain.errors import GenerationError
from typevet.domain.judgment_answers import ChoiceAnswer
from typevet.domain.judgment_response import JudgmentResponse, OffOptionReceipt
from typevet.domain.media import ImageInput
from typevet_evals.cord.expense_receipt_requirement import judge_cord_expense_arm
from typevet_evals.datasets.cord_expense import (
    INSUFFICIENT_EVIDENCE,
    MATCH,
    MISMATCH,
)
from typevet_evals.vllm_acceptance.core import RunState
from typevet_evals.vllm_acceptance.sets import (
    CORD_OFF_OPTION_THRESHOLD,
    SET_RUNNERS,
    failed_row,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
_MASS = 0.4


class _RecordingPort:
    """Expense port that records each threshold and reports a fixed mass.

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
        """Record the threshold and return a ``match`` answer with a receipt.

        Args:
            state: Claim text under evaluation.
            questions: Named native questions.
            model: Model id.
            media: Receipt images.
            off_option_threshold: Threshold to record and to evaluate.

        Returns:
            A response with one ``match`` answer and one off-option receipt
            per question.
        """
        self.thresholds.append(off_option_threshold)
        answer = ChoiceAnswer(
            choice=MATCH,
            confidence=1.0,
            probabilities={INSUFFICIENT_EVIDENCE: 0.0, MISMATCH: 0.0, MATCH: 1.0},
        )
        receipt = OffOptionReceipt.evaluate(mass=_MASS, threshold=off_option_threshold)
        return JudgmentResponse(
            model=model,
            answers=dict.fromkeys(questions, answer),
            off_option=dict.fromkeys(questions, receipt),
        )


def test_cord_set_passes_the_threshold_and_records_the_receipt() -> None:
    """Every scored arm gets 0.25; each combined row records mass and flag."""
    port = _RecordingPort()
    run = RunState(port, MagicMock(), "fake", _FIXTURES, MagicMock(), {}, MagicMock())
    out: dict[str, Any] = {}
    dict(SET_RUNNERS)["cord"](run, out)
    assert port.thresholds
    assert all(t == 0.25 for t in port.thresholds), port.thresholds
    assert out["combined"]
    for row in out["combined"].values():
        assert row["off_option_mass"] == _MASS
        assert row["off_option_flag"] is True
    omission = out["image_only_omission"]
    assert omission["off_option_mass"] == _MASS


def test_threshold_constant_is_a_quarter() -> None:
    """The CORD set uses the fixed 0.25 off-option threshold."""
    assert CORD_OFF_OPTION_THRESHOLD == 0.25


def test_deterministic_row_carries_no_mass_and_no_flag() -> None:
    """A missing-receipt row never calls the port and records None, False."""
    port = _RecordingPort()
    row = judge_cord_expense_arm(port, "fake", "claim", (), application_mode="combined")
    assert port.thresholds == []
    assert row["off_option_mass"] is None
    assert row["off_option_flag"] is False


def test_row_without_a_receipt_carries_no_mass_and_no_flag() -> None:
    """A response with no off-option receipt records None, False."""
    port = MagicMock()
    answer = ChoiceAnswer(
        choice=MATCH,
        confidence=1.0,
        probabilities={INSUFFICIENT_EVIDENCE: 0.0, MISMATCH: 0.0, MATCH: 1.0},
    )
    port.judge.return_value = JudgmentResponse(
        model="fake", answers={"expense": answer}
    )
    row = judge_cord_expense_arm(
        port, "fake", "claim", (), application_mode="text_only"
    )
    assert row["off_option_mass"] is None
    assert row["off_option_flag"] is False


def test_failed_row_carries_no_mass_and_no_flag() -> None:
    """A call that raised records None, False."""
    row = failed_row(0.0, GenerationError("boom"))
    assert row["off_option_mass"] is None
    assert row["off_option_flag"] is False
