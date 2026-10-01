"""Contract: evals judgment ports carry ``off_option_threshold`` (#368).

The ledger and budget wrappers forward the threshold to the scoring adapter.
The CORD expense port protocol accepts it. Each path flags a 0.3 off-option
mass at 0.25.

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_off_option_threshold_evals_ports.py
    ```

See Also:
    - [typevet.ports.judgment][]: ``JudgmentPort`` protocol
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from typevet.domain.judgment_questions import Noul
from typevet.domain.judgment_response import JudgmentResponse, OffOptionReceipt
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake
from typevet_evals.cord.expense_receipt_requirement import _ExpenseJudgmentPort
from typevet_evals.instruction_variant.matrix import _LedgerJudgmentPort
from typevet_evals.instruction_variant.protocol import VariantDispatchLedger
from typevet_evals.psai_vision_consumer.dispatch import (
    ConsumerDispatchLedger,
    wrap_judgment_port,
)

pytestmark = pytest.mark.contract

_MODEL = "fake-gemma"


def _scoring_port() -> ScoringJudgmentAdapter:
    """Build a scoring adapter whose answers carry an off-option mass of 0.3.

    Returns:
        An offline judgment port over ``ScriptedScoringFake``.
    """
    fake = ScriptedScoringFake(
        logprobs={"True": math.log(0.5), "False": math.log(0.2)},
        off_option_mass=0.3,
    )
    return ScoringJudgmentAdapter(fake, tokenize_content=lambda text: (ord(text),))


def _judge(port: JudgmentPort, threshold: float | None) -> JudgmentResponse:
    return port.judge(
        "Charged twice.",
        {"q": Noul(instructions="Billing issue?")},
        _MODEL,
        off_option_threshold=threshold,
    )


def _ports() -> list[JudgmentPort]:
    return [
        _LedgerJudgmentPort(_scoring_port(), VariantDispatchLedger()),
        wrap_judgment_port(_scoring_port(), ConsumerDispatchLedger()),
    ]


@pytest.mark.parametrize(("threshold", "flag"), [(0.25, True), (None, False)])
def test_ledger_and_budget_wrappers_forward_threshold(
    threshold: float | None, flag: bool
) -> None:
    for port in _ports():
        receipt = _judge(port, threshold).off_option["q"]
        assert receipt.off_option_mass == pytest.approx(0.3)
        assert receipt.off_option_threshold == threshold
        assert receipt.off_option_flag is flag


class _ExpenseFake:
    """CORD expense port that flags a 0.3 off-option mass at the threshold.

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
        """Record the threshold and return one flagged receipt per question.

        Args:
            state: Content under evaluation.
            questions: Named questions.
            model: Model id.
            media: Receipt images.
            off_option_threshold: Threshold to record and apply.

        Returns:
            A response with one receipt per question and no answers.
        """
        self.thresholds.append(off_option_threshold)
        receipt = OffOptionReceipt.evaluate(mass=0.3, threshold=off_option_threshold)
        return JudgmentResponse(
            model=model, off_option=dict.fromkeys(questions, receipt)
        )


def _judge_expense(port: _ExpenseJudgmentPort) -> JudgmentResponse:
    return port.judge(
        "Lunch receipt.",
        {"q": Noul(instructions="Expense?")},
        _MODEL,
        off_option_threshold=0.25,
    )


def test_expense_port_protocol_accepts_threshold() -> None:
    fake = _ExpenseFake()
    response = _judge_expense(fake)
    assert fake.thresholds == [0.25]
    assert response.off_option["q"].off_option_flag is True
