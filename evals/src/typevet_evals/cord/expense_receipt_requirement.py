"""Application receipt requirement before CORD expense scoring ([#183][i183]).

When the smoke mode requires a receipt and the caller omits image bytes, the
harness returns ``insufficient_evidence`` without a ``JudgmentPort`` call. That
path is application validation, not model abstention.

Examples:
    ```python
    from typevet_evals.cord.expense_receipt_requirement import (
        evaluate_receipt_requirement,
        judge_cord_expense_arm,
    )

    gate = evaluate_receipt_requirement("combined", ())
    assert gate.label == "insufficient_evidence"
    assert gate.model_calls == 0
    ```

See Also:
    - [typevet.evaluation.datasets.cord_expense][]: ``expense_question``
    - [evals.tests.live.test_cord_expense_smoke_live][]: live wiring

[i183]: https://github.com/Alberto-Codes/typevet/issues/183
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.evaluation.datasets.cord_expense import (
    INSUFFICIENT_EVIDENCE,
    LABEL_ORDER,
    expense_question,
)

DETERMINISTIC_MISSING_RECEIPT: Final[str] = "deterministic_missing_receipt"
MODEL_ROUTING: Final[str] = "model"
_EXPENSE_FIELD: Final[str] = "expense"
_RECEIPT_REQUIRED_MODES: Final[frozenset[str]] = frozenset({"combined", "image_only"})


@dataclass(frozen=True, slots=True)
class ReceiptRequirementOutcome:
    """Result of the receipt requirement gate before scoring.

    Attributes:
        label (str | None): Deterministic label when set; ``None`` to score.
        application_mode (str): Smoke modality (``text_only``, ``image_only``,
            ``combined``).
        receipt_required (bool): Whether this mode requires receipt bytes.
        routing (str): ``deterministic_missing_receipt`` or ``model``.
        model_calls (int): Scoring calls already performed (always ``0`` here).
        deterministic_abstain (bool): True when the label is not from the model.

    Examples:
        ```python
        outcome = evaluate_receipt_requirement("text_only", ())
        assert outcome.label is None
        assert outcome.receipt_required is False
        ```
    """

    label: str | None
    application_mode: str
    receipt_required: bool
    routing: str
    model_calls: int
    deterministic_abstain: bool


class _ExpenseJudgmentPort(Protocol):
    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, object],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Score one state with native questions and optional media."""
        ...


def receipt_required_for_mode(application_mode: str) -> bool:
    """Return whether the smoke mode requires attached receipt bytes.

    Args:
        application_mode: One of ``text_only``, ``image_only``, ``combined``.

    Returns:
        True for ``combined`` and ``image_only``.
    """
    return application_mode in _RECEIPT_REQUIRED_MODES


def evaluate_receipt_requirement(
    application_mode: str,
    media: tuple[ImageInput, ...] | None,
) -> ReceiptRequirementOutcome:
    """Decide whether to short-circuit before ``JudgmentPort`` scoring.

    Args:
        application_mode: Smoke modality under test.
        media: Attached receipt images, if any.

    Returns:
        Outcome with a deterministic label when media is required but missing;
        otherwise ``label`` is ``None`` and the caller should score once.
    """
    images = media or ()
    required = receipt_required_for_mode(application_mode)
    if required and not images:
        return ReceiptRequirementOutcome(
            label=INSUFFICIENT_EVIDENCE,
            application_mode=application_mode,
            receipt_required=True,
            routing=DETERMINISTIC_MISSING_RECEIPT,
            model_calls=0,
            deterministic_abstain=True,
        )
    return ReceiptRequirementOutcome(
        label=None,
        application_mode=application_mode,
        receipt_required=required,
        routing=MODEL_ROUTING,
        model_calls=0,
        deterministic_abstain=False,
    )


def _deterministic_probabilities() -> dict[str, float]:
    return {
        label: 1.0 if label == INSUFFICIENT_EVIDENCE else 0.0 for label in LABEL_ORDER
    }


def _row_from_outcome(
    outcome: ReceiptRequirementOutcome,
    *,
    label: str,
    probabilities: dict[str, float],
    tokens_evaluated: int | None,
    seconds: float,
    model_calls: int,
) -> dict[str, object]:
    return {
        "label": label,
        "probabilities": probabilities,
        "tokens_evaluated": tokens_evaluated,
        "seconds": seconds,
        "application_mode": outcome.application_mode,
        "receipt_required": outcome.receipt_required,
        "routing": outcome.routing,
        "model_calls": model_calls,
        "deterministic_abstain": outcome.deterministic_abstain,
    }


def judge_cord_expense_arm(
    port: _ExpenseJudgmentPort,
    model: str,
    state: str,
    media: tuple[ImageInput, ...] | None,
    *,
    application_mode: str,
    question_name: str = _EXPENSE_FIELD,
) -> dict[str, object]:
    """Score one CORD expense arm or return a deterministic missing-receipt row.

    Args:
        port: Judgment port (not called when receipt bytes are required but absent).
        model: Backend model id.
        state: Claim or control text under evaluation.
        media: Receipt images for ``combined`` / ``image_only`` arms.
        application_mode: Smoke modality name.
        question_name: Field id for ``expense_question()``.

    Returns:
        Row dict with label, probabilities, usage metadata and provenance fields.
    """
    gate = evaluate_receipt_requirement(application_mode, media)
    if gate.label is not None:
        return _row_from_outcome(
            gate,
            label=gate.label,
            probabilities=_deterministic_probabilities(),
            tokens_evaluated=0,
            seconds=0.0,
            model_calls=0,
        )
    started = time.perf_counter()
    images = media or ()
    response = port.judge(
        state,
        {question_name: expense_question()},
        model,
        media=images,
    )
    answer = response.choices[question_name]
    return _row_from_outcome(
        gate,
        label=answer.choice,
        probabilities=dict(answer.probabilities),
        tokens_evaluated=response.usage.input_tokens,
        seconds=round(time.perf_counter() - started, 3),
        model_calls=1,
    )
