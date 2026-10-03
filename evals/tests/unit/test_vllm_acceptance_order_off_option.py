"""Offline proof that every scored acceptance row records the off-option guard.

The order set reruns the CORD combined arm with the label order reversed. It
sends ``CORD_OFF_OPTION_THRESHOLD``, and every row that ``_score`` builds
carries ``off_option_mass`` and ``off_option_flag`` ([#409][i409]). A
recording port stands in for the vLLM port, because the mock vLLM server
cannot report a non-null mass.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_vllm_acceptance_order_off_option.py
    ```

See Also:
    - [typevet_evals.vllm_acceptance.sets][]: order and PSAI set runners

[i409]: https://github.com/Alberto-Codes/typevet/issues/409
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer
from typevet.domain.judgment_questions import Choice
from typevet.domain.judgment_response import JudgmentResponse, OffOptionReceipt
from typevet.domain.media import ImageInput
from typevet_evals.vllm_acceptance.core import RunState
from typevet_evals.vllm_acceptance.sets import CORD_OFF_OPTION_THRESHOLD, SET_RUNNERS

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"
_MASS = 0.4
_OFF_OPTION_KEYS = {"off_option_mass", "off_option_flag"}


class _RecordingPort:
    """Port that records each threshold and can report a fixed mass.

    Attributes:
        thresholds (list[float | None]): Threshold of each ``judge`` call.
        with_receipt (bool): Whether each response carries a receipt.
    """

    def __init__(self, *, with_receipt: bool) -> None:
        self.thresholds: list[float | None] = []
        self.with_receipt = with_receipt

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, object],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Record the threshold and answer each question with its first label.

        Args:
            state: State under evaluation.
            questions: Named native questions.
            model: Model id.
            media: Images.
            off_option_threshold: Threshold to record and to evaluate.

        Returns:
            A response with one answer per question and, when
            ``with_receipt`` is set, one off-option receipt per question.
        """
        self.thresholds.append(off_option_threshold)
        answers: dict[str, Any] = {}
        for name, question in questions.items():
            if isinstance(question, Choice):
                labels = list(question.criteria)
                probs = {label: 0.0 for label in labels} | {labels[0]: 1.0}
                answers[name] = ChoiceAnswer(
                    choice=labels[0], confidence=1.0, probabilities=probs
                )
            else:
                answers[name] = NoulAnswer(noul=0.9)
        receipts = {}
        if self.with_receipt:
            receipt = OffOptionReceipt.evaluate(
                mass=_MASS, threshold=off_option_threshold
            )
            receipts = dict.fromkeys(questions, receipt)
        return JudgmentResponse(model=model, answers=answers, off_option=receipts)


def _run(port: _RecordingPort, *names: str) -> dict[str, dict[str, Any]]:
    """Run the named sets in order on one run state.

    Args:
        port: Port for every set.
        *names: Set names in run order.

    Returns:
        The per-set results.
    """
    run = RunState(port, MagicMock(), "fake", _FIXTURES, MagicMock(), {}, MagicMock())
    runners = dict(SET_RUNNERS)
    for name in names:
        run.sets[name] = {}
        runners[name](run, run.sets[name])
    return run.sets


def test_order_set_passes_the_threshold_to_the_cord_rerun() -> None:
    """Every call of the order set's CORD rerun sends 0.25."""
    port = _RecordingPort(with_receipt=True)
    run = RunState(port, MagicMock(), "fake", _FIXTURES, MagicMock(), {}, MagicMock())
    runners = dict(SET_RUNNERS)
    run.sets["cord"], run.sets["order"] = {}, {}
    runners["cord"](run, run.sets["cord"])
    port.thresholds.clear()
    runners["order"](run, run.sets["order"])
    assert port.thresholds
    assert all(t == CORD_OFF_OPTION_THRESHOLD for t in port.thresholds), port.thresholds


def test_every_score_row_carries_the_off_option_fields() -> None:
    """Rows carry the receipt, or None and False; one set has one key set."""
    with_receipt = _run(_RecordingPort(with_receipt=True), "cord", "order", "psai")
    without = _run(_RecordingPort(with_receipt=False), "cord", "order", "psai")
    for row in with_receipt["order"]["combined"].values():
        assert row["off_option_mass"] == _MASS
        assert row["off_option_flag"] is True
    for row in with_receipt["psai"]["rows"]:
        assert row["off_option_mass"] == _MASS
        assert row["off_option_flag"] is False
    for sets in (with_receipt, without):
        order_rows = list(sets["order"]["combined"].values())
        psai_rows = sets["psai"]["rows"]
        for rows in (order_rows, psai_rows):
            assert rows
            keys = {frozenset(row) for row in rows}
            assert len(keys) == 1, keys
            assert set(next(iter(keys))) >= _OFF_OPTION_KEYS
    for row in [*without["order"]["combined"].values(), *without["psai"]["rows"]]:
        assert row["off_option_mass"] is None
        assert row["off_option_flag"] is False
