"""TPJEP probability validity for outcome mapping and runner records (#150)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import ImageInput
from typevet.evaluation.tpjep.loader import TpjepScheduledTask
from typevet.evaluation.tpjep.outcome import outcome_from_answer, prob_valid
from typevet.evaluation.tpjep.runner import TpjepRunConfig, run_tpjep_tasks
from typevet.ports.judgment import JudgmentPort

_LABELS_AB = ("A", "B")


def _choice_task(*, labels: tuple[str, ...] = _LABELS_AB) -> TpjepScheduledTask:
    criteria = {key: f"opt {key}" for key in labels}
    return TpjepScheduledTask(
        task_id="test-choice-0",
        source_tier="easy",
        question_type="Choice",
        state="state",
        question_name="answer",
        question=Choice(criteria=criteria, instructions="pick"),
        expected=labels[0],
        labels=labels,
    )


def _noul_task(*, labels: tuple[str, str] = ("no", "yes")) -> TpjepScheduledTask:
    return TpjepScheduledTask(
        task_id="test-noul-0",
        source_tier="easy",
        question_type="Noul",
        state="state",
        question_name="answer",
        question=Noul(instructions="yes or no"),
        expected=True,
        labels=labels,
    )


def _choice_answer_from_wire(
    *,
    choice: str,
    confidence: float,
    probabilities: dict[str, float],
) -> ChoiceAnswer:
    """Simulate wire payloads that skip native ``ChoiceAnswer`` bounds checks."""
    answer = object.__new__(ChoiceAnswer)
    object.__setattr__(answer, "choice", choice)
    object.__setattr__(answer, "confidence", confidence)
    object.__setattr__(answer, "probabilities", probabilities)
    return answer


class _ScriptedJudgmentPort:
    """Return one scripted answer without criteria-key checks."""

    def __init__(self, answer: Answer) -> None:
        self._answer = answer

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        del state, media
        return JudgmentResponse(
            model=model,
            usage=TokenUsage(),
            answers=dict.fromkeys(questions, self._answer),
        )


def _score_task(*, n_levels: int = 3) -> TpjepScheduledTask:
    criteria = [f"level {i}" for i in range(n_levels)]
    labels = tuple(str(i) for i in range(n_levels))
    legend = {i: criteria[i] for i in range(n_levels)}
    _ = legend
    return TpjepScheduledTask(
        task_id="test-score-0",
        source_tier="easy",
        question_type="Score",
        state="state",
        question_name="answer",
        question=Score(criteria=criteria, instructions="rate"),
        expected=1,
        labels=labels,
    )


@pytest.mark.unit
def test_prob_valid_accepts_in_range_sum_and_exact_label_coverage() -> None:
    probs = {"A": 0.6, "B": 0.4}
    assert prob_valid(probs, labels=_LABELS_AB) is True


@pytest.mark.unit
def test_prob_valid_rejects_compensating_out_of_range_values() -> None:
    probs = {"A": 1.5, "B": -0.5}
    assert prob_valid(probs, labels=_LABELS_AB) is False


@pytest.mark.unit
def test_outcome_from_answer_marks_compensating_out_of_range_invalid() -> None:
    task = _choice_task()
    answer = _choice_answer_from_wire(
        choice="A",
        confidence=0.5,
        probabilities={"A": 1.5, "B": -0.5},
    )
    predicted, probs, correct = outcome_from_answer(task, answer)
    assert predicted == "A"
    assert prob_valid(probs, labels=task.labels) is False
    assert correct is None


@pytest.mark.unit
def test_runner_excludes_compensating_out_of_range_from_accuracy() -> None:
    task = _choice_task()
    answer = _choice_answer_from_wire(
        choice="A",
        confidence=0.5,
        probabilities={"A": 1.5, "B": -0.5},
    )
    port: JudgmentPort = _ScriptedJudgmentPort(answer)
    records = run_tpjep_tasks(port, [task], config=TpjepRunConfig(model="fake"))
    record = records[0]
    assert record.outcome == "prob_invalid"
    assert record.prob_valid is False
    assert record.correct is None


@pytest.mark.unit
def test_prob_valid_rejects_non_finite_values() -> None:
    probs = {"A": math.nan, "B": 0.0}
    assert prob_valid(probs, labels=_LABELS_AB) is False


@pytest.mark.unit
def test_prob_valid_rejects_sum_outside_tolerance() -> None:
    probs = {"A": 0.6, "B": 0.35}
    assert prob_valid(probs, labels=_LABELS_AB) is False


@pytest.mark.unit
def test_prob_valid_rejects_wrong_label_keys() -> None:
    probs = {"X": 0.5, "Y": 0.5}
    assert prob_valid(probs, labels=_LABELS_AB) is False


@pytest.mark.unit
def test_outcome_from_answer_marks_wrong_label_choice_invalid() -> None:
    task = _choice_task()
    answer = ChoiceAnswer(
        choice="X",
        confidence=0.5,
        probabilities={"X": 0.5, "Y": 0.5},
    )
    predicted, probs, correct = outcome_from_answer(task, answer)
    assert predicted == "X"
    assert correct is None
    assert prob_valid(probs, labels=task.labels) is False


@pytest.mark.unit
def test_runner_excludes_wrong_label_choice_from_accuracy() -> None:
    task = _choice_task()
    answer = ChoiceAnswer(
        choice="X",
        confidence=0.5,
        probabilities={"X": 0.5, "Y": 0.5},
    )
    port: JudgmentPort = _ScriptedJudgmentPort(answer)
    records = run_tpjep_tasks(port, [task], config=TpjepRunConfig(model="fake"))
    record = records[0]
    assert record.outcome == "prob_invalid"
    assert record.prob_valid is False
    assert record.correct is None


@pytest.mark.unit
def test_noul_outcome_validates_scheduled_label_keys() -> None:
    task = _noul_task()
    _, probs, correct = outcome_from_answer(task, NoulAnswer(noul=0.7))
    assert prob_valid(probs, labels=task.labels) is True
    assert correct is not None


@pytest.mark.unit
def test_score_outcome_validates_scheduled_label_keys() -> None:
    task = _score_task()
    answer = ScoreAnswer(
        score=1.0,
        confidence=0.9,
        legend={0: "l0", 1: "l1", 2: "l2"},
        probabilities={0: 0.1, 1: 0.8, 2: 0.1},
    )
    _, probs, correct = outcome_from_answer(task, answer)
    assert prob_valid(probs, labels=task.labels) is True
    assert correct is True
