"""Unit tests: CategoricalExecutionResult → typed judgment answers (#122 rev2)."""

from __future__ import annotations

import math

import pytest

from typevet.adapters.outbound.judgment_scoring import (
    answer_from_execution,
    bind_candidates_for_execute,
)
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import CategoricalExecutionResult
from typevet.domain.decisions import Decision
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_normalize import normalize_question
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.domain.judgment_response import TokenUsage


def _single_char_tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def _result(
    decision: Decision,
    *,
    probs: tuple[float, ...],
) -> CategoricalExecutionResult:
    pairs = tuple((decision.choices[i], probs[i]) for i in range(len(probs)))
    idx = max(range(len(probs)), key=probs.__getitem__)
    return CategoricalExecutionResult(
        decision=decision,
        value=decision.choices[idx],
        probabilities=pairs,
        logprobs=tuple(math.log(p) for p in probs),
        model="m",
        usage=TokenUsage(),
    )


@pytest.mark.unit
def test_noul_answer_uses_probability_of_true() -> None:
    question = Noul(instructions="Yes?")
    decision = normalize_question(question, field_name="q")
    # Choices are false-first: (False, True); keep P(true) at 0.7.
    result = _result(decision, probs=(0.3, 0.7))
    answer = answer_from_execution(question, result)
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == pytest.approx(0.7)


@pytest.mark.unit
def test_choice_answer_uses_original_label_not_control_glyph() -> None:
    question = Choice(
        criteria={"billing": "Money", "technical": "Bugs"},
        instructions="Pick:",
    )
    decision = normalize_question(question, field_name="route")
    specs = bind_candidates_for_execute(decision, question, _single_char_tokenize)
    assert [s.label for s in specs] == ["billing", "technical"]
    assert all(len(s.token_ids) == 1 for s in specs)
    result = _result(decision, probs=(0.8, 0.2))
    answer = answer_from_execution(question, result)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "billing"
    assert answer.confidence == pytest.approx(0.8)
    assert answer.probabilities == {
        "billing": pytest.approx(0.8),
        "technical": pytest.approx(0.2),
    }


@pytest.mark.unit
def test_score_expected_value_differs_from_modal_level() -> None:
    question = Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:")
    decision = normalize_question(question, field_name="quality")
    probs = (0.45, 0.45, 0.10)
    result = _result(decision, probs=probs)
    answer = answer_from_execution(question, result)
    assert isinstance(answer, ScoreAnswer)
    assert answer.confidence == pytest.approx(0.45)
    assert answer.score == pytest.approx(0.0 * 0.45 + 1.0 * 0.45 + 2.0 * 0.10)
    assert answer.score != float(result.value)
    assert answer.legend == {0: "Poor", 1: "Fair", 2: "Good"}


@pytest.mark.unit
def test_bind_noul_candidates_use_execute_bool_labels() -> None:
    question = Noul()
    decision = normalize_question(question, field_name="flag")
    specs = bind_candidates_for_execute(decision, question, _single_char_tokenize)
    assert specs == (
        CandidateTokenSpec("False", (ord("0"),)),
        CandidateTokenSpec("True", (ord("1"),)),
    )
