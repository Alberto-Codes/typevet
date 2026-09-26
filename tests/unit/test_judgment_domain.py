"""Unit tests for judgment domain types."""

from __future__ import annotations

from typing import Any, cast

import pytest

from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Noul, question_types
from typevet.domain.judgment_response import JudgmentResponse


@pytest.mark.unit
def test_noul_answer_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="noul"):
        NoulAnswer(noul=1.5)


@pytest.mark.unit
def test_noul_answer_rejects_bool() -> None:
    with pytest.raises(TypeError, match="noul"):
        NoulAnswer(noul=cast(Any, True))


@pytest.mark.unit
def test_choice_answer_requires_probabilities_sum() -> None:
    with pytest.raises(ValueError, match="sum"):
        ChoiceAnswer(
            choice="a",
            confidence=0.5,
            probabilities={"a": 0.4, "b": 0.4},
        )


@pytest.mark.unit
def test_choice_answer_requires_selected_key() -> None:
    with pytest.raises(ValueError, match="not in probabilities"):
        ChoiceAnswer(
            choice="a",
            confidence=1.0,
            probabilities={"b": 1.0},
        )


@pytest.mark.unit
def test_score_answer_legend_keys_must_match() -> None:
    with pytest.raises(ValueError, match="legend keys"):
        ScoreAnswer(
            score=1.0,
            confidence=1.0,
            legend={0: "a", 1: "b"},
            probabilities={0: 1.0},
        )


@pytest.mark.unit
def test_question_types_maps_typed_and_wire() -> None:
    typed = question_types(
        {
            "n": Noul(),
            "c": Choice(criteria={"x": "X"}),
            "w": {"type": "score", "criteria": []},
            "bad": {"type": 1},
        }
    )
    assert typed == {"n": "noul", "c": "choice", "w": "score", "bad": None}


@pytest.mark.unit
def test_judgment_response_typed_views() -> None:
    response = JudgmentResponse(
        model="m",
        answers={
            "n": NoulAnswer(noul=0.2),
            "c": ChoiceAnswer(
                choice="a",
                confidence=1.0,
                probabilities={"a": 1.0},
            ),
            "s": ScoreAnswer(
                score=0.0,
                confidence=1.0,
                legend={0: "low"},
                probabilities={0: 1.0},
            ),
        },
    )
    assert set(response.nouls) == {"n"}
    assert set(response.choices) == {"c"}
    assert set(response.scores) == {"s"}
