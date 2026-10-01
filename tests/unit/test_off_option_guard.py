"""Unit tests for the opt-in off-option mass guard (#353)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from typevet.adapters.outbound.judgment_scoring import (
    ScoringJudgmentAdapter,
    judge_with_scoring,
)
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import (
    apply_off_option_threshold,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.domain.errors import DecisionExecutionError, JudgmentValidationError
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.testing import ScriptedScoringFake

_LOGPROBS = {"billing": math.log(0.3), "technical": math.log(0.1)}
_NOUL = {"True": math.log(0.3), "False": math.log(0.1)}
_LEVELS = {"0": math.log(0.3), "1": math.log(0.1), "2": math.log(0.05)}
_CHOICE = {"a": math.log(0.3), "b": math.log(0.1)}


def _execute(fake: ScriptedScoringFake, threshold: Any = None) -> Any:
    decision = Decision("c", "Pick.", ("billing", "technical"), syntax="Choice")
    candidates = (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    executed = execute_categorical_decision(
        decision,
        prefix="Answer:",
        candidates=candidates,
        port=fake,
        model="fake",
    )
    return apply_off_option_threshold(executed, threshold)


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),)


@pytest.mark.unit
def test_mass_above_threshold_sets_flag_and_receipt() -> None:
    fake = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=0.4)
    result = _execute(fake, threshold=0.3)
    assert result.off_option.off_option_flag is True
    assert result.off_option.off_option_mass == 0.4
    assert result.off_option.off_option_threshold == 0.3


@pytest.mark.unit
def test_default_threshold_is_off_and_records_mass() -> None:
    fake = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=0.6)
    result = _execute(fake)
    assert result.off_option.off_option_mass == 0.6
    assert result.off_option.off_option_flag is False
    assert result.off_option.off_option_threshold is None


@pytest.mark.unit
def test_mass_equal_to_threshold_does_not_flag() -> None:
    fake = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=0.3)
    assert _execute(fake, threshold=0.3).off_option.off_option_flag is False


@pytest.mark.unit
def test_unavailable_mass_never_flags_and_receipt_says_null() -> None:
    fake = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=None)
    receipt = _execute(fake, threshold=0.0).off_option
    assert receipt.off_option_flag is False
    assert receipt.as_dict() == {
        "off_option_mass": None,
        "off_option_threshold": 0.0,
        "off_option_flag": False,
    }
    assert '"off_option_mass": null' in json.dumps(receipt.as_dict())


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [-0.1, 1.5, math.nan, math.inf, True, "0.2"])
def test_invalid_threshold_is_rejected(threshold: Any) -> None:
    fake = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=0.4)
    with pytest.raises(DecisionExecutionError, match="off_option_threshold"):
        _execute(fake, threshold=threshold)
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=_tokenize)
    calls_before = len(fake.calls)
    with pytest.raises(JudgmentValidationError, match="off_option_threshold"):
        adapter.judge("text", {"n": Noul()}, "fake", off_option_threshold=threshold)
    assert len(fake.calls) == calls_before


@pytest.mark.unit
def test_answers_default_to_unflagged() -> None:
    assert NoulAnswer(noul=0.5).off_option_flag is False
    choice = ChoiceAnswer(choice="a", confidence=1.0, probabilities={"a": 1.0})
    assert choice.off_option_flag is False
    score = ScoreAnswer(
        score=0.0, confidence=1.0, legend={0: "x"}, probabilities={0: 1.0}
    )
    assert score.off_option_flag is False


@pytest.mark.unit
def test_answer_rejects_non_bool_flag() -> None:
    flag: Any = 1
    with pytest.raises(TypeError, match="off_option_flag"):
        NoulAnswer(noul=0.5, off_option_flag=flag)


@pytest.mark.unit
@pytest.mark.parametrize(("mass", "flag"), [(0.4, True), (0.1, False), (None, False)])
def test_judge_flags_every_answer_type(mass: float | None, flag: bool) -> None:
    fake = ScriptedScoringFake(
        logprobs={**_NOUL, **_LEVELS, **_CHOICE},
        off_option_mass=mass,
    )
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=_tokenize)
    questions = {
        "n": Noul(instructions="Is it?"),
        "c": Choice(criteria={"a": "A", "b": "B"}),
        "s": Score(criteria=["Poor", "Fair", "Good"]),
    }
    response = adapter.judge("text", questions, "fake", off_option_threshold=0.2)
    for name in questions:
        assert response.answers[name].off_option_flag is flag
        receipt = response.off_option[name]
        assert receipt.off_option_mass == mass
        assert receipt.off_option_threshold == 0.2
        assert receipt.off_option_flag is flag


@pytest.mark.unit
def test_judge_without_threshold_keeps_today_behaviour() -> None:
    fake = ScriptedScoringFake(logprobs=_NOUL, off_option_mass=0.55)
    response = judge_with_scoring(
        "text",
        {"n": Noul()},
        "fake",
        scoring_port=fake,
        tokenize_content=_tokenize,
    )
    assert response.answers["n"].off_option_flag is False
    assert response.off_option["n"].as_dict() == {
        "off_option_mass": 0.55,
        "off_option_threshold": None,
        "off_option_flag": False,
    }


@pytest.mark.unit
def test_judge_with_scoring_forwards_threshold() -> None:
    fake = ScriptedScoringFake(logprobs=_NOUL, off_option_mass=0.5)
    response = judge_with_scoring(
        "text",
        {"n": Noul()},
        "fake",
        scoring_port=fake,
        tokenize_content=_tokenize,
        off_option_threshold=0.25,
    )
    assert response.answers["n"].off_option_flag is True
    assert response.off_option["n"].off_option_threshold == 0.25
