"""End-to-end native Choice and Score capacity through ``judge_with_scoring``."""

from __future__ import annotations

import pytest

from typevet.adapters.outbound.judgment_scoring import judge_with_scoring
from typevet.domain.errors import DecisionExecutionError
from typevet.domain.judgment_answers import ChoiceAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Score
from typevet.testing import ScriptedScoringFake

_LOW = -40.0


def _char_tokenize(text: str) -> tuple[int, ...]:
    # One token per character: "0"-"9" and "A"-"Z" are single tokens.
    return tuple(ord(char) for char in text)


def _choice(count: int) -> Choice:
    criteria = {f"label{i}": f"Option {i}" for i in range(count)}
    return Choice(instructions="Pick one.", criteria=criteria)


@pytest.mark.unit
def test_judge_24_option_choice_binds_last_label_to_control_n() -> None:
    logprobs = {f"label{i}": _LOW for i in range(24)}
    logprobs["label23"] = 0.0
    fake = ScriptedScoringFake(logprobs=logprobs)
    response = judge_with_scoring(
        "state",
        {"pick": _choice(24)},
        "fake-model",
        scoring_port=fake,
        tokenize_content=_char_tokenize,
    )
    answer = response.answers["pick"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "label23"
    (request,) = fake.calls
    last = request.candidates[-1]
    assert (last.label, last.token_ids) == ("label23", (ord("N"),))
    assert "\nN → label23: Option 23" in request.prefix


@pytest.mark.unit
def test_judge_25_option_choice_fails_at_execute_cap() -> None:
    fake = ScriptedScoringFake(logprobs={f"label{i}": 0.0 for i in range(25)})
    with pytest.raises(DecisionExecutionError, match="between 2 and 24, got 25"):
        judge_with_scoring(
            "state",
            {"pick": _choice(25)},
            "fake-model",
            scoring_port=fake,
            tokenize_content=_char_tokenize,
        )
    assert fake.calls == []


@pytest.mark.unit
def test_judge_12_level_score_expected_level_uses_level_number() -> None:
    levels = 12
    logprobs = {str(level): _LOW for level in range(levels)}
    logprobs["11"] = 0.0
    fake = ScriptedScoringFake(logprobs=logprobs)
    question = Score(
        instructions="Rate it.",
        criteria=[f"Level {level}" for level in range(levels)],
    )
    response = judge_with_scoring(
        "state",
        {"rate": question},
        "fake-model",
        scoring_port=fake,
        tokenize_content=_char_tokenize,
    )
    answer = response.answers["rate"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score == pytest.approx(11.0, abs=1e-6)
    (request,) = fake.calls
    last = request.candidates[-1]
    assert (last.label, last.token_ids) == ("11", (ord("B"),))
    assert "\nControl B → 11" in request.prefix
