"""Unit tests for native judgment question → Decision normalization."""

from __future__ import annotations

import pytest

from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_normalize import (
    bind_control_candidates,
    control_binding_pairs,
    judgment_original_labels,
    normalize_choice,
    normalize_noul,
    normalize_question,
    normalize_score,
)
from typevet.domain.judgment_questions import Choice, Noul, Score


def _single_char_tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


@pytest.mark.unit
def test_normalize_noul_bool_decision_and_labels() -> None:
    question = Noul(
        instructions="Is this about billing?",
        criteria={"true": "Yes", "false": "No"},
    )
    decision = normalize_noul(question, field_name="billing")
    assert decision.syntax == "Bool"
    assert decision.choices == (False, True)
    assert decision.question == "Is this about billing?"
    assert judgment_original_labels(question) == ("false", "true")


@pytest.mark.unit
def test_normalize_choice_preserves_criteria_order() -> None:
    question = Choice(
        criteria={"z": "Last", "a": "First", "m": "Mid"},
        instructions="Pick one:",
    )
    decision = normalize_choice(question, field_name="route")
    assert decision.syntax == "Choice"
    assert decision.choices == ("z", "a", "m")
    assert decision.question == "Pick one:"
    assert judgment_original_labels(question) == ("z", "a", "m")


@pytest.mark.unit
def test_normalize_score_uses_integer_level_choices() -> None:
    question = Score(
        criteria=["Poor", "Fair", "Good"],
        instructions="Rate clarity:",
    )
    decision = normalize_score(question, field_name="quality")
    assert decision.syntax == "Choice"
    assert decision.choices == (0, 1, 2)
    assert decision.question == "Rate clarity:"
    assert judgment_original_labels(question) == ("0", "1", "2")


@pytest.mark.unit
def test_normalize_question_dispatches() -> None:
    noul = Noul(instructions="Yes?")
    assert normalize_question(noul, field_name="q").syntax == "Bool"
    choice = Choice(criteria={"x": None}, instructions="Pick:")
    assert normalize_question(choice, field_name="q").syntax == "Choice"


@pytest.mark.unit
def test_empty_choice_criteria_fails() -> None:
    with pytest.raises(JudgmentValidationError, match="empty"):
        normalize_choice(Choice(criteria={}, instructions="Pick:"), field_name="c")


@pytest.mark.unit
def test_score_needs_at_least_two_levels() -> None:
    with pytest.raises(JudgmentValidationError, match="criteria"):
        normalize_score(
            Score(criteria=["Only"], instructions="Rate:"),
            field_name="s",
        )


@pytest.mark.unit
def test_invalid_instructions_type_fails() -> None:
    with pytest.raises(JudgmentValidationError, match="instructions"):
        normalize_noul(Noul(instructions={"bad": 1}), field_name="n")


@pytest.mark.unit
def test_bind_control_candidates_ordinal_controls() -> None:
    labels = ("billing", "technical")
    specs = bind_control_candidates(labels, _single_char_tokenize)
    assert len(specs) == 2
    assert specs[0].label == "billing"
    assert specs[0].token_ids == (ord("0"),)
    assert specs[1].label == "technical"
    assert specs[1].token_ids == (ord("1"),)


@pytest.mark.unit
def test_control_binding_pairs_ordinal_controls() -> None:
    pairs = control_binding_pairs(("billing", "technical"))
    assert pairs == (("0", "billing"), ("1", "technical"))


@pytest.mark.unit
def test_control_binding_pairs_digits_then_letters() -> None:
    labels = tuple(f"l{i}" for i in range(36))
    controls = [control for control, _ in control_binding_pairs(labels)]
    assert controls == [str(i) for i in range(10)] + [chr(65 + i) for i in range(26)]


@pytest.mark.unit
def test_control_binding_pairs_ten_labels_stay_digits() -> None:
    labels = tuple(f"l{i}" for i in range(10))
    assert control_binding_pairs(labels) == tuple((str(i), f"l{i}") for i in range(10))


@pytest.mark.unit
def test_bind_control_rejects_duplicate_labels() -> None:
    with pytest.raises(JudgmentValidationError, match="duplicate"):
        bind_control_candidates(("a", "a"), _single_char_tokenize)


@pytest.mark.unit
def test_bind_control_rejects_multi_token_control() -> None:
    def multi(_: str) -> tuple[int, ...]:
        return (1, 2)

    with pytest.raises(JudgmentValidationError, match="exactly one token"):
        bind_control_candidates(("x", "y"), multi)
