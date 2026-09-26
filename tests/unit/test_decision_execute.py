"""Unit tests for IO-free categorical Decision execution."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.scoring_contract import ContractScoringFake
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import execute_categorical_decision
from typevet.domain.decisions import MAX_ENUM_CHOICES, Decision
from typevet.domain.errors import DecisionExecutionError, ScoringValidationError


def _softmax_oracle(
    logprobs: tuple[float, ...], *, temperature: float = 1.0
) -> tuple[float, ...]:
    scaled = [lp / temperature for lp in logprobs]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    return tuple(e / total for e in exps)


def _choice_decision(*choices: str) -> Decision:
    return Decision("field", "Pick one.", tuple(choices), syntax="Choice")


def _specs(*labels: str) -> tuple[CandidateTokenSpec, ...]:
    return tuple(
        CandidateTokenSpec(label, (100 + i,)) for i, label in enumerate(labels)
    )


@pytest.mark.unit
def test_two_way_softmax_oracle_exact() -> None:
    logprobs = (-0.5, -1.2)
    expected_probs = _softmax_oracle(logprobs)
    decision = _choice_decision("billing", "technical")
    fake = ContractScoringFake(
        logprobs={"billing": logprobs[0], "technical": logprobs[1]}
    )
    result = execute_categorical_decision(
        decision,
        prefix="Answer:",
        candidates=_specs("billing", "technical"),
        port=fake,
        model="fake",
    )
    assert result.value == "billing"
    assert len(result.probabilities) == 2
    for (_, prob), exp in zip(result.probabilities, expected_probs, strict=True):
        assert math.isfinite(prob)
        assert prob == pytest.approx(exp, abs=1e-12)
    assert sum(p for _, p in result.probabilities) == pytest.approx(1.0, abs=1e-6)
    assert result.logprobs == logprobs


@pytest.mark.unit
def test_three_way_softmax_oracle_exact() -> None:
    logprobs = (-0.1, -0.2, -1.5)
    expected_probs = _softmax_oracle(logprobs)
    labels = ("a", "b", "c")
    decision = _choice_decision(*labels)
    fake = ContractScoringFake(logprobs=dict(zip(labels, logprobs, strict=True)))
    result = execute_categorical_decision(
        decision,
        prefix="P:",
        candidates=_specs(*labels),
        port=fake,
        model="m",
    )
    assert result.value == "a"
    for (_, prob), exp in zip(result.probabilities, expected_probs, strict=True):
        assert prob == pytest.approx(exp, abs=1e-12)


@pytest.mark.unit
def test_absent_candidate_from_fake_raises() -> None:
    decision = _choice_decision("billing", "technical")
    fake = ContractScoringFake(logprobs={"billing": -0.1})
    with pytest.raises(ScoringValidationError, match="missing scripted"):
        execute_categorical_decision(
            decision,
            prefix="Answer:",
            candidates=_specs("billing", "technical"),
            port=fake,
            model="fake",
        )


@pytest.mark.unit
def test_non_finite_logprob_raises() -> None:
    decision = _choice_decision("billing", "technical")
    fake = ContractScoringFake(
        logprobs={"billing": float("nan"), "technical": -1.0},
    )
    with pytest.raises(ScoringValidationError):
        execute_categorical_decision(
            decision,
            prefix="Answer:",
            candidates=_specs("billing", "technical"),
            port=fake,
            model="fake",
        )


@pytest.mark.unit
def test_label_misalignment_raises() -> None:
    decision = _choice_decision("billing", "technical")
    fake = ContractScoringFake(logprobs={"billing": -0.1, "wrong": -0.2})
    with pytest.raises(DecisionExecutionError, match="align"):
        execute_categorical_decision(
            decision,
            prefix="Answer:",
            candidates=_specs("billing", "wrong"),
            port=fake,
            model="fake",
        )


@pytest.mark.unit
def test_twenty_four_way_mapping_and_distribution() -> None:
    labels = tuple(f"c{i}" for i in range(24))
    logprobs = tuple(-float(i) for i in range(24))
    expected_probs = _softmax_oracle(logprobs)
    decision = Decision("x", "q", labels, syntax="Choice")
    scripted = dict(zip(labels, logprobs, strict=True))
    fake = ContractScoringFake(logprobs=scripted)
    result = execute_categorical_decision(
        decision,
        prefix="P:",
        candidates=_specs(*labels),
        port=fake,
        model="m",
    )
    assert result.value == "c0"
    assert len(result.probabilities) == 24
    assert len(result.logprobs) == 24
    for (val, prob), exp in zip(result.probabilities, expected_probs, strict=True):
        assert val in labels
        assert prob == pytest.approx(exp, abs=1e-9)
    assert sum(p for _, p in result.probabilities) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_bool_true_false_mapping() -> None:
    decision = Decision("flag", "On?", (True, False), syntax="Bool")
    fake = ContractScoringFake(logprobs={"True": -0.2, "False": -0.8})
    result = execute_categorical_decision(
        decision,
        prefix="Q:",
        candidates=_specs("True", "False"),
        port=fake,
        model="m",
    )
    assert result.value is True
    assert result.probabilities[0][0] is True
    assert result.probabilities[1][0] is False


@pytest.mark.unit
def test_tie_breaks_to_lowest_index() -> None:
    decision = _choice_decision("first", "second")
    fake = ContractScoringFake(logprobs={"first": -1.0, "second": -1.0})
    result = execute_categorical_decision(
        decision,
        prefix="P:",
        candidates=_specs("first", "second"),
        port=fake,
        model="m",
    )
    assert result.value == "first"
    assert result.probabilities[0][1] == result.probabilities[1][1]


@pytest.mark.unit
def test_nullable_rejected() -> None:
    decision = Decision(
        "field",
        "Pick one.",
        ("a", "b"),
        syntax="Choice",
        nullable=True,
    )
    fake = ContractScoringFake(logprobs={"a": -0.1, "b": -0.2})
    with pytest.raises(DecisionExecutionError, match="nullable"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=_specs("a", "b"),
            port=fake,
            model="m",
        )


@pytest.mark.unit
def test_wrong_syntax_rejected() -> None:
    decision = Decision("n", "q", (), syntax="Integer", numeric_type="integer")
    fake = ContractScoringFake(logprobs={"a": -0.1})
    with pytest.raises(DecisionExecutionError, match="syntax"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=(CandidateTokenSpec("a", (1,)), CandidateTokenSpec("b", (2,))),
            port=fake,
            model="m",
        )


@pytest.mark.unit
def test_multi_token_candidate_rejected() -> None:
    decision = _choice_decision("a", "b")
    fake = ContractScoringFake(logprobs={"a": -0.1, "b": -0.2})
    with pytest.raises(DecisionExecutionError, match="single-token"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=(
                CandidateTokenSpec("a", (1, 2)),
                CandidateTokenSpec("b", (3,)),
            ),
            port=fake,
            model="m",
        )


@pytest.mark.unit
def test_temperature_non_positive_rejected() -> None:
    decision = _choice_decision("a", "b")
    fake = ContractScoringFake(logprobs={"a": -0.1, "b": -0.2})
    with pytest.raises(DecisionExecutionError, match="temperature"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=_specs("a", "b"),
            port=fake,
            model="m",
            temperature=0.0,
        )


@pytest.mark.unit
def test_too_few_choices_rejected() -> None:
    decision = _choice_decision("only")
    fake = ContractScoringFake(logprobs={"only": -0.1})
    with pytest.raises(DecisionExecutionError):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=(CandidateTokenSpec("only", (1,)),),
            port=fake,
            model="m",
        )


@pytest.mark.unit
def test_candidate_count_mismatch_rejected() -> None:
    decision = _choice_decision("a", "b")
    fake = ContractScoringFake(logprobs={"a": -0.1, "b": -0.2, "c": -0.3})
    with pytest.raises(DecisionExecutionError, match="length"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=_specs("a", "b", "c"),
            port=fake,
            model="m",
        )


@pytest.mark.unit
def test_execute_categorical_permutations_one_control_still_scores() -> None:
    decision = Decision(
        "field",
        "Pick one.",
        ("billing", "technical"),
        syntax="Choice",
        permutations=1,
    )
    fake = ContractScoringFake(
        logprobs={"billing": -0.5, "technical": -1.2},
    )
    result = execute_categorical_decision(
        decision,
        prefix="Answer:",
        candidates=_specs("billing", "technical"),
        port=fake,
        model="fake",
    )
    assert len(fake.calls) == 1
    assert result.value == "billing"


@pytest.mark.unit
@pytest.mark.parametrize(
    "permutations",
    [2, "all"],
    ids=["permutations_2", "permutations_all"],
)
def test_execute_categorical_rejects_unsupported_permutations_before_io(
    permutations: int | str,
) -> None:
    decision = Decision(
        "field",
        "Pick one.",
        ("a", "b"),
        syntax="Choice",
        permutations=permutations,
    )
    fake = ContractScoringFake(logprobs={"a": -0.1, "b": -0.2})
    with pytest.raises(DecisionExecutionError, match="permutation"):
        execute_categorical_decision(
            decision,
            prefix="P:",
            candidates=_specs("a", "b"),
            port=fake,
            model="m",
        )
    assert fake.calls == []


@pytest.mark.unit
def test_max_enum_choices_boundary_ok() -> None:
    labels = tuple(f"v{i}" for i in range(MAX_ENUM_CHOICES))
    logprobs = tuple(-1.0 for _ in labels)
    decision = Decision("f", "q", labels, syntax="Choice")
    fake = ContractScoringFake(logprobs=dict(zip(labels, logprobs, strict=True)))
    result = execute_categorical_decision(
        decision,
        prefix="P:",
        candidates=_specs(*labels),
        port=fake,
        model="m",
    )
    assert len(result.probabilities) == MAX_ENUM_CHOICES
