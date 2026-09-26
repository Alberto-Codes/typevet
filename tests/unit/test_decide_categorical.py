"""Unit tests for public ``decide_categorical`` (#124)."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.scoring_contract import ContractScoringFake
from typevet import decide_categorical
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decisions import Decision, SchemaError
from typevet.domain.errors import DecisionExecutionError
from typevet.gemma_served_template import CHATML_ASSISTANT_HEADER

_SINGLE_ENUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["billing", "technical"],
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


def _specs(*labels: str) -> tuple[CandidateTokenSpec, ...]:
    return tuple(
        CandidateTokenSpec(label, (100 + i,)) for i, label in enumerate(labels)
    )


@pytest.mark.unit
def test_decide_categorical_native_context_reaches_scorer_prefix() -> None:
    decision = Decision(
        "label",
        "Pick the department.",
        ("billing", "technical"),
        syntax="Choice",
    )
    fake_a = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    fake_b = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    decide_categorical(
        field=decision,
        context="First ticket context.",
        model="fake",
        scoring_port=fake_a,
        candidates=_specs("billing", "technical"),
    )
    decide_categorical(
        field=decision,
        context="Second ticket context.",
        model="fake",
        scoring_port=fake_b,
        candidates=_specs("billing", "technical"),
    )
    assert fake_a.calls[0].prefix != fake_b.calls[0].prefix
    assert "First ticket context." in fake_a.calls[0].prefix
    assert "Second ticket context." in fake_b.calls[0].prefix
    assert "Pick the department." in fake_a.calls[0].prefix
    assert fake_a.calls[0].prefix.endswith(CHATML_ASSISTANT_HEADER)


@pytest.mark.unit
def test_decide_categorical_native_question_change_reaches_scorer() -> None:
    fake_a = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    fake_b = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    decide_categorical(
        field=Decision(
            "label",
            "Question A.",
            ("billing", "technical"),
            syntax="Choice",
        ),
        context="Same context.",
        model="fake",
        scoring_port=fake_a,
        candidates=_specs("billing", "technical"),
    )
    decide_categorical(
        field=Decision(
            "label",
            "Question B.",
            ("billing", "technical"),
            syntax="Choice",
        ),
        context="Same context.",
        model="fake",
        scoring_port=fake_b,
        candidates=_specs("billing", "technical"),
    )
    assert fake_a.calls[0].prefix != fake_b.calls[0].prefix
    assert "Question A." in fake_a.calls[0].prefix
    assert "Question B." in fake_b.calls[0].prefix


@pytest.mark.unit
def test_decide_categorical_inject_prefix_isolates_context() -> None:
    fixed = "Injected scoring prefix:"
    fake_a = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    fake_b = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    decide_categorical(
        field=_SINGLE_ENUM_SCHEMA,
        context="Context one.",
        prefix=fixed,
        inject_prefix=True,
        model="fake",
        scoring_port=fake_a,
        candidates=_specs("billing", "technical"),
    )
    decide_categorical(
        field=_SINGLE_ENUM_SCHEMA,
        context="Context two.",
        prefix=fixed,
        inject_prefix=True,
        model="fake",
        scoring_port=fake_b,
        candidates=_specs("billing", "technical"),
    )
    assert fake_a.calls[0].prefix == fixed
    assert fake_b.calls[0].prefix == fixed
    assert len(fake_a.calls) == 1
    assert len(fake_b.calls) == 1


@pytest.mark.unit
def test_decide_categorical_calls_scorer_and_full_distribution() -> None:
    fake = ContractScoringFake(
        logprobs={"billing": -0.5, "technical": -1.2},
    )
    result = decide_categorical(
        field=_SINGLE_ENUM_SCHEMA,
        context="Route this ticket.",
        model="fake",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    assert len(fake.calls) == 1
    assert "Route this ticket." in fake.calls[0].prefix
    assert len(result.probabilities) == 2
    assert result.value == "billing"
    assert sum(p for _, p in result.probabilities) == pytest.approx(1.0, abs=1e-6)
    for _, prob in result.probabilities:
        assert math.isfinite(prob)


@pytest.mark.unit
def test_decide_categorical_accepts_compiled_decision() -> None:
    decision = Decision("label", "Pick.", ("billing", "technical"), syntax="Choice")
    fake = ContractScoringFake(
        logprobs={"billing": -0.1, "technical": -0.2},
    )
    result = decide_categorical(
        field=decision,
        context="Route.",
        model="m",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    assert result.value == "billing"
    assert len(fake.calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "exc_type"),
    [
        (
            {
                "field": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "boolean"},
                        "b": {"type": "boolean"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
            DecisionExecutionError,
        ),
        (
            {
                "field": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                    },
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
            DecisionExecutionError,
        ),
        (
            {
                "field": {
                    "type": "object",
                    "properties": {
                        "label": {"type": ["string", "null"], "enum": ["a", "b"]},
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
            DecisionExecutionError,
        ),
        ({"field": {"type": "array"}}, SchemaError),
        ({"context": ""}, DecisionExecutionError),
        ({"model": ""}, DecisionExecutionError),
    ],
)
def test_decide_categorical_invalid_before_io(
    kwargs: dict[str, Any], exc_type: type[BaseException]
) -> None:
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    field = kwargs.get("field", _SINGLE_ENUM_SCHEMA)
    context = kwargs.get("context", "Route.")
    model = kwargs.get("model", "fake")
    with pytest.raises(exc_type):
        decide_categorical(
            field=field,
            context=context,
            model=model,
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
        )
    assert fake.calls == []


@pytest.mark.unit
def test_decide_categorical_rejects_unsupported_kwargs() -> None:
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    with pytest.raises(DecisionExecutionError, match="unsupported"):
        decide_categorical(
            field=_SINGLE_ENUM_SCHEMA,
            context="Route.",
            model="fake",
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
            post_sampling=True,
        )
    assert fake.calls == []


@pytest.mark.unit
def test_decide_categorical_permutations_one_control_still_scores() -> None:
    schema = {
        **_SINGLE_ENUM_SCHEMA,
        "properties": {
            "label": {
                **_SINGLE_ENUM_SCHEMA["properties"]["label"],
                "permutations": 1,
            }
        },
    }
    fake = ContractScoringFake(
        logprobs={"billing": -0.5, "technical": -1.2},
    )
    result = decide_categorical(
        field=schema,
        context="Route.",
        model="fake",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    assert len(fake.calls) == 1
    assert result.value == "billing"


@pytest.mark.unit
def test_decide_categorical_inject_prefix_requires_nonempty_prefix() -> None:
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    with pytest.raises(DecisionExecutionError, match="prefix"):
        decide_categorical(
            field=_SINGLE_ENUM_SCHEMA,
            context="Route.",
            prefix="",
            inject_prefix=True,
            model="fake",
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
        )
    assert fake.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "field",
    [
        {
            **_SINGLE_ENUM_SCHEMA,
            "properties": {
                "label": {
                    **_SINGLE_ENUM_SCHEMA["properties"]["label"],
                    "permutations": 2,
                }
            },
        },
        Decision(
            "label",
            "Pick.",
            ("billing", "technical"),
            syntax="Choice",
            permutations=2,
        ),
    ],
    ids=["schema_permutations_2", "decision_permutations_2"],
)
def test_decide_categorical_rejects_permutations_two_before_io(
    field: Decision | dict[str, Any],
) -> None:
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    with pytest.raises(DecisionExecutionError, match="permutation"):
        decide_categorical(
            field=field,
            context="Route.",
            model="fake",
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
        )
        assert fake.calls == []


@pytest.mark.unit
def test_decide_categorical_rejects_permutations_all_before_io() -> None:
    schema = {
        **_SINGLE_ENUM_SCHEMA,
        "properties": {
            "label": {
                **_SINGLE_ENUM_SCHEMA["properties"]["label"],
                "permutations": "all",
            }
        },
    }
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    with pytest.raises(DecisionExecutionError, match="permutation"):
        decide_categorical(
            field=schema,
            context="Route.",
            model="fake",
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
        )
        assert fake.calls == []


@pytest.mark.unit
def test_decide_categorical_does_not_close_injected_port() -> None:
    class _CloseTrackingFake(ContractScoringFake):
        def __init__(self) -> None:
            super().__init__(
                logprobs={"billing": -0.5, "technical": -1.2},
            )
            self.close = MagicMock()

    fake = _CloseTrackingFake()
    decide_categorical(
        field=_SINGLE_ENUM_SCHEMA,
        context="Route.",
        model="fake",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    fake.close.assert_not_called()
