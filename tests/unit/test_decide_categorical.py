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
def test_decide_categorical_calls_scorer_and_full_distribution() -> None:
    fake = ContractScoringFake(
        logprobs={"billing": -0.5, "technical": -1.2},
    )
    result = decide_categorical(
        field=_SINGLE_ENUM_SCHEMA,
        prompt="Route this ticket.",
        prefix="Answer:",
        model="fake",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    assert len(fake.calls) == 1
    assert fake.calls[0].prefix == "Answer:"
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
        prompt="Route.",
        prefix="P:",
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
        ({"prompt": ""}, DecisionExecutionError),
        ({"model": ""}, DecisionExecutionError),
        ({"prefix": ""}, DecisionExecutionError),
    ],
)
def test_decide_categorical_invalid_before_io(
    kwargs: dict[str, Any], exc_type: type[BaseException]
) -> None:
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    field = kwargs.get("field", _SINGLE_ENUM_SCHEMA)
    prompt = kwargs.get("prompt", "Route.")
    prefix = kwargs.get("prefix", "Answer:")
    model = kwargs.get("model", "fake")
    with pytest.raises(exc_type):
        decide_categorical(
            field=field,
            prompt=prompt,
            prefix=prefix,
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
            prompt="Route.",
            prefix="Answer:",
            model="fake",
            scoring_port=fake,
            candidates=_specs("billing", "technical"),
            post_sampling=True,
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
        prompt="Route.",
        prefix="Answer:",
        model="fake",
        scoring_port=fake,
        candidates=_specs("billing", "technical"),
    )
    fake.close.assert_not_called()
