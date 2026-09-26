"""Contract tests: offline scoring fake + categorical decision executor."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.scoring_contract import ContractScoringFake
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.domain.scoring_stage import ScoreStage


@pytest.mark.contract
def test_executor_uses_pre_sampling_stage() -> None:
    decision = Decision("c", "Pick.", ("billing", "technical"), syntax="Choice")
    candidates = (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    execute_categorical_decision(
        decision,
        prefix="Answer:",
        candidates=candidates,
        port=fake,
        model="fake-scoring",
    )
    assert len(fake.calls) == 1
    assert fake.calls[0].stage is ScoreStage.PRE_SAMPLING
    assert fake.calls[0].prefix == "Answer:"
    assert fake.calls[0].candidates == candidates


@pytest.mark.contract
def test_executor_result_shape_matches_contract() -> None:
    decision = Decision("c", "Pick.", ("billing", "technical"), syntax="Choice")
    candidates = (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    result = execute_categorical_decision(
        decision,
        prefix="Answer:",
        candidates=candidates,
        port=fake,
        model="fake-scoring",
    )
    assert isinstance(result, CategoricalExecutionResult)
    assert result.model == "fake-scoring"
    assert result.value == "billing"
    assert len(result.probabilities) == 2
    assert len(result.logprobs) == 2
    assert result.probabilities[0][0] == "billing"
    assert result.probabilities[1][0] == "technical"
    for _, prob in result.probabilities:
        assert math.isfinite(prob)
        assert prob > 0.0
    total = sum(p for _, p in result.probabilities)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result.logprobs == (-0.5, -1.2)
