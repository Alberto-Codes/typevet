"""Contract tests: public decide_categorical + offline scoring fake."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.scoring_contract import ContractScoringFake
from typevet import decide_categorical
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import CategoricalExecutionResult
from typevet.domain.scoring_stage import ScoreStage


@pytest.mark.contract
def test_decide_categorical_public_entry_matches_executor_semantics() -> None:
    schema = {
        "type": "object",
        "properties": {
            "c": {
                "type": "string",
                "enum": ["billing", "technical"],
            }
        },
        "required": ["c"],
        "additionalProperties": False,
    }
    candidates = (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    fake = ContractScoringFake(logprobs={"billing": -0.5, "technical": -1.2})
    result = decide_categorical(
        field=schema,
        context="Classify.",
        prefix="Answer:",
        inject_prefix=True,
        model="fake-scoring",
        scoring_port=fake,
        candidates=candidates,
    )
    assert isinstance(result, CategoricalExecutionResult)
    assert result.model == "fake-scoring"
    assert result.value == "billing"
    assert len(result.probabilities) == 2
    assert len(result.logprobs) == 2
    assert fake.calls[0].stage is ScoreStage.PRE_SAMPLING
    assert fake.calls[0].candidates == candidates
    total = sum(p for _, p in result.probabilities)
    assert total == pytest.approx(1.0, abs=1e-6)
    for _, prob in result.probabilities:
        assert math.isfinite(prob)
        assert prob > 0.0
