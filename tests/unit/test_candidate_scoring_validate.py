"""Unit tests for validating a scoring result against its exact request."""

from __future__ import annotations

import math
from typing import Any

import pytest

from tests.fixtures.scoring_contract import (
    MISMATCH_CANDIDATES,
    get_result_mismatch_fixtures,
)
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
)
from typevet.domain.candidate_scoring_validate import (
    build_and_validate_result,
    validate_result_against_request,
)
from typevet.domain.errors import ScoringValidationError
from typevet.domain.scoring_stage import ScoreStage


def _request() -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model="requested-alias",
        prefix="Answer:",
        candidates=MISMATCH_CANDIDATES,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "fixture", get_result_mismatch_fixtures(), ids=lambda f: f["name"]
)
def test_mismatched_result_rejected(fixture: dict[str, Any]) -> None:
    result = CandidateScoringResult(
        model="requested-alias",
        stage=fixture.get("stage", ScoreStage.PRE_SAMPLING),
        candidates=fixture["rows"],
    )
    with pytest.raises(ScoringValidationError, match=fixture["match"]):
        validate_result_against_request(_request(), result)


@pytest.mark.unit
def test_matching_result_accepted_with_different_model_id() -> None:
    result = CandidateScoringResult(
        model="resolved-backend-id",
        stage=ScoreStage.PRE_SAMPLING,
        candidates=(
            ScoredCandidate("billing", (101,), -0.5),
            ScoredCandidate("technical", (202,), -1.2),
        ),
    )
    validate_result_against_request(_request(), result)


def _result_with_logprob(logprob: float) -> CandidateScoringResult:
    return CandidateScoringResult(
        model="requested-alias",
        stage=ScoreStage.PRE_SAMPLING,
        candidates=(
            ScoredCandidate("billing", (101,), logprob),
            ScoredCandidate("technical", (202,), -1.2),
        ),
    )


@pytest.mark.unit
@pytest.mark.parametrize("logprob", [1.0, 2e-6])
def test_result_with_positive_logprob_rejected(logprob: float) -> None:
    with pytest.raises(ScoringValidationError, match="positive logprob"):
        validate_result_against_request(_request(), _result_with_logprob(logprob))


@pytest.mark.unit
@pytest.mark.parametrize("logprob", [0.0, -0.0, 1e-6])
def test_result_with_logprob_within_tolerance_accepted(logprob: float) -> None:
    result = _result_with_logprob(logprob)
    validate_result_against_request(_request(), result)
    assert result.candidates[0].logprob == logprob


@pytest.mark.unit
@pytest.mark.parametrize("logprob", [1.0, 2e-6])
def test_build_with_positive_logprob_rejected(logprob: float) -> None:
    with pytest.raises(ScoringValidationError, match="positive logprob"):
        build_and_validate_result(
            _request(),
            raw_logprobs={"billing": logprob, "technical": -1.2},
            model="m",
        )


@pytest.mark.unit
@pytest.mark.parametrize("logprob", [0.0, -0.0, 1e-6])
def test_build_with_logprob_within_tolerance_keeps_value(logprob: float) -> None:
    result = build_and_validate_result(
        _request(),
        raw_logprobs={"billing": logprob, "technical": -1.2},
        model="m",
    )
    kept = result.candidates[0].logprob
    assert kept == logprob
    assert math.copysign(1.0, kept) == math.copysign(1.0, logprob)
