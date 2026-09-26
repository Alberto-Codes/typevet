"""Unit tests for validating a scoring result against its exact request."""

from __future__ import annotations

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
from typevet.domain.candidate_scoring_validate import validate_result_against_request
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
