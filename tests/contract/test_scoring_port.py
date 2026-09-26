"""Contract tests for CandidateScoringPort offline fakes."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.scoring_contract import exc_type_from_name, fake_for, get_fixtures
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import ScoringValidationError


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda f: f["name"])
def test_scoring_fake_honors_fixture(fixture: dict) -> None:
    fake = fake_for(fixture)
    expect = fixture["expect"]
    request = fixture["request"]
    if expect["kind"] == "success":
        result = fake.score_candidates(request)
        assert result.model == fixture["request"].model
        assert len(result.candidates) == len(expect["logprobs"])
        for scored, logprob in zip(result.candidates, expect["logprobs"], strict=True):
            assert math.isfinite(scored.logprob)
            assert scored.logprob == logprob
        for scored, spec in zip(result.candidates, request.candidates, strict=True):
            assert scored.label == spec.label
            assert scored.token_ids == spec.token_ids
        assert len(fake.calls) == 1
        return

    exc_type = exc_type_from_name(expect["exc_type"])
    with pytest.raises(exc_type):
        fake.score_candidates(request)
    assert len(fake.calls) == 1


@pytest.mark.contract
def test_scoring_request_rejects_duplicate_labels() -> None:
    with pytest.raises(ScoringValidationError, match="duplicate candidate labels"):
        CandidateScoringRequest(
            model="m",
            prefix="P:",
            candidates=(
                CandidateTokenSpec("a", (1,)),
                CandidateTokenSpec("a", (2,)),
            ),
        )


@pytest.mark.contract
def test_scoring_validate_rejects_unexpected_label() -> None:
    request = CandidateScoringRequest(
        model="m",
        prefix="P:",
        candidates=(CandidateTokenSpec("a", (1,)),),
    )
    with pytest.raises(ScoringValidationError, match="unexpected score labels"):
        build_and_validate_result(
            request,
            raw_logprobs={"a": -1.0, "extra": -2.0},
            model="m",
        )
