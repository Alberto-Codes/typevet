"""Fail-closed validation for candidate scoring coverage.

Examples:
    ```python
    from typevet.domain.candidate_scoring_validate import build_and_validate_result
    from typevet.domain.candidate_scoring_request import (
        CandidateScoringRequest,
        CandidateTokenSpec,
    )

    request = CandidateScoringRequest(
        model="m",
        prefix="P:",
        candidates=(CandidateTokenSpec("a", (1,)),),
    )
    result = build_and_validate_result(request, raw_logprobs={"a": -0.5}, model="m")
    assert result.candidates[0].logprob == -0.5
    ```

See Also:
    - [typevet.domain.errors][]: ScoringValidationError
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
    ScoringTermination,
)
from typevet.domain.errors import ScoringValidationError
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.scoring_stage import ScoreStage


def _validate_raw_logprobs(
    request: CandidateScoringRequest,
    raw_logprobs: Mapping[str, float],
) -> None:
    expected = [spec.label for spec in request.candidates]
    seen = list(raw_logprobs.keys())
    if len(seen) != len(set(seen)):
        msg = f"duplicate score labels in response: {seen!r}"
        raise ScoringValidationError(msg)
    missing = [label for label in expected if label not in raw_logprobs]
    if missing:
        msg = f"missing scores for requested candidates: {missing!r}"
        raise ScoringValidationError(msg)
    unexpected = [label for label in seen if label not in expected]
    if unexpected:
        msg = f"unexpected score labels not in request: {unexpected!r}"
        raise ScoringValidationError(msg)
    for label, value in raw_logprobs.items():
        if not math.isfinite(value):
            msg = f"non-finite logprob for candidate {label!r}: {value!r}"
            raise ScoringValidationError(msg)


def build_and_validate_result(
    request: CandidateScoringRequest,
    *,
    raw_logprobs: Mapping[str, float],
    model: str,
    stage: ScoreStage | None = None,
    usage: TokenUsage | None = None,
    termination: ScoringTermination | None = None,
) -> CandidateScoringResult:
    """Build a result with request order and fail-closed coverage checks.

    Args:
        request: Original scoring ask.
        raw_logprobs: Mapping from candidate label to raw logprob.
        model: Model id recorded on the result.
        stage: Stage recorded on the result; defaults to ``request.stage``.
        usage: Optional token usage metadata.
        termination: Optional termination metadata.

    Returns:
        Validated ``CandidateScoringResult``.

    Raises:
        ScoringValidationError: Missing, duplicate, or unexpected labels, or
            non-finite logprobs.
    """
    _validate_raw_logprobs(request, raw_logprobs)
    scored = tuple(
        ScoredCandidate(
            label=spec.label,
            token_ids=spec.token_ids,
            logprob=raw_logprobs[spec.label],
        )
        for spec in request.candidates
    )
    return CandidateScoringResult(
        model=model,
        stage=stage if stage is not None else request.stage,
        candidates=scored,
        usage=usage or TokenUsage(),
        termination=termination,
    )
