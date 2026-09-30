"""Fail-closed validation for candidate scoring coverage.

Examples:
    ```python
    from typevet.domain.candidate_scoring_validate import (
        build_and_validate_result,
        validate_result_against_request,
    )
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
    validate_result_against_request(request, result)
    ```

``build_and_validate_result`` builds a result from a label mapping.
``validate_result_against_request`` checks a finished port result against the
exact request before a consumer normalizes its logprobs. Both reject a
non-finite logprob and a positive logprob above ``1e-6``. When a result
carries ``off_option_mass``, both also reject a value that is not finite, is
outside ``[0, 1]``, or makes the candidate mass plus the off-option mass go
above 1 by more than ``1e-2``.

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

_POSITIVE_LOGPROB_TOLERANCE = 1e-6
"""Largest logprob accepted above zero.

Log-softmax output is at most 0. A float32 rounding error stays below
``1.2e-7``. A probability sent in place of a logprob is far above this value.
"""

_TOTAL_MASS_TOLERANCE = 1e-2
"""Largest amount by which candidate plus off-option mass may exceed 1.

A float32 softmax makes a probability sum drift from 1. One local Gemma 4 call
summed to 1.0006 over the full vocabulary (#297 review). The llama.cpp adapter
accepts the same margin, so a result it builds never fails this check. A wrong
off-option mass is far above it.
"""


def _check_off_option_mass(logprobs: list[float], value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value):
        msg = f"non-finite off_option_mass: {value!r}"
        raise ScoringValidationError(msg)
    if not 0.0 <= value <= 1.0:
        msg = f"off_option_mass {value!r} is outside [0, 1]"
        raise ScoringValidationError(msg)
    total = math.fsum(math.exp(logprob) for logprob in logprobs) + value
    if total > 1.0 + _TOTAL_MASS_TOLERANCE:
        msg = (
            f"candidate mass plus off_option_mass is {total!r}, above 1 by more "
            f"than {_TOTAL_MASS_TOLERANCE!r}"
        )
        raise ScoringValidationError(msg)


def _check_logprob(label: str, value: float) -> None:
    if not math.isfinite(value):
        msg = f"non-finite logprob for candidate {label!r}: {value!r}"
        raise ScoringValidationError(msg)
    if value > _POSITIVE_LOGPROB_TOLERANCE:
        msg = (
            f"positive logprob for candidate {label!r}: {value!r} is above "
            f"the tolerance {_POSITIVE_LOGPROB_TOLERANCE!r}"
        )
        raise ScoringValidationError(msg)


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
        _check_logprob(label, value)


def validate_result_against_request(
    request: CandidateScoringRequest,
    result: CandidateScoringResult,
) -> None:
    """Reject a result that does not score exactly the requested candidates.

    Rows must match ``request.candidates`` one to one, in request order, with
    the same label and token ids, at the requested stage, with finite logprobs
    that are not above ``1e-6``. Accepted logprobs are not changed.

    ``result.model`` is not compared with ``request.model``. A backend may
    report a resolved id for a requested alias, so the result model is
    provenance only. No alias table exists and no alias matching is done.

    Args:
        request: The exact scoring ask sent to the port.
        result: The port result for ``request``.

    Raises:
        ScoringValidationError: On a stage, row count, duplicate label,
            unexpected label, label order, token-id, or non-finite logprob
            mismatch, on a positive logprob above ``1e-6``, or on an
            off-option mass that is not finite, is outside ``[0, 1]``, or
            makes the total mass go above 1.

    Examples:
        ```python
        validate_result_against_request(request, port.score_candidates(request))
        ```
    """
    if result.stage is not request.stage:
        msg = (
            f"result score stage {result.stage.value!r} does not match "
            f"requested stage {request.stage.value!r}"
        )
        raise ScoringValidationError(msg)
    expected = [spec.label for spec in request.candidates]
    got = [row.label for row in result.candidates]
    if len(got) != len(expected):
        msg = (
            f"result candidate count {len(got)} does not match request count "
            f"{len(expected)}: got {got!r}, expected {expected!r}"
        )
        raise ScoringValidationError(msg)
    duplicates = sorted({label for label in got if got.count(label) > 1})
    if duplicates:
        msg = f"duplicate score labels in result: {duplicates!r}"
        raise ScoringValidationError(msg)
    unexpected = [label for label in got if label not in expected]
    if unexpected:
        missing = [label for label in expected if label not in got]
        msg = (
            f"unexpected score labels not in request: {unexpected!r}; "
            f"missing: {missing!r}"
        )
        raise ScoringValidationError(msg)
    if got != expected:
        msg = f"result label order {got!r} does not match request order {expected!r}"
        raise ScoringValidationError(msg)
    for spec, row in zip(request.candidates, result.candidates, strict=True):
        if row.token_ids != spec.token_ids:
            msg = (
                f"result token ids {row.token_ids!r} for candidate {spec.label!r} "
                f"do not match requested token ids {spec.token_ids!r}"
            )
            raise ScoringValidationError(msg)
        _check_logprob(spec.label, row.logprob)
    _check_off_option_mass(
        [row.logprob for row in result.candidates], result.off_option_mass
    )


def build_and_validate_result(
    request: CandidateScoringRequest,
    *,
    raw_logprobs: Mapping[str, float],
    model: str,
    stage: ScoreStage | None = None,
    usage: TokenUsage | None = None,
    termination: ScoringTermination | None = None,
    off_option_mass: float | None = None,
) -> CandidateScoringResult:
    """Build a result with request order and fail-closed coverage checks.

    Args:
        request: Original scoring ask.
        raw_logprobs: Mapping from candidate label to raw logprob.
        model: Model id recorded on the result.
        stage: Stage recorded on the result; defaults to ``request.stage``.
        usage: Optional token usage metadata.
        termination: Optional termination metadata.
        off_option_mass: Probability mass outside the candidate tokens, or
            ``None`` when the backend cannot report it.

    Returns:
        Validated ``CandidateScoringResult``.

    Raises:
        ScoringValidationError: Missing, duplicate, or unexpected labels,
            non-finite logprobs, a positive logprob above ``1e-6``, or an
            off-option mass that is not finite, is outside ``[0, 1]``, or
            makes the total mass go above 1.
    """
    _validate_raw_logprobs(request, raw_logprobs)
    _check_off_option_mass(list(raw_logprobs.values()), off_option_mass)
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
        off_option_mass=off_option_mass,
    )
