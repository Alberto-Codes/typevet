"""Public offline fakes built from the domain alone.

Examples:
    ```python
    from typevet.testing.fakes import ScriptedScoringFake, StaticGenerationFake
    from typevet.domain.models import GenerationRequest

    gen = StaticGenerationFake({"ok": True})
    result = gen.generate(
        GenerationRequest(
            prompt="x",
            schema={"type": "object", "additionalProperties": False},
            model="fake",
        )
    )
    assert result.value["ok"] is True
    ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
    ```

See Also:
    - [typevet.adapters.outbound.fake][]: Validating fake used in contract tests
    - [typevet.domain.models][]: Request and result types
    - [typevet.ports.scoring][]: CandidateScoringPort protocol
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import (
    ScoringError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.models import GenerationRequest, GenerationResult
from typevet.domain.scoring_stage import ScoreStage


class StaticGenerationFake:
    """Return a fixed mapping without importing adapters.

    Callers that need schema validation should use
    ``typevet.adapters.outbound.FakeGenerationAdapter`` in contract tests.
    This fake is for inbound unit tests that only need a port double.

    Attributes:
        _value (dict[str, Any]): Stored mapping returned from generate.

    Examples:
        ```python
        from typevet.testing.fakes import StaticGenerationFake

        StaticGenerationFake({"a": 1})
        ```
    """

    def __init__(self, value: Mapping[str, Any]) -> None:
        """Store the fixed return value.

        Args:
            value: Mapping returned from every ``generate`` call.
        """
        self._value = dict(value)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return the fixed value wrapped as a result.

        Args:
            request: Incoming request (model is copied onto the result).

        Returns:
            GenerationResult with the stored mapping.
        """
        return GenerationResult(value=self._value, model=request.model)


class ScriptedScoringFake:
    """Offline ``CandidateScoringPort`` with author-scripted logprobs.

    Use from installed wheels for tutorials and unit doubles. Contract tests
    that need richer rogue ports keep fixtures under ``tests/fixtures``.

    Attributes:
        calls (list[CandidateScoringRequest]): Recorded ``score_candidates`` inputs.

    Examples:
        ```python
        from typevet.testing import ScriptedScoringFake

        ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
        ```
    """

    def __init__(
        self,
        *,
        logprobs: Mapping[str, float] | None = None,
        fail: ScoringError | None = None,
        supported_stages: frozenset[ScoreStage] | None = None,
    ) -> None:
        """Configure scripted logprobs, optional failure, and supported stages.

        Args:
            logprobs: Per-label logprobs returned for requested candidates.
            fail: When set, ``score_candidates`` raises instead of succeeding.
            supported_stages: Stages this fake accepts; defaults to pre-sampling.
        """
        self._scripted = dict(logprobs or {})
        self._fail = fail
        self._supported = supported_stages or frozenset({ScoreStage.PRE_SAMPLING})
        self.calls: list[CandidateScoringRequest] = []

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Return scripted logprobs or raise configured errors.

        Args:
            request: Candidate labels and prefix the adapter bound.

        Returns:
            Validated scoring result for the request.

        Raises:
            ScoringError: When ``fail`` was set at construction.
            ScoringUnsupportedCapabilityError: When ``request.stage`` is unsupported.
            ScoringValidationError: When a candidate label lacks a scripted logprob.
        """
        self.calls.append(request)
        if self._fail is not None:
            raise self._fail
        if request.stage not in self._supported:
            msg = f"unsupported score stage {request.stage!r}"
            raise ScoringUnsupportedCapabilityError(msg)
        raw: dict[str, float] = {}
        for spec in request.candidates:
            if spec.label not in self._scripted:
                msg = f"missing scripted logprob for candidate {spec.label!r}"
                raise ScoringValidationError(msg)
            raw[spec.label] = self._scripted[spec.label]
        return build_and_validate_result(
            request,
            raw_logprobs=raw,
            model=request.model,
            usage=TokenUsage(),
        )
