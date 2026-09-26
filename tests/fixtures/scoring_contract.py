"""Shared CandidateScoringPort contract fixtures.

Each fixture is **synthetic**: the author defines the request, scripted scores
or failures, and expected outcomes. Contract tests exercise offline fakes that
implement [typevet.ports.scoring.CandidateScoringPort][].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import (
    ScoringError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.scoring_stage import ScoreStage


class ContractScoringFake:
    """Offline fake that implements ``CandidateScoringPort`` for contract tests."""

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


def _req(
    *,
    model: str = "fake-scoring",
    prefix: str = "Answer:",
    candidates: tuple[CandidateTokenSpec, ...] | None = None,
    stage: ScoreStage = ScoreStage.PRE_SAMPLING,
) -> CandidateScoringRequest:
    specs = candidates or (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    return CandidateScoringRequest(
        model=model,
        prefix=prefix,
        candidates=specs,
        stage=stage,
    )


def get_fixtures() -> list[dict[str, Any]]:
    """Return shared scoring contract fixtures."""
    base = _req()
    return [
        {
            "name": "two_candidates_scripted",
            "label": "synthetic",
            "request": base,
            "fake": {"logprobs": {"billing": -0.5, "technical": -1.2}},
            "expect": {"kind": "success", "logprobs": [-0.5, -1.2]},
        },
        {
            "name": "missing_requested_candidate",
            "label": "synthetic",
            "request": base,
            "fake": {"logprobs": {"billing": -0.1}},
            "expect": {"kind": "error", "exc_type": "ScoringValidationError"},
        },
        {
            "name": "non_finite_logprob",
            "label": "synthetic",
            "request": base,
            "fake": {"logprobs": {"billing": float("nan"), "technical": -1.0}},
            "expect": {"kind": "error", "exc_type": "ScoringValidationError"},
        },
        {
            "name": "unsupported_stage",
            "label": "synthetic",
            "request": _req(stage=ScoreStage.POST_SAMPLING),
            "fake": {},
            "expect": {
                "kind": "error",
                "exc_type": "ScoringUnsupportedCapabilityError",
            },
        },
        {
            "name": "configured_failure",
            "label": "synthetic",
            "request": base,
            "fake": {"fail": ScoringError("offline failure")},
            "expect": {"kind": "error", "exc_type": "ScoringError"},
        },
    ]


def fake_for(fixture: dict[str, Any]) -> ContractScoringFake:
    """Build the contract fake for a fixture."""
    fake_cfg = fixture.get("fake", {})
    return ContractScoringFake(
        logprobs=fake_cfg.get("logprobs"),
        fail=fake_cfg.get("fail"),
        supported_stages=fake_cfg.get("supported_stages"),
    )


def exc_type_from_name(name: str) -> type[ScoringError]:
    mapping: dict[str, type[ScoringError]] = {
        "ScoringError": ScoringError,
        "ScoringValidationError": ScoringValidationError,
        "ScoringUnsupportedCapabilityError": ScoringUnsupportedCapabilityError,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        msg = f"unknown exc_type: {name}"
        raise ValueError(msg) from exc
