"""Shared CandidateScoringPort contract fixtures.

Each fixture is **synthetic**: the author defines the request, scripted scores
or failures, and expected outcomes. ``ContractScoringFake`` aliases
[typevet.testing.ScriptedScoringFake][] for contract tests.
"""

from __future__ import annotations

import math
from typing import Any

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
)
from typevet.domain.errors import (
    ScoringError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.scoring_stage import ScoreStage
from typevet.testing import ScriptedScoringFake

ContractScoringFake = ScriptedScoringFake


class RogueScoringPort:
    """Offline port that returns author-built rows without request checks."""

    def __init__(
        self,
        *,
        rows: tuple[ScoredCandidate, ...],
        stage: ScoreStage = ScoreStage.PRE_SAMPLING,
    ) -> None:
        """Configure the rows and stage returned for every request.

        Args:
            rows: Scored rows returned as is, in this order.
            stage: Stage recorded on the returned result.
        """
        self._rows = rows
        self._stage = stage
        self.calls: list[CandidateScoringRequest] = []

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        self.calls.append(request)
        return CandidateScoringResult(
            model=request.model,
            stage=self._stage,
            candidates=self._rows,
        )


MISMATCH_CANDIDATES = (
    CandidateTokenSpec("billing", (101,)),
    CandidateTokenSpec("technical", (202,)),
)


def get_result_mismatch_fixtures() -> list[dict[str, Any]]:
    """Return results that disagree with ``MISMATCH_CANDIDATES``.

    Each fixture names the rows and stage a rogue port returns and a regex
    that the ``ScoringValidationError`` message matches.
    """
    billing = ScoredCandidate("billing", (101,), -3.0)
    technical = ScoredCandidate("technical", (202,), -0.1)
    return [
        {
            "name": "reversed_rows",
            "rows": (technical, billing),
            "match": "order",
        },
        {
            "name": "missing_row",
            "rows": (billing,),
            "match": "count",
        },
        {
            "name": "extra_row",
            "rows": (billing, technical, ScoredCandidate("other", (303,), -1.0)),
            "match": "count",
        },
        {
            "name": "duplicate_row",
            "rows": (billing, ScoredCandidate("billing", (101,), -0.1)),
            "match": "duplicate",
        },
        {
            "name": "unexpected_row",
            "rows": (billing, ScoredCandidate("other", (202,), -0.1)),
            "match": "unexpected",
        },
        {
            "name": "wrong_token_ids",
            "rows": (billing, ScoredCandidate("technical", (999,), -0.1)),
            "match": "token ids",
        },
        {
            "name": "wrong_stage",
            "rows": (billing, technical),
            "stage": ScoreStage.POST_SAMPLING,
            "match": "stage",
        },
        {
            "name": "non_finite_logprob",
            "rows": (billing, ScoredCandidate("technical", (202,), float("inf"))),
            "match": "non-finite",
        },
    ]


def rogue_port_for(fixture: dict[str, Any]) -> RogueScoringPort:
    """Build the rogue port for a result mismatch fixture."""
    return RogueScoringPort(
        rows=fixture["rows"],
        stage=fixture.get("stage", ScoreStage.PRE_SAMPLING),
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


OFF_OPTION_N_VOCAB = 4
"""Vocabulary size of the synthetic full distribution below."""


def get_off_option_fixture() -> dict[str, Any]:
    """Return one synthetic full-vocabulary distribution and its expected scores.

    The four token probabilities are 0.5, 0.2, 0.2 and 0.1. The candidates are
    ids 101 and 202, so the off-option mass is ``1 - (0.5 + 0.2) = 0.3``.
    ``top_logprobs`` is the llama.cpp ``/completion`` entry list for the whole
    vocabulary. ``fake`` scripts the same scores on ``ContractScoringFake``.
    """
    probs = {101: 0.5, 202: 0.2, 303: 0.2, 404: 0.1}
    logprobs = {token_id: math.log(p) for token_id, p in probs.items()}
    return {
        "name": "full_vocabulary_two_candidates",
        "request": _req(),
        "n_vocab": OFF_OPTION_N_VOCAB,
        "top_logprobs": [
            {"id": token_id, "logprob": value} for token_id, value in logprobs.items()
        ],
        "fake": {
            "logprobs": {"billing": logprobs[101], "technical": logprobs[202]},
            "off_option_mass": 0.3,
        },
        "expect": {
            "logprobs": [logprobs[101], logprobs[202]],
            "off_option_mass": 0.3,
        },
    }
