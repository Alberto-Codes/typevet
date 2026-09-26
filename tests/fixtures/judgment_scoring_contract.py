"""Shared fixtures for ScoringJudgmentAdapter contract tests."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from tests.fixtures.scoring_contract import ContractScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.errors import JudgmentValidationError, ScoringValidationError
from typevet.domain.judgment_questions import Choice, Noul, Score


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


class SequentialScoringFake(ContractScoringFake):
    """Scoring fake that returns a different logprob map per call."""

    def __init__(self, logprobs_sequence: list[dict[str, float]]) -> None:
        """Store one logprob map per ``score_candidates`` call."""
        super().__init__()
        self._sequence = logprobs_sequence
        self._next = 0

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        if self._next >= len(self._sequence):
            msg = "sequential scoring fake exhausted scripted calls"
            raise ValueError(msg)
        self._scripted = dict(self._sequence[self._next])
        self._next += 1
        return super().score_candidates(request)


def adapter_for(
    *,
    logprobs_by_call: list[dict[str, float]] | None = None,
    tokenize: Callable[[str], tuple[int, ...]] = _tokenize,
    served_template: ServedTemplateClass | None = None,
) -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake(logprobs_by_call or [])
    adapter = ScoringJudgmentAdapter(
        fake, tokenize_content=tokenize, served_template=served_template
    )
    return adapter, fake


def get_fixtures() -> list[dict[str, Any]]:
    return [
        {
            "name": "mixed_questions_deterministic",
            "state": "Charged twice.",
            "model": "fake-judgment",
            "questions": {
                "billing": Noul(instructions="Billing?"),
                "route": Choice(
                    criteria={"billing": "Money", "technical": "Bugs"},
                    instructions="Pick:",
                ),
                "quality": Score(
                    criteria=["Poor", "Fair", "Good"],
                    instructions="Rate:",
                ),
            },
            "logprobs": [
                {"True": math.log(0.6), "False": math.log(0.4)},
                {"billing": math.log(0.7), "technical": math.log(0.3)},
                {"0": math.log(0.45), "1": math.log(0.45), "2": math.log(0.10)},
            ],
            "expect": {
                "kind": "success",
                "nouls": {"billing": 0.6},
                "choices": {"route": "billing"},
                "score_not_modal": {"quality": True},
            },
        },
        {
            "name": "invalid_empty_choice_zero_scorer_calls",
            "state": "x",
            "model": "m",
            "questions": {
                "bad": Choice(criteria={}, instructions="Pick:"),
            },
            "logprobs": [],
            "expect": {
                "kind": "error",
                "exc_type": "JudgmentValidationError",
                "calls": 0,
            },
        },
        {
            "name": "missing_scorer_candidate_label_one_call_scoring_validation",
            "state": "Charged twice.",
            "model": "m",
            "questions": {
                "route": Choice(
                    criteria={"billing": "Money", "technical": "Bugs"},
                    instructions="Pick:",
                ),
            },
            "logprobs": [{"billing": math.log(0.7)}],
            "expect": {
                "kind": "error",
                "exc_type": "ScoringValidationError",
                "calls": 1,
            },
        },
    ]


def exc_type_from_name(
    name: str,
) -> type[JudgmentValidationError | ScoringValidationError]:
    mapping: dict[str, type[JudgmentValidationError | ScoringValidationError]] = {
        "JudgmentValidationError": JudgmentValidationError,
        "ScoringValidationError": ScoringValidationError,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        msg = f"unknown exc_type: {name}"
        raise ValueError(msg) from exc
