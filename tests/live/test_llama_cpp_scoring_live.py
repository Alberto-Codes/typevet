"""Live pre-sampling scoring against local llama.cpp (opt-in)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)

_LLAMA = load_llama_settings()


@pytest.fixture
def llama_scoring_model() -> str:
    gate_live(_LLAMA)
    assert _LLAMA.default_model is not None
    return _LLAMA.default_model


@pytest.mark.live
def test_llama_cpp_scoring_live_receipt(llama_scoring_model: str) -> None:
    """Smoke: POST /completion with n_probs=vocab returns candidate logprobs."""
    live_settings = replace(_LLAMA, timeout=600.0)
    request = CandidateScoringRequest(
        model=llama_scoring_model,
        prefix="Answer:",
        candidates=(
            CandidateTokenSpec("a", (32,)),
            CandidateTokenSpec("b", (33,)),
        ),
    )
    with LlamaCppCandidateScoringAdapter(
        base_url=live_settings.base_url,
        timeout=live_settings.timeout,
        n_vocab=262144,
    ) as adapter:
        result = adapter.score_candidates(request)
    assert len(result.candidates) == 2
    for scored in result.candidates:
        assert scored.logprob <= 0.0
