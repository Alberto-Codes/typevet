"""Contract: the scoring fake and the llama.cpp adapter agree on off-option mass."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.fixtures.scoring_contract import ContractScoringFake, get_off_option_fixture
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_response import CandidateScoringResult


def _llama_cpp_result(fixture: dict[str, Any]) -> CandidateScoringResult:
    payload = {"completion_probabilities": [{"top_logprobs": fixture["top_logprobs"]}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=client, n_vocab=fixture["n_vocab"]
    )
    return adapter.score_candidates(fixture["request"])


@pytest.mark.contract
def test_fake_and_llama_cpp_agree_on_off_option_mass() -> None:
    fixture = get_off_option_fixture()
    from_fake = ContractScoringFake(**fixture["fake"]).score_candidates(
        fixture["request"]
    )
    from_adapter = _llama_cpp_result(fixture)
    expect = fixture["expect"]
    for result in (from_fake, from_adapter):
        assert [c.logprob for c in result.candidates] == expect["logprobs"]
        assert result.off_option_mass == pytest.approx(
            expect["off_option_mass"], abs=1e-12
        )
