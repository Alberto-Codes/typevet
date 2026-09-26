"""Contract tests for ScoringJudgmentAdapter over CandidateScoringPort."""

from __future__ import annotations

import pytest

from tests.fixtures.judgment_scoring_contract import (
    adapter_for,
    exc_type_from_name,
    get_fixtures,
)
from typevet.ports.judgment import JudgmentPort


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda f: f["name"])
def test_scoring_judgment_adapter_honors_fixture(fixture: dict) -> None:
    adapter, fake = adapter_for(logprobs_by_call=fixture["logprobs"])
    expect = fixture["expect"]
    if expect["kind"] == "success":
        response = adapter.judge(
            fixture["state"],
            fixture["questions"],
            fixture["model"],
        )
        assert response.model == fixture["model"]
        for name, prob in expect.get("nouls", {}).items():
            assert response.nouls[name].noul == pytest.approx(prob)
        for name, label in expect.get("choices", {}).items():
            assert response.choices[name].choice == label
        for name in expect.get("score_not_modal", {}):
            ans = response.scores[name]
            modal_level = max(ans.probabilities, key=ans.probabilities.__getitem__)
            assert ans.score != float(modal_level)
        assert len(fake.calls) == len(fixture["logprobs"])
        return

    exc_type = exc_type_from_name(expect["exc_type"])
    with pytest.raises(exc_type):
        adapter.judge(fixture["state"], fixture["questions"], fixture["model"])
    assert len(fake.calls) == expect.get("calls", 0)


@pytest.mark.contract
def test_scoring_judgment_adapter_structurally_implements_judgment_port() -> None:
    adapter, _ = adapter_for()

    def _accept(port: JudgmentPort) -> None:
        assert callable(port.judge)

    _accept(adapter)
