"""Contract tests for ScoringJudgmentAdapter over CandidateScoringPort."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import (
    adapter_for,
    exc_type_from_name,
    get_fixtures,
)
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.ports.judgment import JudgmentPort


@pytest.mark.contract
def test_adapter_prompt_maps_controls_while_answers_keep_original_labels() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=[
            {label: math.log(p) for label, p in (("True", 0.6), ("False", 0.4))},
            {
                label: math.log(p)
                for label, p in (("billing", 0.75), ("technical", 0.25))
            },
            {label: math.log(p) for label, p in (("0", 0.5), ("1", 0.3), ("2", 0.2))},
        ],
    )
    questions = {
        "noul": Noul(
            instructions="Billing issue?", criteria={"true": "Yes", "false": "No"}
        ),
        "route": Choice(
            criteria={"billing": "Money", "technical": "Bugs"},
            instructions="Pick:",
        ),
        "quality": Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:"),
    }
    response = adapter.judge("Charged twice.", questions, "fake-judgment")
    assert response.nouls["noul"].noul == pytest.approx(0.6)
    assert response.choices["route"].choice == "billing"
    assert response.scores["quality"].legend == {0: "Poor", 1: "Fair", 2: "Good"}

    noul_prefix = fake.calls[0].prefix
    false_pos = noul_prefix.index("Control 0 → false")
    true_pos = noul_prefix.index("Control 1 → true")
    assert false_pos < true_pos
    choice_prefix = fake.calls[1].prefix
    assert "\n0 → billing: Money" in choice_prefix
    score_prefix = fake.calls[2].prefix
    assert "Control 1 → 1: Fair" in score_prefix


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda f: f["name"])
def test_scoring_judgment_adapter_honors_fixture(fixture: dict) -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=fixture["logprobs"],
        pinned_model=fixture.get("pinned_model"),
    )
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
