"""Contract: ScriptedJudgmentFake agrees with ContractJudgmentFake (#387).

For every success fixture in ``tests/fixtures/judgment_contract.py``, the
public fake built from matching distributions returns the same ``noul``,
``choice``, ``score`` and ``probabilities`` as the contract fake. A
two-level Score case checks the expected-level rule against the scoring
adapter (#400).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

from tests.fixtures.judgment_contract import fake_for, get_fixtures
from tests.fixtures.judgment_scoring_contract import adapter_for
from typevet.domain.judgment_answers import Answer, ChoiceAnswer, NoulAnswer
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.testing import ScriptedJudgmentFake

_SUCCESS = [f for f in get_fixtures() if f["expect"]["kind"] == "success"]


def _from_answer(answer: Answer) -> float | dict[Any, float]:
    if isinstance(answer, NoulAnswer):
        return answer.noul
    return dict(answer.probabilities)


def _default(question: Any) -> float | dict[Any, float]:
    if isinstance(question, Noul):
        return 0.5
    if isinstance(question, Choice):
        labels = list(question.criteria)
        n = len(labels)
        weights = {label: 1.0 / n for label in labels}
        weights[labels[0]] = 1.0 - (n - 1) / n
        return weights
    assert isinstance(question, Score)
    mid = len(question.criteria) // 2
    return {mid: 1.0}


def _distributions(fixture: dict[str, Any]) -> dict[str, Any]:
    scripted: Mapping[str, Answer] = fixture.get("fake", {}).get("answers") or {}
    return {
        name: _from_answer(scripted[name]) if name in scripted else _default(q)
        for name, q in fixture["questions"].items()
    }


def _view(answer: Answer) -> dict[str, Any]:
    if isinstance(answer, NoulAnswer):
        return {"noul": answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return {"choice": answer.choice, "probabilities": answer.probabilities}
    return {"score": answer.score, "probabilities": answer.probabilities}


@pytest.mark.contract
@pytest.mark.parametrize("fixture", _SUCCESS, ids=lambda f: f["name"])
def test_scripted_fake_agrees_with_contract_fake(fixture: dict[str, Any]) -> None:
    args = (fixture["state"], fixture["questions"], fixture["model"])
    expected = fake_for(fixture).judge(*args)
    actual = ScriptedJudgmentFake(_distributions(fixture)).judge(*args)
    assert set(actual.answers) == set(expected.answers)
    for name, answer in expected.answers.items():
        assert _view(actual.answers[name]) == _view(answer), name


@pytest.mark.contract
def test_agreement_covers_every_question_kind() -> None:
    kinds = {type(q).__name__ for f in _SUCCESS for q in f["questions"].values()}
    assert kinds == {"Noul", "Choice", "Score"}


@pytest.mark.contract
def test_two_level_score_matches_adapter_expected_level_rule() -> None:
    question = Score(criteria=["Poor", "Good"], instructions="Rate:")
    questions = {"quality": question}
    fake = ScriptedJudgmentFake({"quality": {0: 1.0, 1: 3.0}})
    actual = fake.judge("text", questions, "m").scores["quality"]
    rule = sum(level * p for level, p in actual.probabilities.items())
    assert actual.score == pytest.approx(rule)
    assert actual.score == pytest.approx(0.75)
    logprobs = {"0": math.log(0.25), "1": math.log(0.75)}
    adapter, _ = adapter_for(logprobs_by_call=[logprobs])
    expected = adapter.judge("text", questions, "m").scores["quality"]
    assert actual.score == pytest.approx(expected.score)
    assert actual.confidence == pytest.approx(expected.confidence)
    assert actual.probabilities == pytest.approx(expected.probabilities)
