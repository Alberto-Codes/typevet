"""Contract: ScriptedJudgmentFake agrees with ContractJudgmentFake (#387).

For every success fixture in ``tests/fixtures/judgment_contract.py``, the
public fake built from matching distributions returns the same ``noul``,
``choice``, ``score`` and ``probabilities`` as the contract fake.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from tests.fixtures.judgment_contract import fake_for, get_fixtures
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
