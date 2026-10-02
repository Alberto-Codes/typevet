"""Contract: the fake judgment session matches the real session shape (#388).

``GemmaNativeVisionSession``, ``VllmJudgmentSession`` and
``FakeJudgmentSession`` all declare ``port`` and ``model`` fields, so a caller
of ``open_judgment`` reads ``session.port`` and ``session.model`` on every
backend. The fake session judges a Noul, a Choice and a Score offline.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import pytest

from typevet.adapters.inbound import open_judgment
from typevet.adapters.inbound.fake_backend import FakeJudgmentSession
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
)
from typevet.adapters.outbound.vllm.judgment_factory import VllmJudgmentSession
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Noul, Score

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "session_type",
    [GemmaNativeVisionSession, VllmJudgmentSession, FakeJudgmentSession],
)
def test_every_session_declares_port_and_model(
    session_type: type[DataclassInstance],
) -> None:
    names = {f.name for f in fields(session_type)}
    assert {"port", "model"} <= names


def test_fake_session_judges_each_question_kind_offline() -> None:
    questions = {
        "billing": Noul(instructions="Is this about billing?"),
        "route": Choice(criteria={"billing": "Money", "technical": "Bugs"}),
        "quality": Score(criteria=["Poor", "Fair", "Good"]),
    }
    with open_judgment({"TYPEVET_BACKEND": "fake"}) as session:
        assert isinstance(session, FakeJudgmentSession)
        response = session.port.judge("I was charged twice.", questions, session.model)

    assert response.model == session.model == "fake"
    assert isinstance(response.answers["billing"], NoulAnswer)
    assert isinstance(response.answers["route"], ChoiceAnswer)
    assert isinstance(response.answers["quality"], ScoreAnswer)
    assert response.choices["route"].choice in questions["route"].criteria
    assert response.scores["quality"].score in {0.0, 1.0, 2.0}
