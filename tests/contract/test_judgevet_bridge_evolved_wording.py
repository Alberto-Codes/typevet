"""Evolved Noul wording crosses the judgevet bridge unchanged (#366).

An evolution run renders a multi-part candidate to the existing question
fields: ``instructions`` as text and Noul ``criteria`` as a true/false dict.
These tests send such a Noul through ``TypevetSystemOnePort`` over the offline
typevet judgment path. They check that the inner typevet port and the scoring
prefix receive the exact texts, for a typed judgevet question and for a raw
wire dict. An object ``instructions`` stays refused with no text in the error.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from judgevet.domain.questions import Noul
from judgevet.providers import ProviderCapabilityError

from tests.fixtures.judgevet_bridge import FAKE_MODEL, LOGPROBS, STATE, judgment_port
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.domain.judgment_questions import Noul as TvNoul
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

EVOLVED_INSTRUCTIONS = "Does the message push the reader to pay before a check?"
"""An evolved instructions text, unlike any fixture question."""

EVOLVED_CRITERIA = {
    "true": "The message demands payment through an unusual channel.",
    "false": "The message asks for nothing or uses a normal channel.",
}
"""Evolved true/false texts; the typevet Noul default is no criteria."""

OBJECT_RULE = "rank urgency cues above sender identity"
"""An object-instructions value that the refusal must not echo."""


class RecordingJudgment:
    """A typevet judgment port that records each call and delegates."""

    def __init__(self, scorer: ScriptedScoringFake) -> None:
        """Wrap the offline typevet judgment path over ``scorer``.

        Args:
            scorer: The scripted scorer that records each scoring prefix.
        """
        self.inner = judgment_port(scorer=scorer)
        self.calls: list[Mapping[str, Any]] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        self.calls.append(questions)
        return self.inner.judge(state, questions, model, media=media)


def _bridge() -> tuple[TypevetSystemOnePort, RecordingJudgment, ScriptedScoringFake]:
    scorer = ScriptedScoringFake(logprobs=LOGPROBS)
    recorder = RecordingJudgment(scorer)
    return TypevetSystemOnePort(recorder), recorder, scorer


EVOLVED_QUESTIONS = {
    "typed": Noul(instructions=EVOLVED_INSTRUCTIONS, criteria=dict(EVOLVED_CRITERIA)),
    "wire": {
        "type": "noul",
        "instructions": EVOLVED_INSTRUCTIONS,
        "criteria": dict(EVOLVED_CRITERIA),
    },
}
"""The same evolved Noul as a typed judgevet question and as a wire dict."""


@pytest.mark.parametrize("shape", sorted(EVOLVED_QUESTIONS))
def test_evolved_noul_texts_reach_the_inner_port_unchanged(shape: str) -> None:
    port, recorder, _ = _bridge()
    port.system_one(STATE, {"is_scam": EVOLVED_QUESTIONS[shape]}, FAKE_MODEL)
    (questions,) = recorder.calls
    question = questions["is_scam"]
    assert isinstance(question, TvNoul)
    assert question.instructions == EVOLVED_INSTRUCTIONS
    assert question.criteria == EVOLVED_CRITERIA


@pytest.mark.parametrize("shape", sorted(EVOLVED_QUESTIONS))
def test_evolved_noul_texts_reach_the_scoring_prefix(shape: str) -> None:
    port, _, scorer = _bridge()
    response = port.system_one(STATE, {"is_scam": EVOLVED_QUESTIONS[shape]}, FAKE_MODEL)
    assert set(response.answers) == {"is_scam"}
    (request,) = scorer.calls
    for text in (EVOLVED_INSTRUCTIONS, *EVOLVED_CRITERIA.values()):
        assert text in request.prefix


OBJECT_INSTRUCTIONS = {
    "typed": Noul(instructions={"rule": OBJECT_RULE}, criteria=dict(EVOLVED_CRITERIA)),
    "wire": {
        "type": "noul",
        "instructions": {"rule": OBJECT_RULE},
        "criteria": dict(EVOLVED_CRITERIA),
    },
}
"""A Noul with object instructions, as a typed question and as a wire dict."""


@pytest.mark.parametrize("shape", sorted(OBJECT_INSTRUCTIONS))
def test_object_instructions_are_refused_without_echo(shape: str) -> None:
    port, recorder, scorer = _bridge()
    with pytest.raises(ProviderCapabilityError) as caught:
        port.system_one(STATE, {"is_scam": OBJECT_INSTRUCTIONS[shape]}, FAKE_MODEL)
    message = str(caught.value)
    assert OBJECT_RULE not in message
    for text in EVOLVED_CRITERIA.values():
        assert text not in message
    assert recorder.calls == []
    assert scorer.calls == []
