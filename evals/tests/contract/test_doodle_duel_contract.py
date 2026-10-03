"""Offline contract: a doodle Choice request through ScoringJudgmentAdapter.

The fake scorer replaces the served model. These tests prove that one doodle
becomes one judgment with one PNG and one ``Choice`` over every category, and
that the answer carries the full distribution. They say nothing about model
quality.
"""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import Choice, ImageInput
from typevet.domain.media import count_media_markers
from typevet_evals.datasets.quickdraw import Doodle
from typevet_evals.doodle_duel import (
    DOODLE_CATEGORIES,
    DOODLE_QUESTION,
    build_doodle_request,
    doodle_questions,
    judge_doodle,
    render_strokes,
)

pytestmark = pytest.mark.contract

MODEL = "fake-doodle"
DOODLE = Doodle(
    key_id="4567",
    word="castle",
    countrycode="GB",
    strokes=(((10, 10), (200, 10), (200, 200)), ((128, 128),)),
)


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def test_question_is_one_choice_over_every_category_in_order() -> None:
    questions = doodle_questions()
    assert list(questions) == [DOODLE_QUESTION]
    question = questions[DOODLE_QUESTION]
    assert isinstance(question, Choice)
    assert tuple(question.criteria) == DOODLE_CATEGORIES
    request = build_doodle_request(DOODLE, png=render_strokes(DOODLE.strokes))
    assert "castle" not in request.state
    assert request.key_id == "4567"
    assert request.true_label == "castle"


def test_fake_port_sees_one_png_and_returns_full_distribution() -> None:
    weights = {label: 1.0 for label in DOODLE_CATEGORIES}
    weights["castle"] = 5.0
    total = sum(weights.values())
    fake = SequentialScoringFake(
        [{label: math.log(w / total) for label, w in weights.items()}]
    )
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=_tokenize,
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    png = render_strokes(DOODLE.strokes)
    request = build_doodle_request(DOODLE, png=png)

    response = judge_doodle(adapter, request, MODEL)

    assert request.media == (ImageInput(data=png, mime_type="image/png"),)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert [image.data for image in call.media] == [png]
    assert count_media_markers(call.prefix) == 1
    answer = response.choices[DOODLE_QUESTION]
    assert answer.choice == "castle"
    assert list(answer.probabilities) == list(DOODLE_CATEGORIES)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.probabilities["castle"] == pytest.approx(5.0 / total)
