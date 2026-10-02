"""Unit tests for the public ScriptedJudgmentFake (#387)."""

from __future__ import annotations

from typing import Any

import pytest

from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort
from typevet.testing import ScriptedJudgmentFake

_PNG = b"\x89PNG\r\n\x1a\nfake"
_ROUTE = Choice(criteria={"billing": "Money", "technical": "Bugs"}, instructions="Q")
_QUALITY = Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:")


@pytest.mark.unit
def test_noul_returns_scripted_probability_of_true() -> None:
    fake: JudgmentPort = ScriptedJudgmentFake({"billing": 0.8})
    response = fake.judge("text", {"billing": Noul()}, "m")
    assert response.model == "m"
    assert response.nouls["billing"].noul == 0.8
    assert set(response.answers) == {"billing"}


@pytest.mark.unit
def test_choice_normalises_weights_and_picks_argmax() -> None:
    fake = ScriptedJudgmentFake({"route": {"billing": 1.0, "technical": 3.0}})
    answer = fake.judge("text", {"route": _ROUTE}, "m").choices["route"]
    assert answer.choice == "technical"
    assert answer.confidence == pytest.approx(0.75)
    assert answer.probabilities == pytest.approx({"billing": 0.25, "technical": 0.75})


@pytest.mark.unit
def test_choice_fills_unlisted_labels_with_zero() -> None:
    fake = ScriptedJudgmentFake({"route": {"billing": 2.0}})
    answer = fake.judge("text", {"route": _ROUTE}, "m").choices["route"]
    assert answer.probabilities == {"billing": 1.0, "technical": 0.0}


@pytest.mark.unit
def test_score_normalises_weights_and_returns_argmax_level() -> None:
    fake = ScriptedJudgmentFake({"quality": {0: 1.0, 2: 3.0}})
    answer = fake.judge("text", {"quality": _QUALITY}, "m").scores["quality"]
    assert answer.score == 2.0
    assert answer.confidence == pytest.approx(0.75)
    assert answer.probabilities == pytest.approx({0: 0.25, 1: 0.0, 2: 0.75})
    assert answer.legend == {0: "Poor", 1: "Fair", 2: "Good"}


@pytest.mark.unit
@pytest.mark.parametrize("count", [0, 1, 3])
def test_media_of_any_length_is_ignored(count: int) -> None:
    images = tuple(ImageInput(data=_PNG, mime_type="image/png") for _ in range(count))
    fake = ScriptedJudgmentFake({"billing": 0.3, "route": {"billing": 1.0}})
    questions = {"billing": Noul(), "route": _ROUTE}
    with_media = fake.judge("text", questions, "m", media=images)
    without = fake.judge("text", questions, "m")
    assert with_media == without


@pytest.mark.unit
def test_unknown_question_name_raises() -> None:
    fake = ScriptedJudgmentFake({"billing": 0.5})
    with pytest.raises(JudgmentValidationError, match="urgent"):
        fake.judge("text", {"urgent": Noul()}, "m")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "distribution"),
    [
        (_ROUTE, {"refund": 1.0}),
        (_QUALITY, {3: 1.0}),
        (_QUALITY, {-1: 1.0}),
        (_ROUTE, {"billing": -1.0, "technical": 2.0}),
        (_ROUTE, {"billing": 0.0}),
        (_ROUTE, {"billing": float("nan")}),
        (_QUALITY, {0: 0.0, 1: 0.0}),
        (_ROUTE, 0.5),
        (_QUALITY, {"0": 1.0}),
        (Noul(), 1.5),
        (Noul(), -0.1),
        (Noul(), True),
        (Noul(), {"True": 1.0}),
        ({"type": "noul"}, 0.5),
    ],
)
def test_invalid_distribution_raises(question: Any, distribution: Any) -> None:
    fake = ScriptedJudgmentFake({"q": distribution})
    with pytest.raises(JudgmentValidationError):
        fake.judge("text", {"q": question}, "m")


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [1.5, -0.1, True])
def test_invalid_off_option_threshold_raises(threshold: Any) -> None:
    fake = ScriptedJudgmentFake({"billing": 0.5})
    with pytest.raises(JudgmentValidationError):
        fake.judge("text", {"billing": Noul()}, "m", off_option_threshold=threshold)


@pytest.mark.unit
def test_valid_off_option_threshold_does_not_flag() -> None:
    fake = ScriptedJudgmentFake({"billing": 0.5})
    response = fake.judge("t", {"billing": Noul()}, "m", off_option_threshold=0.2)
    assert response.nouls["billing"].off_option_flag is False


@pytest.mark.unit
def test_blank_model_raises() -> None:
    fake = ScriptedJudgmentFake({"billing": 0.5})
    with pytest.raises(JudgmentValidationError):
        fake.judge("text", {"billing": Noul()}, "  ")
