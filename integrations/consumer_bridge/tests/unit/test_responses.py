"""Check explicit consumer answer reconstruction and malformed outputs.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/unit/test_responses.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

from dataclasses import replace

import pytest
from judgevet import ChoiceAnswer as ConsumerChoiceAnswer
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet import ScoreAnswer as ConsumerScoreAnswer
from typevet_consumer_bridge import BridgeResponseError
from typevet_consumer_bridge.responses import convert_response

from typevet.domain import (
    Choice,
    ChoiceAnswer,
    JudgmentResponse,
    Noul,
    Score,
    ScoreAnswer,
    TokenUsage,
)
from typevet.domain import NoulAnswer as EngineNoulAnswer

pytestmark = pytest.mark.unit
QUESTIONS = {" exact ": Noul()}
BASE = JudgmentResponse(model="model", answers={" exact ": EngineNoulAnswer(noul=0.7)})


def require(condition: bool, message: str) -> None:
    """Fail a named output assertion.

    Raises:
        AssertionError: If the named contract assertion fails.
    """
    if not condition:
        raise AssertionError(message)


@pytest.mark.parametrize("counts", [(None, None), (0, 0), (12, 3)])
def test_consumer_owned_values(counts: tuple[int | None, int | None]) -> None:
    """Copy exact probabilities and known or unknown usage into consumer classes."""
    response = replace(BASE, usage=TokenUsage(*counts))
    converted = convert_response(response, QUESTIONS, "model")
    require(type(converted) is SystemOneResponse, "consumer response")
    require(type(converted.usage) is Usage, "consumer usage")
    require(converted.usage == Usage(*counts), "usage values")
    require(type(converted.answers[" exact "]) is NoulAnswer, "consumer answer")
    require(converted.answers[" exact "] == NoulAnswer(noul=0.7), "exact probability")
    require(converted.answers is not response.answers, "new answer container")


@pytest.mark.parametrize(
    "response",
    [
        None,
        SystemOneResponse(model="model", usage=Usage()),
        replace(BASE, model="wrong"),
        replace(BASE, answers={}),
        replace(BASE, answers={"other": EngineNoulAnswer(noul=0.5)}),
        replace(BASE, answers={" exact ": NoulAnswer(noul=0.5)}),
        replace(
            BASE,
            answers={
                " exact ": ChoiceAnswer(
                    choice="x", confidence=1.0, probabilities={"x": 1.0}
                )
            },
        ),
        replace(BASE, usage=TokenUsage(input_tokens=-1)),
        replace(BASE, usage=TokenUsage(output_tokens=True)),
    ],
)
def test_invalid_response(response: object) -> None:
    """Reject wrong identities, foreign classes, variants and token counts."""
    with pytest.raises(BridgeResponseError):
        convert_response(response, QUESTIONS, "model")


def test_invalid_probability() -> None:
    """Revalidate even malformed engine-owned answer values."""
    answer = EngineNoulAnswer(noul=0.5)
    object.__setattr__(answer, "noul", float("nan"))
    with pytest.raises(BridgeResponseError):
        convert_response(replace(BASE, answers={" exact ": answer}), QUESTIONS, "model")


CHOICE = Choice(criteria={"seven": None, "forty_two": "42"})
SCORE = Score(criteria=["seven dollars", "forty-two dollars", "other"])
CHOICE_ANSWER = ChoiceAnswer(
    choice="forty_two",
    confidence=0.875,
    probabilities={"seven": 0.125, "forty_two": 0.875},
)
SCORE_ANSWER = ScoreAnswer(
    score=1.625,
    confidence=0.75,
    legend={0: "seven dollars", 1: "forty-two dollars", 2: "other"},
    probabilities={0: 0.125, 1: 0.125, 2: 0.75},
)


def test_choice_score_answers() -> None:
    """Copy exact distributions and expected value into distinct consumer classes."""
    response = JudgmentResponse(
        model="model", answers={"c": CHOICE_ANSWER, "s": SCORE_ANSWER}
    )
    converted = convert_response(response, {"c": CHOICE, "s": SCORE}, "model")
    choice, score = converted.answers["c"], converted.answers["s"]
    if not isinstance(choice, ConsumerChoiceAnswer) or not isinstance(
        score, ConsumerScoreAnswer
    ):
        raise TypeError("Incorrect consumer answer variants")
    require(type(choice) is ConsumerChoiceAnswer, "consumer Choice class")
    require(type(score) is ConsumerScoreAnswer, "consumer Score class")
    require(
        choice == ConsumerChoiceAnswer("forty_two", 0.875, CHOICE_ANSWER.probabilities),
        "exact Choice values",
    )
    require(
        score
        == ConsumerScoreAnswer(
            1.625, 0.75, SCORE_ANSWER.legend, SCORE_ANSWER.probabilities
        ),
        "expected value, confidence and zero-based legend",
    )
    require(
        choice.probabilities is not CHOICE_ANSWER.probabilities,
        "fresh Choice distribution",
    )
    require(
        score.probabilities is not SCORE_ANSWER.probabilities,
        "fresh Score distribution",
    )
    require(score.legend is not SCORE_ANSWER.legend, "fresh Score legend")


@pytest.mark.parametrize(
    "question,answer",
    [
        (CHOICE, SCORE_ANSWER),
        (SCORE, CHOICE_ANSWER),
        (
            CHOICE,
            replace(CHOICE_ANSWER, probabilities={"other": 0.125, "forty_two": 0.875}),
        ),
        (
            SCORE,
            replace(
                SCORE_ANSWER, legend={0: "wrong", 1: "forty-two dollars", 2: "other"}
            ),
        ),
        (
            SCORE,
            ScoreAnswer(
                score=1.5,
                confidence=0.5,
                legend={1: "a", 2: "b"},
                probabilities={1: 0.5, 2: 0.5},
            ),
        ),
    ],
)
def test_invalid_choice_score_response(
    question: Choice | Score, answer: ChoiceAnswer | ScoreAnswer
) -> None:
    """Reject mismatched variants, option keys and rubric descriptions."""
    with pytest.raises(BridgeResponseError):
        convert_response(
            JudgmentResponse(model="model", answers={"q": answer}),
            {"q": question},
            "model",
        )


def test_boolean_legend_key() -> None:
    """Reject boolean legend keys even though Python equates them with integers."""
    answer = replace(
        SCORE_ANSWER,
        legend={False: "seven dollars", 1: "forty-two dollars", 2: "other"},
    )
    with pytest.raises(BridgeResponseError):
        convert_response(
            JudgmentResponse(model="model", answers={"q": answer}),
            {"q": SCORE},
            "model",
        )
