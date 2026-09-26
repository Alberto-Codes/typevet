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
from judgevet import NoulAnswer, SystemOneResponse, Usage
from typevet_consumer_bridge import BridgeResponseError
from typevet_consumer_bridge.responses import convert_response

from typevet.domain import ChoiceAnswer, JudgmentResponse, Noul, TokenUsage
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
