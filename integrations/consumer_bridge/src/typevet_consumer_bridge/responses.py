"""Reconstruct consumer-owned responses from verified engine answers.

See Also:
    - [typevet_consumer_bridge.questions][]: Request conversion.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from collections.abc import Mapping

from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.answers import Answer

from typevet.domain import JudgmentResponse, TokenUsage
from typevet.domain import Noul as EngineNoul
from typevet.domain import NoulAnswer as EngineNoulAnswer
from typevet_consumer_bridge.errors import BridgeResponseError


def convert_response(
    response: object, questions: Mapping[str, EngineNoul], model: str
) -> SystemOneResponse:
    """Validate identity and reconstruct each answer and usage value.

    Args:
        response: Engine result, never a consumer response.
        questions: Validated request questions.
        model: Session-bound requested model identity.

    Returns:
        New consumer-owned response, answers and usage.

    Raises:
        BridgeResponseError: If engine output violates the supported contract.
    """
    if (
        not isinstance(response, JudgmentResponse)
        or response.model != model
        or not isinstance(response.answers, dict)
        or response.answers.keys() != questions.keys()
        or not isinstance(response.usage, TokenUsage)
    ):
        raise BridgeResponseError("Invalid runtime response.")
    answers: dict[str, Answer] = {}
    try:
        for key in questions:
            answer = response.answers[key]
            if not isinstance(answer, EngineNoulAnswer):
                raise BridgeResponseError("Invalid runtime answer variant.")
            answers[key] = NoulAnswer(noul=answer.noul)
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except (ValueError, TypeError):
        raise BridgeResponseError("Invalid runtime answer values.") from None
    return SystemOneResponse(model=response.model, usage=usage, answers=answers)
