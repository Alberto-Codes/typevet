"""Reconstruct consumer Noul, Choice and Score responses from verified engine answers.

See Also:
    - [typevet_consumer_bridge.questions][]: Request conversion.


Examples:
    ```python
    from typevet_consumer_bridge import BridgeSettings

    settings = BridgeSettings("http://localhost:8080", 30.0, "served-model")
    ```
"""

from collections.abc import Mapping

from judgevet import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage
from judgevet.domain.answers import Answer

from typevet.domain import Choice as EngineChoice
from typevet.domain import ChoiceAnswer as EngineChoiceAnswer
from typevet.domain import JudgmentResponse, TokenUsage
from typevet.domain import Noul as EngineNoul
from typevet.domain import NoulAnswer as EngineNoulAnswer
from typevet.domain import Score as EngineScore
from typevet.domain import ScoreAnswer as EngineScoreAnswer
from typevet_consumer_bridge.errors import BridgeResponseError


def _answer(
    answer: object, question: EngineNoul | EngineChoice | EngineScore
) -> Answer:
    if isinstance(question, EngineNoul) and isinstance(answer, EngineNoulAnswer):
        return NoulAnswer(noul=answer.noul)
    if isinstance(question, EngineChoice) and isinstance(answer, EngineChoiceAnswer):
        if (
            not isinstance(answer.probabilities, dict)
            or answer.probabilities.keys() != question.criteria.keys()
        ):
            raise BridgeResponseError("Invalid runtime Choice options.")
        return ChoiceAnswer(
            choice=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(question, EngineScore) and isinstance(answer, EngineScoreAnswer):
        legend = dict(enumerate(question.criteria))
        if (
            not isinstance(answer.legend, dict)
            or not isinstance(answer.probabilities, dict)
            or any(type(key) is not int for key in answer.legend)
            or answer.legend != legend
            or answer.probabilities.keys() != legend.keys()
        ):
            raise BridgeResponseError("Invalid runtime Score levels.")
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            legend=dict(answer.legend),
            probabilities=dict(answer.probabilities),
        )
    raise BridgeResponseError("Invalid runtime answer variant.")


def convert_response(
    response: object,
    questions: Mapping[str, EngineNoul | EngineChoice | EngineScore],
    model: str,
) -> SystemOneResponse:
    """Validate question variants and reconstruct consumer answers with exact distributions.

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
            answers[key] = _answer(response.answers[key], questions[key])
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except (ValueError, TypeError):
        raise BridgeResponseError("Invalid runtime answer values.") from None
    return SystemOneResponse(model=response.model, usage=usage, answers=answers)
