"""Pure mapping between judgevet and typevet for the judgevet bridge (#380).

The judgevet bridge imports this module after its judgevet import check. This
module holds the conversions, the error map, the off-option threshold rule and
the media grouping. It holds no port and never imports the bridge. With the
bridge, it is one of the two typevet modules that import judgevet, and no
package ``__init__`` imports it. Install the ``typevet[judgevet]`` extra to
use it.

Attributes:
    State (type): The content a judgevet port judges.
    Questions (type): judgevet questions or raw wire mappings, keyed by name.
    Options (type): judgevet ``provider_options``, or ``None``.
    ERROR_MAP (tuple): typevet error class to judgevet error class, in order.
    provider_error (function): Map one typevet failure to a judgevet error.
    to_typevet (function): Convert one judgevet question to typevet.
    to_judgevet (function): Convert one typevet answer to judgevet.
    to_response (function): Convert a typevet response to judgevet.
    pick_threshold (function): Pick the off-option threshold.
    image_groups (function): Group the questions by their bound images.
    add_counts (function): Add token counts that can be unknown.
    merge_responses (function): Merge image group responses into one.

Examples:
    ```python
    from judgevet.domain.questions import Noul as JevNoul

    from typevet.adapters.inbound.judgevet_mapping import to_typevet

    question = to_typevet("q", JevNoul(), 24)
    ```

See Also:
    - [typevet.adapters.inbound.judgevet][]: The bridge that uses this module
    - [typevet.domain.errors][]: The typevet errors the map reads
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from judgevet import ChoiceAnswer as JevChoiceAnswer
from judgevet import NoulAnswer as JevNoulAnswer
from judgevet import ScoreAnswer as JevScoreAnswer
from judgevet import SystemOneResponse, Usage
from judgevet.domain.answers import Answer as JevAnswer
from judgevet.domain.questions import Choice as JevChoice
from judgevet.domain.questions import Noul as JevNoul
from judgevet.domain.questions import Question as JevQuestion
from judgevet.domain.questions import Score as JevScore
from judgevet.media import ImageEvidence
from judgevet.providers import (
    ProviderCapabilityError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTransportError,
)

from typevet.domain.decision_execute import check_off_option_threshold
from typevet.domain.errors import (
    BackendHttpError,
    DecisionExecutionError,
    GemmaTemplateError,
    GenerationError,
    GenerationUnsupportedCapabilityError,
    JudgmentValidationError,
    ScoringUnsupportedCapabilityError,
    TransportError,
)
from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import SUPPORTED_IMAGE_MIME_TYPES, ImageInput

__all__ = [
    "ERROR_MAP",
    "Options",
    "Questions",
    "State",
    "add_counts",
    "image_groups",
    "merge_responses",
    "pick_threshold",
    "provider_error",
    "to_judgevet",
    "to_response",
    "to_typevet",
]

_CLIENT_ERRORS = range(400, 500)
"""HTTP statuses that mean the backend rejected the request itself."""

_WIRE_FIELDS = frozenset({"type", "instructions", "criteria"})

ERROR_MAP: tuple[tuple[type[GenerationError], type[ProviderError]], ...] = (
    (TransportError, ProviderTransportError),
    (GenerationUnsupportedCapabilityError, ProviderCapabilityError),
    (ScoringUnsupportedCapabilityError, ProviderCapabilityError),
    (GemmaTemplateError, ProviderCapabilityError),
    (JudgmentValidationError, ProviderRequestError),
    (DecisionExecutionError, ProviderRequestError),
    (GenerationError, ProviderResponseError),
)
"""typevet error class to judgevet error class; the first match wins."""

State = str | dict[str, Any] | list[Any]
Questions = Mapping[str, JevQuestion | Mapping[str, Any]]
Options = Mapping[str, object] | None


def provider_error(error: GenerationError) -> ProviderError:
    """Map one typevet failure to the judgevet provider error class.

    Args:
        error: The typevet failure.

    Returns:
        A judgevet error that carries the typevet message.
    """
    if isinstance(error, BackendHttpError):
        client_side = _CLIENT_ERRORS.start <= error.status_code < _CLIENT_ERRORS.stop
        target = ProviderRequestError if client_side else ProviderTransportError
    else:
        target = next(dst for src, dst in ERROR_MAP if isinstance(error, src))
    return target(str(error))


def _unpack(name: str, question: object) -> tuple[object, object, object]:
    """Read the kind, instructions and criteria of a typed or wire question.

    Args:
        name: The question name, for error messages.
        question: A judgevet question or a raw wire mapping.

    Returns:
        The kind, instructions and criteria, unchecked.

    Raises:
        ProviderRequestError: The question has an unknown shape or field.
    """
    if isinstance(question, JevNoul):
        return "noul", question.instructions, question.criteria
    if isinstance(question, JevChoice):
        return "choice", question.instructions, question.criteria
    if isinstance(question, JevScore):
        return "score", question.instructions, question.criteria
    if not isinstance(question, Mapping):
        raise ProviderRequestError(f"question {name!r} is not a typed or wire question")
    if set(question) - _WIRE_FIELDS:
        raise ProviderRequestError(f"question {name!r} has unknown fields")
    return question.get("type"), question.get("instructions"), question.get("criteria")


def _check_cap(name: str, count: int, cap: int) -> None:
    """Refuse a question with more options than the declared cap.

    Args:
        name: The question name, for the error message.
        count: The number of Choice options or Score levels.
        cap: The declared ``max_choice_options``.

    Raises:
        ProviderCapabilityError: ``count`` is above ``cap``.
    """
    if count > cap:
        msg = f"question {name!r} has {count} options; typevet accepts at most {cap}"
        raise ProviderCapabilityError(msg)


def to_typevet(name: str, question: object, cap: int) -> Question:
    """Convert one judgevet question to the typevet question of the same kind.

    Args:
        name: The question name.
        question: A judgevet question or a raw wire mapping.
        cap: The declared ``max_choice_options``.

    Returns:
        The typevet ``Noul``, ``Choice`` or ``Score``.

    Raises:
        ProviderRequestError: The question is malformed or of an unknown kind.
        ProviderCapabilityError: The question needs a capability typevet lacks.
    """
    kind, instructions, criteria = _unpack(name, question)
    if instructions is not None and not isinstance(instructions, str):
        msg = f"question {name!r}: typevet reads text instructions only"
        raise ProviderCapabilityError(msg)
    if kind == "noul" and (criteria is None or isinstance(criteria, dict)):
        return Noul(instructions=instructions, criteria=criteria)
    if kind == "choice" and isinstance(criteria, Mapping):
        _check_cap(name, len(criteria), cap)
        return Choice(criteria=criteria, instructions=instructions)
    if (
        kind == "score"
        and isinstance(criteria, Sequence)
        and not isinstance(criteria, (str, bytes))
    ):
        _check_cap(name, len(criteria), cap)
        return Score(criteria=criteria, instructions=instructions)
    raise ProviderRequestError(f"question {name!r} has an invalid kind or criteria")


def to_judgevet(answer: Answer | object) -> JevAnswer:
    """Convert one typevet answer to the judgevet answer of the same kind.

    Args:
        answer: The typevet answer.

    Returns:
        The judgevet ``NoulAnswer``, ``ChoiceAnswer`` or ``ScoreAnswer``.

    Raises:
        ProviderResponseError: The value is not a typevet answer.
    """
    if isinstance(answer, NoulAnswer):
        return JevNoulAnswer(noul=answer.noul)
    if isinstance(answer, ChoiceAnswer):
        return JevChoiceAnswer(
            answer.choice, answer.confidence, dict(answer.probabilities)
        )
    if isinstance(answer, ScoreAnswer):
        return JevScoreAnswer(
            answer.score,
            answer.confidence,
            dict(answer.legend),
            dict(answer.probabilities),
        )
    raise ProviderResponseError(f"typevet returned a {type(answer).__name__} answer")


def _check_threshold(value: object, source: str) -> None:
    """Refuse an invalid threshold with a message that does not show it.

    Args:
        value: The threshold to check.
        source: Where the value came from, for the message.

    Raises:
        ProviderRequestError: ``value`` is not ``None`` or a number in
            ``[0, 1]``. The error has no cause, so the value stays hidden.
    """
    try:
        check_off_option_threshold(value)
    except DecisionExecutionError:
        msg = f"{source} off_option_threshold must be None or in [0, 1]"
        raise ProviderRequestError(msg) from None


def pick_threshold(keyword: float | None, options: Options) -> float | None:
    """Pick the off-option threshold from the keyword or the provider options.

    Both values are checked before any backend call, also when the keyword
    wins, so an invalid option value fails closed.

    Args:
        keyword: The ``off_option_threshold`` keyword; it wins when set.
        options: judgevet ``provider_options``, or ``None`` for none.

    Returns:
        The keyword, else the option value, else ``None``.

    Raises:
        ProviderCapabilityError: The options carry a key other than
            ``off_option_threshold``.
        ProviderRequestError: The option value or the keyword is not ``None``
            or a number in ``[0, 1]``. The message does not show the value.
    """
    if set(options or {}) - {"off_option_threshold"}:
        raise ProviderCapabilityError("typevet reads only off_option_threshold")
    value = (options or {}).get("off_option_threshold")
    _check_threshold(value, "provider option")
    _check_threshold(keyword, "keyword")
    if keyword is None and isinstance(value, (int, float)):
        return float(value)
    return keyword


def to_response(response: JudgmentResponse, questions: Questions) -> SystemOneResponse:
    """Convert a typevet response, keeping the model id typevet reports.

    Args:
        response: The typevet response.
        questions: The judgevet questions, keyed by name.

    Returns:
        The judgevet response with one answer per question and one receipt
        per answer that has an off-option receipt.

    Raises:
        ProviderResponseError: Answer names differ from question names.
    """
    if set(response.answers) != set(questions):
        raise ProviderResponseError("typevet answer names differ from the questions")
    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    answers = {name: to_judgevet(value) for name, value in response.answers.items()}
    receipts = {name: r.as_dict() for name, r in response.off_option.items()}
    return SystemOneResponse(response.model, usage, answers, receipts)


def image_groups(
    questions: Questions, evidence: ImageEvidence
) -> list[tuple[Questions, tuple[ImageInput, ...] | None]]:
    """Group the questions by their bound images, in first-appearance order.

    Args:
        questions: The judgevet questions keyed by name.
        evidence: The validated judgevet image evidence.

    Returns:
        One ``(questions, images)`` pair per image set, with ``None`` images
        for the questions that have no binding.

    Raises:
        ProviderCapabilityError: An image type is outside typevet's supported
            set. The check runs before the bridge reads any image data.
    """
    if any(i.media_type not in SUPPORTED_IMAGE_MIME_TYPES for i in evidence.images):
        msg = "an image type is outside typevet's supported set"
        raise ProviderCapabilityError(msg)
    images = {i.id: ImageInput(i.data, i.media_type) for i in evidence.images}
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for name, question in questions.items():
        key = tuple(evidence.by_question.get(name, ()))
        groups.setdefault(key, {})[name] = question
    return [
        (group, tuple(images[i] for i in key) or None) for key, group in groups.items()
    ]


def add_counts(counts: Sequence[int | None]) -> int | None:
    """Add token counts, keeping the total unknown when any count is unknown.

    Args:
        counts: One count per image group.

    Returns:
        The sum, or ``None`` when any count is ``None``.
    """
    return None if None in counts else sum(c or 0 for c in counts)


def merge_responses(responses: Sequence[SystemOneResponse]) -> SystemOneResponse:
    """Merge the group responses into one, adding up the token counts.

    Args:
        responses: One response per image group, in group order.

    Returns:
        One response with every answer and receipt; a token count is ``None``
        when any group left it unknown.

    Raises:
        ProviderResponseError: The groups report different model ids.
    """
    if len({response.model for response in responses}) > 1:
        raise ProviderResponseError("typevet image groups reported different models")
    usage = Usage(
        add_counts([r.usage.input_tokens for r in responses]),
        add_counts([r.usage.output_tokens for r in responses]),
    )
    answers = {k: v for r in responses for k, v in r.answers.items()}
    receipts = {k: v for r in responses for k, v in r.receipts.items()}
    return SystemOneResponse(responses[0].model, usage, answers, receipts)
