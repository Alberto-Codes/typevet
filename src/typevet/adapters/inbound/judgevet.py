"""judgevet provider bridge: judgevet's ``SystemOnePort`` over typevet judgment (#284).

judgevet calls into typevet through this module. It is the only typevet module
that imports judgevet, and no package ``__init__`` imports it, so a bare
``import typevet`` stays free of judgevet. Install the ``typevet[judgevet]``
extra to use it.

The bridge is compatible in shape with Jev, not equivalent in judgment. The
answers come from the typevet backend's model, and calibration does not
transfer between judge models.

Declared capabilities (``BridgeCapabilities``):

- Logprobs are required. typevet scores every answer from next-token
  logprobs. A backend that returns none raises a typed ``ProviderError``;
  the bridge never falls back to generated text.
- Choice options and Score levels are capped at ``max_choice_options``
  (default ``DEFAULT_MAX_CHOICE_OPTIONS``, typevet's ``MAX_ENUM_CHOICES``,
  24). A larger question is refused with ``ProviderCapabilityError`` before
  any backend call.
- Instructions must be text or absent. Object and array instructions are
  refused with ``ProviderCapabilityError``.
- Media: ``TypevetSystemOnePort`` has no media methods, so
  ``judgevet.media.judge_with_images`` refuses it.
  ``TypevetMediaSystemOnePort`` implements ``MediaSystemOnePort`` for the
  declared ``MediaCapabilities``. It groups the questions by their bound
  images, in binding order, and sends one typevet judgment per group. The
  backend still gets one scoring call per question. Questions with no
  images go as one text judgment. The answers merge into one response;
  the groups must report the same model id, and token counts add up.
- Off-option guard: the ``off_option_threshold`` keyword, or the
  ``off_option_threshold`` key of judgevet ``provider_options`` when the
  keyword is ``None``, goes to the typevet port. Any other option key is
  refused with ``ProviderCapabilityError``. An invalid option value raises
  ``ProviderRequestError``; the message does not show the value.
- Receipts: each typevet ``OffOptionReceipt`` goes into
  ``SystemOneResponse.receipts`` under its answer name, with
  ``off_option_mass``, ``off_option_threshold`` and ``off_option_flag``. An
  answer with no receipt has no ``receipts`` entry.

Error mapping (typevet error to judgevet error):

- ``TransportError``: ``ProviderTransportError``.
- ``BackendHttpError``: ``ProviderRequestError`` for status 400 to 499, else
  ``ProviderTransportError``.
- ``GenerationUnsupportedCapabilityError``,
  ``ScoringUnsupportedCapabilityError`` and ``GemmaTemplateError``:
  ``ProviderCapabilityError``.
- ``JudgmentValidationError`` and ``DecisionExecutionError``:
  ``ProviderRequestError``.
- Every other typevet ``GenerationError``: ``ProviderResponseError``.

Other exceptions keep their type. The judgevet error keeps the typevet message
and chains the typevet error as its cause.

Examples:
    ```python
    from judgevet.adapters.inbound.cli import create_cli_app

    from typevet.adapters.inbound.backend_settings import open_judgment
    from typevet.adapters.inbound.judgevet import provider_factory

    app = create_cli_app(provider_factory=provider_factory(open_judgment))
    ```

See Also:
    - [typevet.ports.judgment][]: The ``JudgmentPort`` the bridge wraps
    - [typevet.domain.judgment_questions][]: The typevet question types
    - [typevet.domain.errors][]: The typevet errors the bridge maps

Attributes:
    DEFAULT_MAX_CHOICE_OPTIONS (int): Declared Choice and Score option cap.
    BridgeCapabilities (type): Capabilities a bridge port declares.
    JudgmentSession (type): A typevet session that exposes a ``port``.
    TypevetSystemOnePort (type): Text ``SystemOnePort`` over ``JudgmentPort``.
    TypevetMediaSystemOnePort (type): ``MediaSystemOnePort`` over ``JudgmentPort``.
    AsyncTypevetSystemOnePort (type): ``AsyncSystemOnePort`` over the bridge.
    provider_factory (function): Build a judgevet ``ProviderFactory``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

try:
    from judgevet import ChoiceAnswer as JevChoiceAnswer
    from judgevet import NoulAnswer as JevNoulAnswer
    from judgevet import ScoreAnswer as JevScoreAnswer
    from judgevet import SystemOnePort, SystemOneResponse, Usage
    from judgevet.domain.answers import Answer as JevAnswer
    from judgevet.domain.questions import Choice as JevChoice
    from judgevet.domain.questions import Noul as JevNoul
    from judgevet.domain.questions import Question as JevQuestion
    from judgevet.domain.questions import Score as JevScore
    from judgevet.media import ImageEvidence, MediaCapabilities
    from judgevet.providers import (
        ProviderCapabilityError,
        ProviderError,
        ProviderFactory,
        ProviderRequestError,
        ProviderResponseError,
        ProviderTransportError,
    )
except ModuleNotFoundError as missing:
    if (missing.name or "").split(".")[0] != "judgevet":
        raise
    msg = "typevet.adapters.inbound.judgevet needs judgevet; install typevet[judgevet]"
    raise ImportError(msg) from missing

from typevet.domain.decision_execute import check_off_option_threshold
from typevet.domain.decisions import MAX_ENUM_CHOICES
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
from typevet.ports.judgment import JudgmentPort

__all__ = [
    "DEFAULT_MAX_CHOICE_OPTIONS",
    "AsyncTypevetSystemOnePort",
    "BridgeCapabilities",
    "JudgmentSession",
    "TypevetMediaSystemOnePort",
    "TypevetSystemOnePort",
    "provider_factory",
]

DEFAULT_MAX_CHOICE_OPTIONS = MAX_ENUM_CHOICES
"""typevet's native Choice and Score execute limit (#287)."""

_CLIENT_ERRORS = range(400, 500)
"""HTTP statuses that mean the backend rejected the request itself."""

_MIN_OPTIONS = 2
"""typevet needs two options to score a Choice or a Score."""

_WIRE_FIELDS = frozenset({"type", "instructions", "criteria"})

_ERROR_MAP: tuple[tuple[type[GenerationError], type[ProviderError]], ...] = (
    (TransportError, ProviderTransportError),
    (GenerationUnsupportedCapabilityError, ProviderCapabilityError),
    (ScoringUnsupportedCapabilityError, ProviderCapabilityError),
    (GemmaTemplateError, ProviderCapabilityError),
    (JudgmentValidationError, ProviderRequestError),
    (DecisionExecutionError, ProviderRequestError),
    (GenerationError, ProviderResponseError),
)

State = str | dict[str, Any] | list[Any]
Questions = Mapping[str, JevQuestion | Mapping[str, Any]]
Options = Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class BridgeCapabilities:
    """Capabilities a bridge port declares; the bridge never infers them.

    Attributes:
        logprobs_required (bool): Always ``True``; answers come from logprobs.
        max_choice_options (int): Most Choice options or Score levels accepted.
        media (MediaCapabilities | None): Declared media support, or ``None``.

    Examples:
        ```python
        caps = BridgeCapabilities(True, 24, None)
        assert caps.logprobs_required
        ```
    """

    logprobs_required: bool
    max_choice_options: int
    media: MediaCapabilities | None


class JudgmentSession(Protocol):
    """A typevet judgment session, such as the one ``open_judgment`` yields.

    Attributes:
        port (JudgmentPort): The session's judgment port.

    Examples:
        ```python
        def port_of(session: JudgmentSession) -> JudgmentPort:
            return session.port
        ```
    """

    @property
    def port(self) -> JudgmentPort:
        """Return the session's judgment port."""
        ...


def _provider_error(error: GenerationError) -> ProviderError:
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
        target = next(dst for src, dst in _ERROR_MAP if isinstance(error, src))
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


def _to_typevet(name: str, question: object, cap: int) -> Question:
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


def _to_judgevet(answer: Answer | object) -> JevAnswer:
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


def _threshold(keyword: float | None, options: Options) -> float | None:
    """Pick the off-option threshold from the keyword or the provider options.

    Args:
        keyword: The ``off_option_threshold`` keyword; it wins when set.
        options: judgevet ``provider_options``, or ``None`` for none.

    Returns:
        The keyword, else the option value, else ``None``.

    Raises:
        ProviderCapabilityError: The options carry a key other than
            ``off_option_threshold``.
        ProviderRequestError: The option value is not ``None`` or a number in
            ``[0, 1]``. The message does not show the value.
    """
    if set(options or {}) - {"off_option_threshold"}:
        raise ProviderCapabilityError("typevet reads only off_option_threshold")
    value = (options or {}).get("off_option_threshold")
    try:
        check_off_option_threshold(value)
    except DecisionExecutionError:
        msg = "provider option off_option_threshold must be None or in [0, 1]"
        raise ProviderRequestError(msg) from None
    if keyword is None and isinstance(value, (int, float)):
        return float(value)
    return keyword


def _to_response(response: JudgmentResponse, questions: Questions) -> SystemOneResponse:
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
    answers = {name: _to_judgevet(value) for name, value in response.answers.items()}
    receipts = {name: r.as_dict() for name, r in response.off_option.items()}
    return SystemOneResponse(response.model, usage, answers, receipts)


class TypevetSystemOnePort:
    """judgevet ``SystemOnePort`` over a typevet ``JudgmentPort``, text only.

    The port has no media methods, so ``judge_with_images`` refuses it. It
    borrows ``judgment`` and never closes it.

    Attributes:
        bridge_capabilities (BridgeCapabilities): The declared capabilities.

    Examples:
        ```python
        port = TypevetSystemOnePort(judgment)
        response = port.system_one("text", {"q": Noul()}, "gemma")
        ```
    """

    def __init__(
        self,
        judgment: JudgmentPort,
        *,
        max_choice_options: int = DEFAULT_MAX_CHOICE_OPTIONS,
    ) -> None:
        """Wrap a typevet judgment port.

        Args:
            judgment: The typevet port that scores each question.
            max_choice_options: Most Choice options or Score levels accepted.

        Raises:
            ValueError: ``max_choice_options`` is below two.
        """
        if max_choice_options < _MIN_OPTIONS:
            raise ValueError("max_choice_options must be at least 2")
        self._judgment = judgment
        self._cap = max_choice_options

    @property
    def bridge_capabilities(self) -> BridgeCapabilities:
        """Return the declared capabilities.

        Returns:
            Logprobs required, the option cap and no media.
        """
        return BridgeCapabilities(True, self._cap, None)

    def system_one(
        self,
        state: State,
        questions: Questions,
        model: str,
        *,
        off_option_threshold: float | None = None,
        provider_options: Options = None,
    ) -> SystemOneResponse:
        """Judge ``state`` against every question through typevet.

        Args:
            state: The content to judge: text, a JSON object or an array.
            questions: judgevet questions or raw wire mappings, keyed by name.
            model: The model id typevet sends to its backend.
            off_option_threshold: Forwarded to the typevet port, or ``None``
                to read ``provider_options`` instead.
            provider_options: judgevet provider options. The bridge reads only
                ``off_option_threshold``, and only when the keyword is ``None``.

        Returns:
            Typed answers, the model id the backend reports, token usage and
            the off-option receipt of each answer in ``receipts``.

        Raises:
            ProviderRequestError: A question is malformed, the option threshold
                is invalid, or typevet rejects the request.
            ProviderCapabilityError: The request needs a capability typevet
                lacks, or ``provider_options`` has another key.
            ProviderTransportError: The backend could not be reached.
            ProviderResponseError: The backend answer broke the typed contract.
        """
        threshold = _threshold(off_option_threshold, provider_options)
        return self._judge(state, questions, model, None, threshold)

    def _judge(
        self,
        state: State,
        questions: Questions,
        model: str,
        media: tuple[ImageInput, ...] | None,
        threshold: float | None,
    ) -> SystemOneResponse:
        """Convert, call typevet, map failures and convert the answers back.

        Args:
            state: The content to judge.
            questions: judgevet questions keyed by name.
            model: The model id.
            media: Images for every question, or ``None`` for text.
            threshold: The typevet off-option guard, or ``None``.

        Returns:
            The judgevet response.

        Raises:
            ProviderError: A mapped typevet failure or a refused request.
        """
        converted = {n: _to_typevet(n, q, self._cap) for n, q in questions.items()}
        try:
            response = self._judgment.judge(
                state, converted, model, media=media, off_option_threshold=threshold
            )
        except GenerationError as error:
            raise _provider_error(error) from error
        return _to_response(response, questions)


def _image_groups(
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


def _add_counts(counts: Sequence[int | None]) -> int | None:
    """Add token counts, keeping the total unknown when any count is unknown.

    Args:
        counts: One count per image group.

    Returns:
        The sum, or ``None`` when any count is ``None``.
    """
    return None if None in counts else sum(c or 0 for c in counts)


def _merge(responses: Sequence[SystemOneResponse]) -> SystemOneResponse:
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
        _add_counts([r.usage.input_tokens for r in responses]),
        _add_counts([r.usage.output_tokens for r in responses]),
    )
    answers = {k: v for r in responses for k, v in r.answers.items()}
    receipts = {k: v for r in responses for k, v in r.receipts.items()}
    return SystemOneResponse(responses[0].model, usage, answers, receipts)


class TypevetMediaSystemOnePort(TypevetSystemOnePort):
    """judgevet ``MediaSystemOnePort`` over a typevet ``JudgmentPort``.

    Use it only with a backend that accepts images, such as a native Gemma
    vision session. The caller declares the media capabilities.

    Attributes:
        bridge_capabilities (BridgeCapabilities): The declared capabilities.

    Examples:
        ```python
        port = TypevetMediaSystemOnePort(
            session.port, media=MediaCapabilities({"image/png"})
        )
        ```
    """

    def __init__(
        self,
        judgment: JudgmentPort,
        *,
        media: MediaCapabilities,
        max_choice_options: int = DEFAULT_MAX_CHOICE_OPTIONS,
    ) -> None:
        """Wrap a media-capable typevet judgment port.

        Args:
            judgment: The typevet port that scores each question.
            media: The declared image types and ceilings.
            max_choice_options: Most Choice options or Score levels accepted.

        Raises:
            ValueError: ``media`` declares a type typevet does not support.
        """
        super().__init__(judgment, max_choice_options=max_choice_options)
        if not media.image_types <= SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError("media declares an image type typevet does not support")
        self._media = media

    @property
    def bridge_capabilities(self) -> BridgeCapabilities:
        """Return the declared capabilities.

        Returns:
            Logprobs required, the option cap and the declared media.
        """
        return BridgeCapabilities(True, self._cap, self._media)

    def capabilities(self, model: str) -> MediaCapabilities:
        """Return the declared media capabilities, with no IO.

        Args:
            model: The caller-selected model; the declaration does not vary.

        Returns:
            The media capabilities passed at construction.
        """
        return self._media

    def system_one_media(
        self,
        state: State,
        questions: Questions,
        model: str,
        *,
        evidence: ImageEvidence,
        off_option_threshold: float | None = None,
        provider_options: Options = None,
    ) -> SystemOneResponse:
        """Judge ``state`` and the images against every question.

        Args:
            state: The content to judge.
            questions: judgevet questions or raw wire mappings, keyed by name.
            model: The model id typevet sends to its backend.
            evidence: The images and the images each question is bound to.
            off_option_threshold: Forwarded to the typevet port for each
                image group, or ``None`` to read ``provider_options`` instead.
            provider_options: judgevet provider options. The bridge reads only
                ``off_option_threshold``, and only when the keyword is ``None``.

        Returns:
            Typed answers, the model id the backend reports, summed usage and
            the off-option receipt of each answer in ``receipts``.

        Raises:
            ProviderCapabilityError: An image type is outside typevet's
                supported set, the request needs a capability typevet lacks,
                or ``provider_options`` has another key.
            ProviderResponseError: The image groups report different models.
            ProviderError: A mapped typevet failure or a refused request.
        """
        threshold = _threshold(off_option_threshold, provider_options)
        groups = _image_groups(questions, evidence)
        return _merge([self._judge(state, g, model, m, threshold) for g, m in groups])


class AsyncTypevetSystemOnePort:
    """judgevet ``AsyncSystemOnePort`` that runs the sync bridge in a thread.

    typevet judgment is synchronous, so each call runs on a worker thread and
    the event loop stays free.

    Attributes:
        bridge_capabilities (BridgeCapabilities): The wrapped port's capabilities.

    Examples:
        ```python
        port = AsyncTypevetSystemOnePort(TypevetSystemOnePort(judgment))
        response = await port.system_one("text", {"q": Noul()}, "gemma")
        ```
    """

    def __init__(self, port: TypevetSystemOnePort) -> None:
        """Wrap a sync bridge port.

        Args:
            port: The bridge port each call runs.
        """
        self._port = port

    @property
    def bridge_capabilities(self) -> BridgeCapabilities:
        """Return the wrapped port's declared capabilities.

        Returns:
            The capabilities of the sync port.
        """
        return self._port.bridge_capabilities

    async def system_one(
        self,
        state: State,
        questions: Questions,
        model: str,
        *,
        off_option_threshold: float | None = None,
    ) -> SystemOneResponse:
        """Judge ``state`` on a worker thread.

        Args:
            state: The content to judge.
            questions: judgevet questions or raw wire mappings, keyed by name.
            model: The model id typevet sends to its backend.
            off_option_threshold: Forwarded to the sync port.

        Returns:
            The sync port's response.

        Raises:
            ProviderError: The sync port's mapped failure.
        """
        return await asyncio.to_thread(
            self._port.system_one,
            state,
            questions,
            model,
            off_option_threshold=off_option_threshold,
        )


def provider_factory(
    open_session: Callable[[], AbstractContextManager[JudgmentSession]],
    *,
    media: MediaCapabilities | None = None,
    max_choice_options: int = DEFAULT_MAX_CHOICE_OPTIONS,
) -> ProviderFactory:
    """Build a judgevet ``ProviderFactory`` that owns one typevet session.

    Each call opens a new session and closes it when the provider scope exits.
    Pass it to ``create_cli_app(provider_factory=...)`` or to the MCP entry
    point ``main(provider_factory=...)``.

    Args:
        open_session: Opens a typevet session, for example ``open_judgment``.
        media: Declared media capabilities, or ``None`` for a text-only port.
        max_choice_options: Most Choice options or Score levels accepted.

    Returns:
        A factory whose context yields a bridge port over the session's port.

    Examples:
        ```python
        factory = provider_factory(open_judgment)
        with factory() as port:
            port.system_one("text", {"q": Noul()}, "gemma")
        ```
    """

    @contextmanager
    def scope() -> Iterator[SystemOnePort]:
        """Open the session and yield a bridge port over its judgment port.

        Yields:
            The bridge port; the session closes when the scope exits.
        """
        with open_session() as session:
            if media is None:
                yield TypevetSystemOnePort(
                    session.port, max_choice_options=max_choice_options
                )
            else:
                yield TypevetMediaSystemOnePort(
                    session.port, media=media, max_choice_options=max_choice_options
                )

    return scope
