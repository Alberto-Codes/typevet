"""judgevet provider bridge: judgevet's ``SystemOnePort`` over typevet judgment (#284).

judgevet calls into typevet through this module. This module and its pure
mapping sibling ``judgevet_mapping`` are the only typevet modules that import
judgevet. The bridge imports the sibling after its judgevet import check. No
package ``__init__`` imports either module, so a bare ``import typevet`` stays
free of judgevet. Install the ``typevet[judgevet]`` extra to use it.

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
  refused with ``ProviderCapabilityError``. An invalid option value or
  keyword raises ``ProviderRequestError`` before any backend call; the
  message does not show the value. The option value is checked also when
  the keyword wins. ``AsyncTypevetSystemOnePort`` takes the same
  ``provider_options`` and forwards them to the sync port.
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
    - [typevet.adapters.inbound.judgevet_mapping][]: The conversions and
      the error map

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
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol

try:
    from judgevet import SystemOnePort, SystemOneResponse
    from judgevet.media import ImageEvidence, MediaCapabilities
    from judgevet.providers import ProviderFactory
except ModuleNotFoundError as missing:
    if (missing.name or "").split(".")[0] != "judgevet":
        raise
    msg = "typevet.adapters.inbound.judgevet needs judgevet; install typevet[judgevet]"
    raise ImportError(msg) from missing

from typevet.adapters.inbound.judgevet_mapping import (
    Options,
    Questions,
    State,
    image_groups,
    merge_responses,
    pick_threshold,
    provider_error,
    to_response,
    to_typevet,
)
from typevet.domain.decisions import MAX_ENUM_CHOICES
from typevet.domain.errors import GenerationError
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

_MIN_OPTIONS = 2
"""typevet needs two options to score a Choice or a Score."""


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
            ProviderRequestError: A question is malformed, the keyword or
                option threshold is invalid, or typevet rejects the request.
                A threshold error does not show the value.
            ProviderCapabilityError: The request needs a capability typevet
                lacks, or ``provider_options`` has another key.
            ProviderTransportError: The backend could not be reached.
            ProviderResponseError: The backend answer broke the typed contract.
        """
        threshold = pick_threshold(off_option_threshold, provider_options)
        return self._judge(state, questions, model, None, threshold)

    def _judge(
        self,
        state: State,
        questions: Questions,
        model: str,
        media: tuple[ImageInput, ...] | None,
        threshold: float | None,
    ) -> SystemOneResponse:
        """Convert, call typevet, map failures and convert back via the sibling.

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
        converted = {n: to_typevet(n, q, self._cap) for n, q in questions.items()}
        try:
            response = self._judgment.judge(
                state, converted, model, media=media, off_option_threshold=threshold
            )
        except GenerationError as error:
            raise provider_error(error) from error
        return to_response(response, questions)


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
            ProviderRequestError: The keyword or option threshold is
                invalid; the message does not show the value.
            ProviderResponseError: The image groups report different models.
            ProviderError: A mapped typevet failure or a refused request.
        """
        threshold = pick_threshold(off_option_threshold, provider_options)
        groups = image_groups(questions, evidence)
        return merge_responses(
            [self._judge(state, g, model, m, threshold) for g, m in groups]
        )


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
        provider_options: Options = None,
    ) -> SystemOneResponse:
        """Judge ``state`` on a worker thread.

        Args:
            state: The content to judge.
            questions: judgevet questions or raw wire mappings, keyed by name.
            model: The model id typevet sends to its backend.
            off_option_threshold: Forwarded to the sync port.
            provider_options: Forwarded to the sync port, which reads only
                ``off_option_threshold``, and only when the keyword is ``None``.

        Returns:
            The sync port's response, with the off-option receipts.

        Raises:
            ProviderError: The sync port's mapped failure or refused request.
        """
        return await asyncio.to_thread(
            self._port.system_one,
            state,
            questions,
            model,
            off_option_threshold=off_option_threshold,
            provider_options=provider_options,
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
