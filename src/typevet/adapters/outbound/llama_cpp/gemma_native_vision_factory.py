"""Gemma native-turn vision judgment factory over llama.cpp ([#174][i174], [#196][i196]).

Examples:
    ```python
    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
        open_gemma_native_vision_judgment,
    )

    settings = load_llama_settings()
    with open_gemma_native_vision_judgment(settings=settings) as session:
        session.port.judge("Hello", questions, session.model)
    ```

The factory pins ``session.model`` on ``session.port``; other model ids fail
before tokenization. Optional ``tokenize_content`` and ``scoring_port_wrapper``
hooks support consumer dispatch ledgers without importing ``typevet_evals``.
An optional ``text_parts`` keyword passes caller templates to the scoring
adapter ([#373][i373]).
The default ``/tokenize`` hook maps an httpx failure to ``TransportError`` and
a status of 400 or above to ``BackendHttpError`` ([#298][i298]). A 200 body
that is not JSON, or that has no ``tokens`` list of integers, raises
``GenerationError`` ([#310][i310]). When the router closes the connection
(new or reused) before a response head, the hook sends the request once more ([#305][i305]).
The ``/apply-template`` probe uses the same mapping and retry. A 200 body that
is not JSON, or that has no ``prompt`` string, raises ``GenerationError``
([#311][i311]).
When Gemma 4 is required and the probe renders another family, the
``ValueError`` names the model id and what chose it: the ``model`` argument or
``settings.multimodal_model`` (``TYPEVET_LLAMA__MULTIMODAL_MODEL``) ([#425][i425]).
The port sets ``JudgmentResponse.request_ids`` from the request id of each
question's ``/completion`` request when the client stamps one ([#411][i411]).
With ``require_vision=False`` the factory and the probe accept a model whose
``/props`` reports text-only input. That session records
``capability.vision is False``, and its port raises
``ScoringUnsupportedCapabilityError`` for an image judgment before any request
([#438][i438]). The default stays ``True``, so vision smokes still fail fast on
a text-only model.

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: ``ScoringJudgmentAdapter``
    - [typevet.adapters.outbound.request_ids][]: Request id per question
    - [typevet.runtime][]: public re-exports for library callers
    - [docs.how-to.connect-gemma4-native-vision-judgment][]: Composition how-to

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i196]: https://github.com/Alberto-Codes/typevet/issues/196
[i298]: https://github.com/Alberto-Codes/typevet/issues/298
[i305]: https://github.com/Alberto-Codes/typevet/issues/305
[i310]: https://github.com/Alberto-Codes/typevet/issues/310
[i311]: https://github.com/Alberto-Codes/typevet/issues/311
[i373]: https://github.com/Alberto-Codes/typevet/issues/373
[i411]: https://github.com/Alberto-Codes/typevet/issues/411
[i425]: https://github.com/Alberto-Codes/typevet/issues/425
[i438]: https://github.com/Alberto-Codes/typevet/issues/438
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, Unpack

import httpx

from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    parse_json_response,
    send_idempotent,
)
from typevet.adapters.outbound.llama_cpp.multimodal import (
    MediaCapability,
    fetch_media_capability,
)
from typevet.adapters.outbound.llama_cpp.scoring import (
    DEFAULT_N_VOCAB,
    LlamaCppCandidateScoringAdapter,
)
from typevet.adapters.outbound.request_ids import (
    RequestIdJudgmentPort,
    RequestIdScoringPort,
)
from typevet.domain.errors import GenerationError, ScoringUnsupportedCapabilityError
from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.domain.text_parts import TextParts
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort


class GemmaVisionSettings(Protocol):
    """Router connection options the factory reads from the composition root.

    Examples:
        ```python
        from typevet.adapters.inbound.settings import load_llama_settings

        settings = load_llama_settings()
        assert settings.base_url
        ```
    """

    @property
    def base_url(self) -> str:
        """Router base URL."""
        ...

    @property
    def timeout(self) -> float:
        """HTTP timeout in seconds."""
        ...

    @property
    def multimodal_model(self) -> str:
        """Default multimodal model id."""
        ...


_SETTINGS_SOURCE = "settings.multimodal_model (TYPEVET_LLAMA__MULTIMODAL_MODEL)"
_SUPPORTED_NATIVE = frozenset(
    {
        ServedTemplateClass.NATIVE_GEMMA3_TURN,
        ServedTemplateClass.NATIVE_GEMMA4_TURN,
    }
)


@dataclass(frozen=True, slots=True)
class GemmaNativeVisionSession:
    """Live router session with a configured ``JudgmentPort``.

    Attributes:
        port (JudgmentPort): Scoring-backed judgment for native media turns.
        client (httpx.Client): Shared HTTP client for the router base URL.
        model (str): Model id probed for template and vision capability.
        served (ServedTemplateClass): Classified native template family.
        capability (MediaCapability): Router media capability for ``model``.

    Examples:
        ```python
        from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
            GemmaNativeVisionSession,
        )

        assert GemmaNativeVisionSession.__dataclass_fields__
        ```
    """

    port: JudgmentPort
    client: httpx.Client
    model: str
    served: ServedTemplateClass
    capability: MediaCapability


def _classify_native_template(
    client: httpx.Client,
    model: str,
    *,
    require_gemma4: bool,
    source: str,
) -> ServedTemplateClass:
    """Render one probe turn through ``/apply-template`` and classify it.

    The probe changes no server state, so an early close gets one retry. A
    Gemma 4 mismatch names ``model`` and the ``source`` that chose it (#425).

    Returns:
        The served template family.

    Raises:
        ValueError: When the family is unsupported or not the one required.
        TransportError: When the probe fails before a response.
        BackendHttpError: When the probe returns status 400 or higher.
        GenerationError: When the ``/apply-template`` body has no ``prompt``
            string.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "add_generation_prompt": True,
    }
    response = send_idempotent(lambda: client.post("/apply-template", json=body))
    ensure_success_status(response)
    family = classify_served_template(_rendered_prompt(parse_json_response(response)))
    if require_gemma4 and family is not ServedTemplateClass.NATIVE_GEMMA4_TURN:
        msg = (
            f"expected NATIVE_GEMMA4_TURN, got {family.value} for model {model!r}"
            f" from {source}; name a Gemma 4 model whose template renders <|turn>"
        )
        raise ValueError(msg)
    if family not in _SUPPORTED_NATIVE:
        msg = f"unsupported served template for native vision: {family.value}"
        raise ValueError(msg)
    return family


def _check_vision(capability: MediaCapability, *, require_vision: bool) -> None:
    """Refuse a text-only model when the caller requires image input.

    Args:
        capability: Media capability that ``/props`` reports for the model.
        require_vision: When true, a model without image input is refused.

    Raises:
        ValueError: When ``require_vision`` is true and the model is text-only.
    """
    if require_vision and not capability.vision:
        msg = "model reports text-only input modalities"
        raise ValueError(msg)


class _TextOnlyJudgmentPort:
    """Judgment port that refuses images for a text-only model (#438).

    Attributes:
        _inner (JudgmentPort): Port that serves text judgments.
        _model (str): Text-only model id, named in the refusal.

    Examples:
        ```python
        port = _TextOnlyJudgmentPort(inner, "gemma-4-31b-kv9-text")
        port.judge("Hello", questions, "gemma-4-31b-kv9-text")
        ```
    """

    def __init__(self, inner: JudgmentPort, model: str) -> None:
        self._inner = inner
        self._model = model

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Refuse images, else forward the text judgment to the inner port.

        Args:
            state: Content under evaluation.
            questions: Question names to typed questions or wire dictionaries.
            model: Backend model id.
            media: Images; a non-empty tuple is refused.
            off_option_threshold: Forwarded to the inner port.

        Returns:
            The inner port's response.

        Raises:
            ScoringUnsupportedCapabilityError: When ``media`` holds an image.
        """
        if media:
            msg = (
                f"llama.cpp model {self._model!r} reports text-only input; "
                "an image judgment is refused"
            )
            raise ScoringUnsupportedCapabilityError(msg)
        return self._inner.judge(
            state, questions, model, off_option_threshold=off_option_threshold
        )


def _model_source(model: str | None) -> str:
    """Name what chose the probed model id for error messages (#425).

    Args:
        model: The explicit ``model`` argument, or ``None``.

    Returns:
        ``"the model argument"`` for an explicit id, else
        ``settings.multimodal_model``, which ``load_llama_settings`` reads
        from ``TYPEVET_LLAMA__MULTIMODAL_MODEL``.
    """
    return "the model argument" if model else _SETTINGS_SOURCE


def _rendered_prompt(payload: Any) -> str:
    """Read the ``prompt`` string from a parsed ``/apply-template`` body.

    Args:
        payload: Parsed JSON body from the router.

    Returns:
        The rendered prompt text.

    Raises:
        GenerationError: When the body has no ``prompt`` string.
    """
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str):
        msg = "llama.cpp /apply-template response missing a prompt string"
        raise GenerationError(msg)
    return prompt


def _tokenize_factory(
    client: httpx.Client,
    model: str,
) -> Callable[[str], tuple[int, ...]]:
    def tokenize(text: str) -> tuple[int, ...]:
        """Tokenize ``text`` through the router ``/tokenize`` endpoint.

        An early close on a new or reused connection gets one retry (#305).

        Returns:
            Token id tuple from the router JSON body.

        Raises:
            TransportError: When the HTTP client fails before a response.
            BackendHttpError: When ``/tokenize`` returns status 400 or higher.
            GenerationError: When the body has no ``tokens`` list of integers.
        """
        body = {"model": model, "content": text, "add_special": False}
        response = send_idempotent(lambda: client.post("/tokenize", json=body))
        ensure_success_status(response)
        return _token_ids(parse_json_response(response))

    return tokenize


def _token_ids(payload: Any) -> tuple[int, ...]:
    """Read the ``tokens`` list from a parsed ``/tokenize`` body.

    Args:
        payload: Parsed JSON body from the router.

    Returns:
        Token ids in router order.

    Raises:
        GenerationError: When the body has no ``tokens`` list of integers.
    """
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        msg = "llama.cpp /tokenize response missing a tokens list of integers"
        raise GenerationError(msg)
    return tuple(tokens)


class _JudgmentHooks(TypedDict, total=False):
    """Optional keywords that ``open_gemma_native_vision_judgment`` wires in.

    Attributes:
        tokenize_content (Callable | None): Tokenizer hook; default router
            ``/tokenize``.
        scoring_port_wrapper (Callable | None): Wrapper applied before
            ``JudgmentPort`` wiring.
        text_parts (TextParts | None): Option block and context templates;
            default ``None`` (#373).
        require_vision (bool): When false, accept a text-only model; default
            ``True`` (#438).
    """

    tokenize_content: Callable[[str], Sequence[int]] | None
    scoring_port_wrapper: Callable[[CandidateScoringPort], CandidateScoringPort] | None
    text_parts: TextParts | None
    require_vision: bool


@contextmanager
def open_gemma_native_vision_judgment(
    *,
    settings: GemmaVisionSettings,
    model: str | None = None,
    require_gemma4: bool = True,
    n_vocab: int | None = DEFAULT_N_VOCAB,
    http_client: httpx.Client | None = None,
    **hooks: Unpack[_JudgmentHooks],
) -> Iterator[GemmaNativeVisionSession]:
    """Open a judgment port for Gemma native-turn vision on a llama.cpp router.

    Probes ``/apply-template`` and media capability before the first ``judge``
    call. Unsupported template families raise ``ValueError`` before scoring
    dispatch. A text-only model raises ``ValueError`` unless
    ``require_vision=False``; its port then refuses images (#438).

    Args:
        settings: Router connection options from the composition root.
        model: Model id; defaults to ``settings.multimodal_model``.
        require_gemma4: When true, require ``NATIVE_GEMMA4_TURN``.
        n_vocab: Model vocabulary size. The default is the Gemma 4 class
            size, so the session sends no ``/v1/models`` request. ``None``
            lets the scoring adapter read it from ``/v1/models`` (#321).
        http_client: Optional pre-built client (for tests); not closed on exit.

    Other Parameters:
        tokenize_content (Callable | None): Optional tokenizer hook; defaults
            to router ``/tokenize``.
        scoring_port_wrapper (Callable | None): Optional wrapper applied
            before ``JudgmentPort`` wiring.
        text_parts (TextParts | None): Option block and context templates, or
            ``None`` for the defaults (#373).
        require_vision (bool): When ``False``, accept a text-only model. The
            port then raises ``ScoringUnsupportedCapabilityError`` for an
            image judgment before any request (#438). Default ``True``.

    Yields:
        A session holding the configured ``JudgmentPort`` and probe metadata.
        Its responses map each question name to the ``/completion`` request
        id in ``request_ids``, or keep it empty when no id was sent.

    Raises:
        ValueError: When vision is required and unavailable, or the template is
            unsupported. A Gemma 4 mismatch names the model id and whether the ``model``
            argument or ``settings.multimodal_model`` chose it.
        TransportError: When a probe fails before a response.
        BackendHttpError: When a probe returns status 400 or higher.
        GenerationError: When the ``/apply-template`` body has no ``prompt``
            string.
    """
    model_id = model or settings.multimodal_model
    source = _model_source(model)
    base = settings.base_url.rstrip("/")
    tokenize_content = hooks.get("tokenize_content")
    scoring_port_wrapper = hooks.get("scoring_port_wrapper")

    def _session(client: httpx.Client) -> Iterator[GemmaNativeVisionSession]:
        capability = fetch_media_capability(client, f"{base}/", model_id)
        _check_vision(capability, require_vision=hooks.get("require_vision", True))
        served = _classify_native_template(
            client, model_id, require_gemma4=require_gemma4, source=source
        )
        tokenize = tokenize_content or _tokenize_factory(client, model_id)
        scoring = LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=settings.timeout,
            client=client,
            n_vocab=n_vocab,
            media_capabilities={model_id: capability},
        )
        scoring_port: CandidateScoringPort = scoring
        if scoring_port_wrapper is not None:
            scoring_port = scoring_port_wrapper(scoring)
        adapter = ScoringJudgmentAdapter(
            RequestIdScoringPort(scoring_port),
            tokenize_content=tokenize,
            served_template=served,
            pinned_model=model_id,
            text_parts=hooks.get("text_parts"),
        )
        port: JudgmentPort = RequestIdJudgmentPort(adapter)
        if not capability.vision:
            port = _TextOnlyJudgmentPort(port, model_id)
        try:
            yield GemmaNativeVisionSession(
                port=port,
                client=client,
                model=model_id,
                served=served,
                capability=capability,
            )
        finally:
            scoring.close()

    if http_client is not None:
        yield from _session(http_client)
        return
    with httpx.Client(base_url=base, timeout=settings.timeout) as owned:
        yield from _session(owned)


def probe_gemma_native_vision_support(
    *,
    settings: GemmaVisionSettings,
    model: str | None = None,
    require_gemma4: bool = True,
    require_vision: bool = True,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return probe metadata without constructing a long-lived judgment port.

    Args:
        settings: Router connection options.
        model: Model id; defaults to ``settings.multimodal_model``.
        require_gemma4: When true, require ``NATIVE_GEMMA4_TURN``.
        require_vision: When true (the default), refuse a text-only model, as
            vision smokes need. When false, report ``vision`` as ``False``
            for a text-only model (#438).
        http_client: Optional pre-built client (for tests).

    Returns:
        Mapping with ``ok``, ``model``, ``served``, and ``vision`` keys.

    Raises:
        ValueError: When the router rejects the probe (same rules as open,
            including the model id and its source in a Gemma 4 mismatch).
        TransportError: When the probe fails before a response.
        BackendHttpError: When the probe returns status 400 or higher.
        GenerationError: When the ``/apply-template`` body has no ``prompt``
            string.
    """
    model_id = model or settings.multimodal_model
    source = _model_source(model)
    base = settings.base_url.rstrip("/")

    def _probe(client: httpx.Client) -> dict[str, Any]:
        capability = fetch_media_capability(client, f"{base}/", model_id)
        _check_vision(capability, require_vision=require_vision)
        served = _classify_native_template(
            client, model_id, require_gemma4=require_gemma4, source=source
        )
        return {
            "ok": True,
            "model": model_id,
            "served": served.value,
            "vision": capability.vision,
        }

    if http_client is not None:
        return _probe(http_client)
    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        return _probe(client)
