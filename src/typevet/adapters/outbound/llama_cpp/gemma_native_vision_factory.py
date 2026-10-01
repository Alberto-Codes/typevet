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

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: ``ScoringJudgmentAdapter``
    - [typevet.runtime][]: public re-exports for library callers
    - [docs.how-to.connect-gemma4-native-vision-judgment][]: Composition how-to

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i196]: https://github.com/Alberto-Codes/typevet/issues/196
[i298]: https://github.com/Alberto-Codes/typevet/issues/298
[i305]: https://github.com/Alberto-Codes/typevet/issues/305
[i310]: https://github.com/Alberto-Codes/typevet/issues/310
[i311]: https://github.com/Alberto-Codes/typevet/issues/311
[i373]: https://github.com/Alberto-Codes/typevet/issues/373
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
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
from typevet.domain.errors import GenerationError
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
) -> ServedTemplateClass:
    """Render one probe turn through ``/apply-template`` and classify it.

    The probe changes no server state, so an early close gets one retry.

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
        msg = f"expected NATIVE_GEMMA4_TURN, got {family.value}"
        raise ValueError(msg)
    if family not in _SUPPORTED_NATIVE:
        msg = f"unsupported served template for native vision: {family.value}"
        raise ValueError(msg)
    return family


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
    """

    tokenize_content: Callable[[str], Sequence[int]] | None
    scoring_port_wrapper: Callable[[CandidateScoringPort], CandidateScoringPort] | None
    text_parts: TextParts | None


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
    call. Unsupported template families or text-only models raise ``ValueError``
    before scoring dispatch.

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

    Yields:
        A session holding the configured ``JudgmentPort`` and probe metadata.

    Raises:
        ValueError: When vision is unavailable or the template is unsupported.
        TransportError: When a probe fails before a response.
        BackendHttpError: When a probe returns status 400 or higher.
        GenerationError: When the ``/apply-template`` body has no ``prompt``
            string.
    """
    model_id = model or settings.multimodal_model
    base = settings.base_url.rstrip("/")
    tokenize_content = hooks.get("tokenize_content")
    scoring_port_wrapper = hooks.get("scoring_port_wrapper")

    def _session(client: httpx.Client) -> Iterator[GemmaNativeVisionSession]:
        capability = fetch_media_capability(client, f"{base}/", model_id)
        if not capability.vision:
            msg = "model reports text-only input modalities"
            raise ValueError(msg)
        served = _classify_native_template(
            client, model_id, require_gemma4=require_gemma4
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
        port: JudgmentPort = ScoringJudgmentAdapter(
            scoring_port,
            tokenize_content=tokenize,
            served_template=served,
            pinned_model=model_id,
            text_parts=hooks.get("text_parts"),
        )
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
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Return probe metadata without constructing a long-lived judgment port.

    Args:
        settings: Router connection options.
        model: Model id; defaults to ``settings.multimodal_model``.
        require_gemma4: When true, require ``NATIVE_GEMMA4_TURN``.
        http_client: Optional pre-built client (for tests).

    Returns:
        Mapping with ``ok``, ``model``, ``served``, and ``vision`` keys.

    Raises:
        ValueError: When the router rejects the probe (same rules as open).
        TransportError: When the probe fails before a response.
        BackendHttpError: When the probe returns status 400 or higher.
        GenerationError: When the ``/apply-template`` body has no ``prompt``
            string.
    """
    model_id = model or settings.multimodal_model
    base = settings.base_url.rstrip("/")

    def _probe(client: httpx.Client) -> dict[str, Any]:
        capability = fetch_media_capability(client, f"{base}/", model_id)
        if not capability.vision:
            msg = "model reports text-only input modalities"
            raise ValueError(msg)
        served = _classify_native_template(
            client, model_id, require_gemma4=require_gemma4
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
