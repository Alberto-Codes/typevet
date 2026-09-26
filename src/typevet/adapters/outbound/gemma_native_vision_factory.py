"""Gemma native-turn vision judgment factory over llama.cpp ([#174][i174], [#196][i196]).

Examples:
    ```python
    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet.adapters.outbound.gemma_native_vision_factory import (
        open_gemma_native_vision_judgment,
    )

    settings = load_llama_settings()
    with open_gemma_native_vision_judgment(settings=settings) as session:
        session.port.judge("Hello", questions, session.model)
    ```

The factory pins ``session.model`` on ``session.port``; other model ids fail
before tokenization. Optional ``tokenize_content`` and ``scoring_port_wrapper``
hooks support consumer dispatch ledgers without importing ``typevet.evaluation``.

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: ``ScoringJudgmentAdapter``
    - [typevet.runtime][]: public re-exports for library callers
    - [docs.how-to.connect-gemma4-native-vision-judgment][]: Composition how-to

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i196]: https://github.com/Alberto-Codes/typevet/issues/196
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp_multimodal import (
    MediaCapability,
    fetch_media_capability,
)
from typevet.adapters.outbound.llama_cpp_scoring import (
    DEFAULT_N_VOCAB,
    LlamaCppCandidateScoringAdapter,
)
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
        from typevet.adapters.outbound.gemma_native_vision_factory import (
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
    rendered = (
        client.post(
            "/apply-template",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
        )
        .raise_for_status()
        .json()["prompt"]
    )
    family = classify_served_template(rendered)
    if require_gemma4 and family is not ServedTemplateClass.NATIVE_GEMMA4_TURN:
        msg = f"expected NATIVE_GEMMA4_TURN, got {family.value}"
        raise ValueError(msg)
    if family not in _SUPPORTED_NATIVE:
        msg = f"unsupported served template for native vision: {family.value}"
        raise ValueError(msg)
    return family


def _tokenize_factory(
    client: httpx.Client,
    model: str,
) -> Callable[[str], tuple[int, ...]]:
    def tokenize(text: str) -> tuple[int, ...]:
        """Tokenize ``text`` through the router ``/tokenize`` endpoint.

        Returns:
            Token id tuple from the router JSON body.
        """
        body = client.post(
            "/tokenize",
            json={"model": model, "content": text, "add_special": False},
        )
        return tuple(body.raise_for_status().json()["tokens"])

    return tokenize


@contextmanager
def open_gemma_native_vision_judgment(
    *,
    settings: GemmaVisionSettings,
    model: str | None = None,
    require_gemma4: bool = True,
    n_vocab: int = DEFAULT_N_VOCAB,
    http_client: httpx.Client | None = None,
    tokenize_content: Callable[[str], Sequence[int]] | None = None,
    scoring_port_wrapper: Callable[[CandidateScoringPort], CandidateScoringPort]
    | None = None,
) -> Iterator[GemmaNativeVisionSession]:
    """Open a judgment port for Gemma native-turn vision on a llama.cpp router.

    Probes ``/apply-template`` and media capability before the first ``judge``
    call. Unsupported template families or text-only models raise ``ValueError``
    before scoring dispatch.

    Args:
        settings: Router connection options from the composition root.
        model: Model id; defaults to ``settings.multimodal_model``.
        require_gemma4: When true, require ``NATIVE_GEMMA4_TURN``.
        n_vocab: ``n_probs`` budget for Gemma 4 class vocab scoring.
        http_client: Optional pre-built client (for tests); not closed on exit.
        tokenize_content: Optional tokenizer hook; defaults to router ``/tokenize``.
        scoring_port_wrapper: Optional wrapper applied before ``JudgmentPort`` wiring.

    Yields:
        A session holding the configured ``JudgmentPort`` and probe metadata.

    Raises:
        ValueError: When vision is unavailable or the template is unsupported.
    """
    model_id = model or settings.multimodal_model
    base = settings.base_url.rstrip("/")

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
        )
        scoring_port: CandidateScoringPort = scoring
        if scoring_port_wrapper is not None:
            scoring_port = scoring_port_wrapper(scoring)
        port: JudgmentPort = ScoringJudgmentAdapter(
            scoring_port,
            tokenize_content=tokenize,
            served_template=served,
            pinned_model=model_id,
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
