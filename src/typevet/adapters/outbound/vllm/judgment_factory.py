"""vLLM judgment factory over chat-completions scoring ([#169][i169]).

The vLLM server applies the served chat template, so the factory needs no
template probe. It does not call ``/props``, ``/apply-template`` or any
other probe. Prefixes are plain chat content from ``ChatContentFraming``.
Control strings are tokenized through vLLM ``/tokenize`` with
``add_special_tokens: false``, as vLLM v0.30.0
``vllm/entrypoints/serve/tokenize/protocol.py`` defines the
``TokenizeCompletionRequest`` and ``TokenizeResponse`` shapes.

``VllmCandidateScoringAdapter`` joins its own base URL into an absolute URL,
so the factory passes one base URL to the scoring adapter and to the
``/tokenize`` hook. Both requests go to the same host.

Examples:
    ```python
    import httpx

    from typevet.adapters.outbound.vllm.judgment_factory import open_vllm_judgment

    with (
        httpx.Client(base_url="http://127.0.0.1:8000") as client,
        open_vllm_judgment(client=client, model="served-model") as session,
    ):
        session.port.judge("Hello", questions, session.model)
    ```

See Also:
    - [typevet.adapters.outbound.vllm.scoring][]: Scoring adapter and framing
    - [typevet.adapters.outbound.judgment_scoring][]: ``ScoringJudgmentAdapter``
    - [typevet.adapters.outbound.vllm.http_mapping][]: Error mapping
    - [typevet.runtime][]: Public re-exports for library callers

[i169]: https://github.com/Alberto-Codes/typevet/issues/169
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.vllm.http_mapping import post_json
from typevet.adapters.outbound.vllm.scoring import (
    ChatContentFraming,
    VllmCandidateScoringAdapter,
)
from typevet.domain.errors import GenerationError
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort


@dataclass(frozen=True, slots=True)
class VllmJudgmentSession:
    """Open vLLM session with a configured ``JudgmentPort``.

    Attributes:
        port (JudgmentPort): Scoring-backed judgment pinned to ``model``.
        client (httpx.Client): HTTP client the session sends requests on.
        model (str): Served model name that ``port`` accepts.

    Examples:
        ```python
        from typevet.adapters.outbound.vllm.judgment_factory import (
            VllmJudgmentSession,
        )

        assert VllmJudgmentSession.__dataclass_fields__
        ```
    """

    port: JudgmentPort
    client: httpx.Client
    model: str


def _resolve_base_url(client: httpx.Client, base_url: str | None) -> str:
    """Return the server root without a trailing slash.

    Args:
        client: HTTP client whose ``base_url`` is the fallback.
        base_url: Explicit server root, or ``None``.

    Returns:
        ``base_url`` when given, else the client ``base_url``.

    Raises:
        ValueError: When neither gives a server root.
    """
    resolved = (str(client.base_url) if base_url is None else base_url).rstrip("/")
    if not resolved:
        msg = "base_url is required when the client has no base_url"
        raise ValueError(msg)
    return resolved


def _tokens(payload: object) -> tuple[int, ...]:
    """Read the ``tokens`` list from a vLLM ``/tokenize`` reply.

    Args:
        payload: Parsed JSON body.

    Returns:
        The token ids in order.

    Raises:
        GenerationError: When the body has no list of integer ``tokens``.
    """
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        msg = "vLLM /tokenize response must hold a list of integer tokens"
        raise GenerationError(msg)
    return tuple(tokens)


def vllm_tokenize(
    client: httpx.Client,
    model: str,
    *,
    base_url: str | None = None,
) -> Callable[[str], tuple[int, ...]]:
    """Return a hook that tokenizes text through vLLM ``/tokenize``.

    Args:
        client: HTTP client for the vLLM server.
        model: Served model name sent in each request.
        base_url: Server root. Defaults to the client ``base_url``.

    Returns:
        A function that POSTs ``{"model", "prompt", "add_special_tokens":
        false}`` and returns the reply ``tokens`` as a tuple.

    Raises:
        ValueError: When no server root is known.
    """
    url = f"{_resolve_base_url(client, base_url)}/tokenize"

    def tokenize(text: str) -> tuple[int, ...]:
        """Tokenize ``text`` without special tokens.

        Args:
            text: Control string to tokenize.

        Returns:
            Token id tuple from the vLLM reply.

        Raises:
            TransportError: When the HTTP client fails.
            BackendHttpError: When vLLM returns status 400 or above.
            GenerationError: When the body is not JSON or has no ``tokens``.
        """
        body = {"model": model, "prompt": text, "add_special_tokens": False}
        return _tokens(post_json(client, url, body))

    return tokenize


@contextmanager
def open_vllm_judgment(
    *,
    client: httpx.Client,
    model: str,
    base_url: str | None = None,
    tokenize_content: Callable[[str], Sequence[int]] | None = None,
    scoring_port_wrapper: Callable[[CandidateScoringPort], CandidateScoringPort]
    | None = None,
) -> Iterator[VllmJudgmentSession]:
    """Open a judgment port for a vLLM chat-completions server.

    The factory makes no HTTP call before the first ``judge`` call. The port
    rejects other model ids before tokenization or scoring IO. The caller
    owns ``client``; the factory does not close it.

    Args:
        client: HTTP client for the vLLM server, for example one that carries
            an ``Authorization`` header.
        model: Served model name; the port is pinned to it.
        base_url: Server root for scoring and ``/tokenize``. Defaults to the
            client ``base_url``.
        tokenize_content: Optional tokenizer hook; defaults to ``vllm_tokenize``.
        scoring_port_wrapper: Optional wrapper applied before ``JudgmentPort``
            wiring.

    Yields:
        A session holding the configured ``JudgmentPort``.

    Raises:
        ValueError: When no server root is known.
    """
    base = _resolve_base_url(client, base_url)
    tokenize = tokenize_content or vllm_tokenize(client, model, base_url=base)
    scoring = VllmCandidateScoringAdapter(base, client=client)
    scoring_port: CandidateScoringPort = scoring
    if scoring_port_wrapper is not None:
        scoring_port = scoring_port_wrapper(scoring)
    port = ScoringJudgmentAdapter(
        scoring_port,
        framing=ChatContentFraming(),
        tokenize_content=tokenize,
        pinned_model=model,
    )
    try:
        yield VllmJudgmentSession(port=port, client=client, model=model)
    finally:
        scoring.close()
