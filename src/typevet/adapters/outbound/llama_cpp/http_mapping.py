"""Map httpx and llama.cpp HTTP outcomes to domain generation errors.

``send_idempotent`` sends one idempotent scoring request. When the server
closes the connection before a complete status line and headers, it sends the request once
more on a fresh connection ([#305][i305]).

The module also exports ``HTTP_ERROR_STATUS``, ``BODY_SNIPPET_MAX`` and
``body_snippet`` from ``http_errors`` for callers that import them here.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp.http_mapping import (
        ensure_success_status,
        map_transport_error,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.generation][]: Sync adapter consumer
    - [typevet.adapters.outbound.http_errors][]: Shared status and snippet limits
    - [typevet.domain.errors][]: Transport and backend error types

[i305]: https://github.com/Alberto-Codes/typevet/issues/305
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from typevet.adapters.outbound.http_errors import (
    BODY_SNIPPET_MAX,
    HTTP_ERROR_STATUS,
    body_snippet,
)
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError

__all__ = [
    "BODY_SNIPPET_MAX",
    "HTTP_ERROR_STATUS",
    "body_snippet",
    "ensure_success_status",
    "map_http_status",
    "map_transport_error",
    "parse_json_response",
    "send_idempotent",
]

# httpcore raises RemoteProtocolError with this text when the peer closes the
# connection after it reads the request and before it writes a status line.
_CLOSED_BEFORE_RESPONSE = "Server disconnected without sending a response"


def _closed_before_response(exc: httpx.RemoteProtocolError) -> bool:
    """Return whether ``exc`` is a close before a complete response head.

    Args:
        exc: Protocol error from the httpx client.

    Returns:
        ``True`` only for the no-response disconnect. A malformed response
        gives ``False``.
    """
    return str(exc).startswith(_CLOSED_BEFORE_RESPONSE)


def send_idempotent(send: Callable[[], httpx.Response]) -> httpx.Response:
    """Send an idempotent request, with one retry after an early close.

    A llama.cpp router can close a reused keep-alive connection after it
    reads the request and before it writes a response ([#305][i305]).
    httpx then drops that connection, so the retry opens a fresh one. Use
    this helper only for requests that change no server state
    (``/tokenize``, ``/props``, ``/completion`` with ``n_predict: 0``).

    Args:
        send: Callable that sends the request once and returns the response.

    Returns:
        The response from the first or the retried request.

    Raises:
        TransportError: When a request fails before a response. A second
            early close, a timeout and a malformed response do not retry.
    """
    try:
        try:
            return send()
        except httpx.RemoteProtocolError as exc:
            if not _closed_before_response(exc):
                raise
        return send()
    except httpx.HTTPError as exc:
        raise map_transport_error(exc) from exc


def map_transport_error(exc: httpx.HTTPError) -> TransportError:
    """Build a transport error from an httpx client failure.

    Args:
        exc: HTTP client exception from POST or connection setup.

    Returns:
        A ``TransportError`` with no status or body snippet.
    """
    return TransportError(f"llama.cpp request failed: {exc}")


def map_http_status(response: httpx.Response) -> BackendHttpError:
    """Build a backend HTTP error from a non-success status.

    Args:
        response: llama.cpp response with status at or above ``HTTP_ERROR_STATUS``.

    Returns:
        A ``BackendHttpError`` carrying status and a body snippet.
    """
    snippet = body_snippet(response.text)
    return BackendHttpError(
        f"llama.cpp HTTP {response.status_code}: {snippet}",
        status_code=response.status_code,
        body_snippet=snippet,
    )


def ensure_success_status(response: httpx.Response) -> None:
    """Raise when the response status indicates adapter failure.

    Args:
        response: Parsed HTTP response from llama.cpp.

    Raises:
        BackendHttpError: When ``response.status_code`` is at or above 400.
    """
    if response.status_code >= HTTP_ERROR_STATUS:
        raise map_http_status(response)


def parse_json_response(response: httpx.Response) -> Any:
    """Parse the HTTP body as JSON or raise ``GenerationError``.

    Args:
        response: Successful-status response from llama.cpp.

    Returns:
        Parsed JSON value.

    Raises:
        GenerationError: When the body is not valid JSON.
    """
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        msg = "llama.cpp returned non-JSON HTTP body"
        raise GenerationError(msg) from exc
