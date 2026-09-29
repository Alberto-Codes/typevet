"""Map httpx and llama.cpp HTTP outcomes to domain generation errors.

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
"""

from __future__ import annotations

import json
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
]


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
