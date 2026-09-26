"""Map httpx and llama.cpp HTTP outcomes to domain generation errors.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp_http import (
        ensure_success_status,
        map_transport_error,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: Sync adapter consumer
    - [typevet.domain.errors][]: Transport and backend error types
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from typevet.domain.errors import BackendHttpError, GenerationError, TransportError

HTTP_ERROR_STATUS = 400
BODY_SNIPPET_MAX = 500


def body_snippet(text: str) -> str:
    """Return a bounded snippet of response text for error metadata.

    Args:
        text: Full HTTP response body text.

    Returns:
        At most ``BODY_SNIPPET_MAX`` characters from ``text``.
    """
    return text[:BODY_SNIPPET_MAX]


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
