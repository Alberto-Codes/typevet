"""Map httpx and vLLM HTTP outcomes to domain generation errors.

A status of 400 or above becomes ``BackendHttpError`` with a bounded body
snippet. An httpx client failure becomes ``TransportError``. A body that is
not JSON becomes ``GenerationError``. Messages name vLLM. This module never
copies request headers into an error. The snippet is the first 500
characters of the server's own response body, so it holds
whatever the server returns.

Examples:
    ```python
    import httpx

    from typevet.adapters.outbound.vllm_http import post_json

    with httpx.Client() as client:
        payload = post_json(client, "http://127.0.0.1:8000/v1/models", {})
    ```

See Also:
    - [typevet.adapters.outbound.vllm_scoring][]: Scoring adapter consumer
    - [typevet.adapters.outbound.http_errors][]: Shared status and snippet limits
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: llama.cpp counterpart
    - [typevet.domain.errors][]: Transport and backend error types
"""

from __future__ import annotations

from typing import Any

import httpx

from typevet.adapters.outbound.http_errors import HTTP_ERROR_STATUS, body_snippet
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError


def map_transport_error(exc: httpx.HTTPError) -> TransportError:
    """Build a transport error from an httpx client failure.

    Args:
        exc: HTTP client exception from POST or connection setup.

    Returns:
        A ``TransportError`` with no status or body snippet.
    """
    return TransportError(f"vLLM request failed: {exc}")


def map_http_status(response: httpx.Response) -> BackendHttpError:
    """Build a backend HTTP error from a non-success status.

    Args:
        response: vLLM response with status at or above ``HTTP_ERROR_STATUS``.

    Returns:
        A ``BackendHttpError`` carrying status and a bounded body snippet.
    """
    snippet = body_snippet(response.text)
    return BackendHttpError(
        f"vLLM HTTP {response.status_code}: {snippet}",
        status_code=response.status_code,
        body_snippet=snippet,
    )


def ensure_success_status(response: httpx.Response) -> None:
    """Raise when the response status indicates a vLLM failure.

    Args:
        response: HTTP response from vLLM.

    Raises:
        BackendHttpError: When ``response.status_code`` is at or above 400.
    """
    if response.status_code >= HTTP_ERROR_STATUS:
        raise map_http_status(response)


def parse_json_response(response: httpx.Response) -> Any:
    """Parse the HTTP body as JSON or raise ``GenerationError``.

    Args:
        response: Successful-status response from vLLM.

    Returns:
        Parsed JSON value.

    Raises:
        GenerationError: When the body is not valid JSON.
    """
    try:
        return response.json()
    except ValueError as exc:
        msg = "vLLM returned non-JSON HTTP body"
        raise GenerationError(msg) from exc


def post_json(client: httpx.Client, url: str, body: dict[str, Any]) -> Any:
    """POST a JSON body to vLLM once and return the parsed JSON reply.

    No retry is made.

    Args:
        client: Open HTTP client.
        url: Absolute endpoint URL.
        body: JSON request body.

    Returns:
        Parsed JSON value from a successful response.

    Raises:
        TransportError: When the httpx client fails.
        BackendHttpError: When the status is 400 or above.
        GenerationError: When the body is not valid JSON.
    """
    try:
        response = client.post(url, json=body)
    except httpx.HTTPError as exc:
        raise map_transport_error(exc) from exc
    ensure_success_status(response)
    return parse_json_response(response)
