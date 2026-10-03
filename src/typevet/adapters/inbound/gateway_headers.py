"""Header rules and HTTP hooks for a vLLM server behind an API gateway (#331).

The header rules live in [typevet.adapters.inbound.http_headers][], which
both backends share (#410). This module re-exports them, so callers that
import ``check_gateway_headers`` or a ``MAX_*`` limit from here still work.

The hooks go on each client that the composition root builds. The request
hook sets a fresh UUID4 hex value in the request-id header when the request
does not have one. It records the sent value in ``request.extensions``, so
results and errors can carry it (#356); typevet never logs it. The response
hook refuses each redirect, because the
clients do not follow redirects, and withholds an HTML error body. Neither
error holds the ``Location`` header or the body. Both errors keep the
``Retry-After`` wait and the rate-limit headers (#355).

Attributes:
    MAX_HEADER_FIELDS (int): Largest number of extra headers.
    MAX_NAME_BYTES (int): Largest header name, in bytes.
    MAX_VALUE_BYTES (int): Largest header value, in bytes.
    MAX_TOTAL_BYTES (int): Largest sum of extra header names and values.
    PROTECTED_HEADER_NAMES (frozenset[str]): Lowercase names never configured.

Examples:
    ```python
    from typevet.adapters.inbound.gateway_headers import check_gateway_headers

    check_gateway_headers(
        {"X-Tenant": "acme"}, auth_header="X-API-Key", field="headers"
    )
    ```

See Also:
    - [typevet.adapters.inbound.http_headers][]: Shared header rules
    - [typevet.adapters.inbound.backend_settings][]: ``VllmSettings`` and clients
    - [typevet.domain.errors][]: ``BackendHttpError``
    - [typevet.adapters.outbound.vllm.http_mapping][]: ``backend_http_error``
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Final

import httpx

from typevet.adapters.inbound.http_headers import (
    MAX_HEADER_FIELDS,
    MAX_NAME_BYTES,
    MAX_TOTAL_BYTES,
    MAX_VALUE_BYTES,
    PROTECTED_HEADER_NAMES,
    check_auth_scheme,
    check_gateway_headers,
    check_header_name,
    parse_headers_json,
)
from typevet.adapters.outbound.vllm.http_mapping import (
    REQUEST_ID_EXTENSION,
    backend_http_error,
)

__all__ = [
    "MAX_HEADER_FIELDS",
    "MAX_NAME_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_VALUE_BYTES",
    "PROTECTED_HEADER_NAMES",
    "async_event_hooks",
    "check_auth_scheme",
    "check_gateway_headers",
    "check_header_name",
    "parse_headers_json",
    "sync_event_hooks",
]

_REDIRECT: Final = range(300, 400)
_ERROR_STATUS: Final[int] = 400


def _stamp(request: httpx.Request, name: str | None) -> None:
    if name is None:
        return
    if name not in request.headers:
        request.headers[name] = uuid.uuid4().hex
    request.extensions[REQUEST_ID_EXTENSION] = request.headers[name]


def _guard(response: httpx.Response) -> None:
    status = response.status_code
    if status in _REDIRECT:
        msg = f"vLLM HTTP {status}: redirect not followed"
        raise backend_http_error(response, msg, "")
    content_type = response.headers.get("content-type", "").lower()
    if status >= _ERROR_STATUS and (
        "html" in content_type or response.text.lstrip().startswith("<")
    ):
        msg = f"vLLM HTTP {status}: HTML body withheld"
        raise backend_http_error(response, msg, "")


def sync_event_hooks(
    request_id_header: str | None,
) -> dict[str, list[Callable[[Any], Any]]]:
    """Return the ``httpx.Client`` hooks for the request id and responses.

    Args:
        request_id_header: Header that gets a fresh UUID4 hex, or ``None``.

    Returns:
        ``event_hooks`` for ``httpx.Client``.
    """

    def stamp(request: httpx.Request) -> None:
        """Set the request-id header when it is configured and absent.

        The sent value is recorded in ``request.extensions``.

        Args:
            request: Outgoing request.
        """
        _stamp(request, request_id_header)

    def guard(response: httpx.Response) -> None:
        """Read a redirect or error body, then apply the response rules.

        Args:
            response: Received response.

        Raises:
            BackendHttpError: On a redirect or an HTML error body.
        """
        if response.status_code in _REDIRECT or response.status_code >= _ERROR_STATUS:
            response.read()
        _guard(response)

    return {"request": [stamp], "response": [guard]}


def async_event_hooks(
    request_id_header: str | None,
) -> dict[str, list[Callable[[Any], Awaitable[None]]]]:
    """Return the ``httpx.AsyncClient`` hooks for the request id and responses.

    Args:
        request_id_header: Header that gets a fresh UUID4 hex, or ``None``.

    Returns:
        ``event_hooks`` for ``httpx.AsyncClient``.
    """

    async def stamp(request: httpx.Request) -> None:
        """Set the request-id header when it is configured and absent.

        The sent value is recorded in ``request.extensions``.

        Args:
            request: Outgoing request.
        """
        _stamp(request, request_id_header)

    async def guard(response: httpx.Response) -> None:
        """Read a redirect or error body, then apply the response rules.

        Args:
            response: Received response.

        Raises:
            BackendHttpError: On a redirect or an HTML error body.
        """
        if response.status_code in _REDIRECT or response.status_code >= _ERROR_STATUS:
            await response.aread()
        _guard(response)

    return {"request": [stamp], "response": [guard]}
