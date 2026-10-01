"""Header rules and HTTP hooks for a vLLM server behind an API gateway (#331).

``VllmSettings`` calls these checks when it is built. A header name must be
an HTTP token of at most ``MAX_NAME_BYTES`` bytes and must not be protected.
The protected names are the hop-by-hop headers, ``Host``, ``Content-Length``,
the headers that the clients set (``Content-Type``, ``Accept``,
``Accept-Encoding`` and ``User-Agent``, which only ``user_agent`` sets) and
the configured auth header, compared without case (#348). A value holds only
the ASCII characters 0x20 to 0x7E and at most ``MAX_VALUE_BYTES`` bytes. At
most ``MAX_HEADER_FIELDS`` headers are allowed, and their names and values
together hold at most ``MAX_TOTAL_BYTES`` bytes. Values are literal: typevet
does not expand ``$NAME`` or run ``!command``. Each error names the field and
never holds a value.

The hooks go on each client that the composition root builds. The request
hook sets a fresh UUID4 hex value in the request-id header when the request
does not have one. The response hook refuses each redirect, because the
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
    - [typevet.adapters.inbound.backend_settings][]: ``VllmSettings`` and clients
    - [typevet.domain.errors][]: ``BackendHttpError``
    - [typevet.adapters.outbound.vllm.http_mapping][]: ``backend_http_error``
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final

import httpx

from typevet.adapters.outbound.vllm.http_mapping import backend_http_error

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

MAX_HEADER_FIELDS: Final[int] = 32
MAX_NAME_BYTES: Final[int] = 128
MAX_VALUE_BYTES: Final[int] = 2048
MAX_TOTAL_BYTES: Final[int] = 8192
PROTECTED_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "content-type",
        "accept",
        "accept-encoding",
        "user-agent",
    }
)
_TOKEN: Final = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_VALUE: Final = re.compile(r"[\x20-\x7e]*")
_REDIRECT: Final = range(300, 400)
_ERROR_STATUS: Final[int] = 400


def check_header_name(name: object, field: str) -> str:
    """Return ``name`` when it is a permitted header name.

    Args:
        name: Candidate header name.
        field: Field label for the error, for example ``auth_header``.

    Returns:
        ``name`` unchanged.

    Raises:
        ValueError: When ``name`` is not an HTTP token, is longer than
            ``MAX_NAME_BYTES`` or is protected. The message names ``field``.
    """
    if not isinstance(name, str) or not _TOKEN.fullmatch(name):
        msg = f"{field} holds a header name that is not an HTTP token"
        raise ValueError(msg)
    if len(name) > MAX_NAME_BYTES:
        msg = f"{field} holds a header name longer than {MAX_NAME_BYTES} bytes"
        raise ValueError(msg)
    if name.lower() in PROTECTED_HEADER_NAMES:
        msg = f"{field} names the protected header {name}"
        raise ValueError(msg)
    return name


def check_auth_scheme(scheme: object, field: str) -> None:
    """Refuse an auth scheme that is neither empty nor an HTTP token.

    Args:
        scheme: Candidate scheme, for example ``Bearer``.
        field: Field label for the error.

    Raises:
        ValueError: When ``scheme`` is not empty and not a token.
    """
    if not isinstance(scheme, str) or (scheme and not _TOKEN.fullmatch(scheme)):
        msg = f"{field} must be empty or an HTTP token"
        raise ValueError(msg)


def _check_value(value: object, label: str) -> int:
    if not isinstance(value, str) or not _VALUE.fullmatch(value):
        msg = f"{label} value must hold only ASCII characters 0x20 to 0x7E"
        raise ValueError(msg)
    if len(value) > MAX_VALUE_BYTES:
        msg = f"{label} value is longer than {MAX_VALUE_BYTES} bytes"
        raise ValueError(msg)
    return len(value)


def check_gateway_headers(
    headers: Mapping[str, str], *, auth_header: str, field: str
) -> None:
    """Refuse extra headers that break a gateway header rule.

    Args:
        headers: Extra header names and literal values.
        auth_header: Configured auth header name; it is protected too.
        field: Field label for the error.

    Raises:
        ValueError: When a rule fails. The message names ``field`` and, for a
            valid token, the header name, but never a value.
    """
    if len(headers) > MAX_HEADER_FIELDS:
        msg = f"{field} holds more than {MAX_HEADER_FIELDS} headers"
        raise ValueError(msg)
    seen: set[str] = set()
    total = 0
    for name, value in headers.items():
        check_header_name(name, field)
        lowered = name.lower()
        if lowered == auth_header.lower():
            msg = f"{field} names the auth header {name}; set the key instead"
            raise ValueError(msg)
        if lowered in seen:
            msg = f"{field} names the header {name} twice"
            raise ValueError(msg)
        seen.add(lowered)
        total += len(name) + _check_value(value, f"{field} header {name}")
    if total > MAX_TOTAL_BYTES:
        msg = f"{field} names and values hold more than {MAX_TOTAL_BYTES} bytes"
        raise ValueError(msg)


def parse_headers_json(raw: str, field: str) -> dict[str, str]:
    """Parse a JSON object of string header values.

    Args:
        raw: JSON text; blank text gives an empty mapping.
        field: Variable name for the error.

    Returns:
        The header names and literal values.

    Raises:
        ValueError: When ``raw`` is not a JSON object of strings. The error
            has no cause or context and does not hold ``raw``.
    """
    if not raw.strip():
        return {}
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if not isinstance(parsed, dict) or not all(
        isinstance(value, str) for value in parsed.values()
    ):
        msg = f"{field} must be a JSON object of string values"
        raise ValueError(msg) from None
    return parsed


def _stamp(request: httpx.Request, name: str | None) -> None:
    if name is not None and name not in request.headers:
        request.headers[name] = uuid.uuid4().hex


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
