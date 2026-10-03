"""Header rules and client headers shared by the vLLM and llama.cpp settings.

``VllmSettings`` and ``LlamaSettings`` call these checks when they are built
(#331, #410). A header name must be an HTTP token of at most
``MAX_NAME_BYTES`` bytes and must not be protected. The protected names are
the hop-by-hop headers, ``Host``, ``Content-Length``, the headers that the
clients set (``Content-Type``, ``Accept``, ``Accept-Encoding`` and
``User-Agent``, which only ``user_agent`` sets) and the configured auth
header, compared without case (#348). A value holds only the ASCII characters
0x20 to 0x7E and at most ``MAX_VALUE_BYTES`` bytes. At most
``MAX_HEADER_FIELDS`` headers are allowed, and their names and values
together hold at most ``MAX_TOTAL_BYTES`` bytes. Values are literal: typevet
does not expand ``$NAME`` or run ``!command``. Each error names the field and
never holds a value.

``client_headers`` builds the default headers of a composition-root client
from the key, the auth header and scheme, the user agent and the extra
headers. This module imports no outbound code, so both backends can use it.

Attributes:
    MAX_HEADER_FIELDS (int): Largest number of extra headers.
    MAX_NAME_BYTES (int): Largest header name, in bytes.
    MAX_VALUE_BYTES (int): Largest header value, in bytes.
    MAX_TOTAL_BYTES (int): Largest sum of extra header names and values.
    PROTECTED_HEADER_NAMES (frozenset[str]): Lowercase names never configured.

Examples:
    ```python
    from typevet.adapters.inbound.http_headers import check_gateway_headers

    check_gateway_headers(
        {"X-Tenant": "acme"}, auth_header="X-API-Key", field="headers"
    )
    ```

See Also:
    - [typevet.adapters.inbound.gateway_headers][]: vLLM event hooks
    - [typevet.adapters.inbound.backend_settings][]: ``VllmSettings``
    - [typevet.adapters.inbound.settings][]: ``LlamaSettings``
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final, Protocol

__all__ = [
    "MAX_HEADER_FIELDS",
    "MAX_NAME_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_VALUE_BYTES",
    "PROTECTED_HEADER_NAMES",
    "HeaderSettings",
    "check_auth_scheme",
    "check_gateway_headers",
    "check_header_name",
    "client_headers",
    "parse_headers_json",
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


class HeaderSettings(Protocol):
    """Settings fields that ``client_headers`` reads.

    Examples:
        ```python
        from typevet.adapters.inbound.settings import LlamaSettings

        settings: HeaderSettings = LlamaSettings()
        ```
    """

    @property
    def api_key(self) -> str | None:
        """Key sent in ``auth_header``, or ``None``."""
        ...

    @property
    def auth_header(self) -> str:
        """Header that carries the key."""
        ...

    @property
    def auth_scheme(self) -> str:
        """Scheme before the key; empty sends the key bare."""
        ...

    @property
    def user_agent(self) -> str | None:
        """``User-Agent`` value, or ``None`` for the httpx default."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Extra headers with literal values."""
        ...


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


def client_headers(settings: HeaderSettings) -> dict[str, str]:
    """Return the default headers of a composition-root client.

    Args:
        settings: Checked settings that hold the key and header fields.

    Returns:
        The extra ``settings.headers``. When a key is set, the key in
        ``settings.auth_header`` after ``settings.auth_scheme``, which by
        default gives ``Authorization: Bearer <key>``; an empty scheme sends
        the key bare. When a user agent is set, it as ``User-Agent``.
    """
    headers = dict(settings.headers)
    if settings.api_key is not None:
        scheme = settings.auth_scheme
        key = settings.api_key
        headers[settings.auth_header] = f"{scheme} {key}" if scheme else key
    if settings.user_agent is not None:
        headers["User-Agent"] = settings.user_agent
    return headers
