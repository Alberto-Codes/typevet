"""Map httpx and vLLM HTTP outcomes to domain generation errors.

A status of 400 or above becomes ``BackendHttpError`` with a bounded body
snippet. An httpx client failure becomes ``TransportError``. A body that is
not JSON becomes ``GenerationError``. Messages name vLLM. This module never
copies request headers into an error. The snippet is the first 500
characters of the server's own response body, so it holds
whatever the server returns.

The request hook of ``gateway_headers`` puts the request id that it sends in
``request.extensions`` under ``REQUEST_ID_EXTENSION`` of
[typevet.adapters.outbound.request_ids][] (#356, #411).
``BackendHttpError`` and ``TransportError`` get that id as ``request_id``,
and ``post_json_traced`` returns it with the reply. Without a configured
request-id header, the id is ``None``. No message holds the id.

Each ``BackendHttpError`` also gets the retry hints of the response (#355).
``Retry-After`` as delta-seconds gives that number. An HTTP-date gives the
seconds from the response ``Date`` header to that date. Without a valid
``Date``, the count starts at the current UTC time. A past date gives 0. Any
other value, more than one ``Retry-After`` field, or a wait above
``MAX_RETRY_AFTER_SECONDS`` (one year) gives ``None``. The
``x-ratelimit-*`` and ``ratelimit-*`` headers are copied with lowercase names
and verbatim values. A name that the request sent, or ``Authorization``, is
never copied. This module makes no retry.

Attributes:
    RATE_LIMIT_PREFIXES (tuple[str, ...]): Lowercase rate-limit name prefixes.
    MAX_RETRY_AFTER_SECONDS (float): Largest wait kept; one year.

Examples:
    ```python
    import httpx

    from typevet.adapters.outbound.vllm.http_mapping import post_json

    with httpx.Client() as client:
        payload = post_json(client, "http://127.0.0.1:8000/v1/models", {})
    ```

See Also:
    - [typevet.adapters.outbound.vllm.scoring][]: Scoring adapter consumer
    - [typevet.adapters.outbound.http_errors][]: Shared status and snippet limits
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: llama.cpp counterpart
    - [typevet.domain.errors][]: Transport and backend error types
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final

import httpx

from typevet.adapters.outbound.http_errors import HTTP_ERROR_STATUS, body_snippet
from typevet.adapters.outbound.request_ids import request_id_of
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError


def map_transport_error(exc: httpx.HTTPError) -> TransportError:
    """Build a transport error from an httpx client failure.

    Args:
        exc: HTTP client exception from POST or connection setup.

    Returns:
        A ``TransportError`` with no status or body snippet. Its
        ``request_id`` is the id of the failed request, or ``None``.
    """
    try:
        request: httpx.Request | None = exc.request
    except RuntimeError:
        request = None
    request_id = None if request is None else request_id_of(request)
    return TransportError(f"vLLM request failed: {exc}", request_id=request_id)


RATE_LIMIT_PREFIXES: Final[tuple[str, ...]] = ("x-ratelimit-", "ratelimit-")
MAX_RETRY_AFTER_SECONDS: Final[float] = 31_536_000.0
_DELTA_SECONDS: Final = re.compile(r"[0-9]+")


def _http_date(value: str) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def retry_after_seconds(
    headers: httpx.Headers, now: datetime | None = None
) -> float | None:
    """Return the wait in seconds that ``Retry-After`` asks for.

    Args:
        headers: Response headers.
        now: Reference time when ``Date`` is absent or invalid; default is
            the current UTC time.

    Returns:
        The delta-seconds value, or the non-negative seconds from ``Date``
        (else ``now``) to the HTTP-date. ``None`` when the header is absent,
        repeated, neither form, or above ``MAX_RETRY_AFTER_SECONDS``.
    """
    values = headers.get_list("retry-after")
    if len(values) != 1:
        return None
    value = values[0].strip()
    if _DELTA_SECONDS.fullmatch(value):
        seconds = float(value)
    else:
        target = _http_date(value)
        if target is None:
            return None
        reference = _http_date(headers.get("date", "")) or now or datetime.now(UTC)
        seconds = max(0.0, (target - reference).total_seconds())
    return seconds if seconds <= MAX_RETRY_AFTER_SECONDS else None


def rate_limit_headers(response: httpx.Response) -> dict[str, str]:
    """Return the rate-limit headers of ``response``, never a sent header.

    Args:
        response: Response whose ``request`` is set.

    Returns:
        Lowercase names that start with a ``RATE_LIMIT_PREFIXES`` item, with
        verbatim values. Names that the request sent and ``authorization``
        are left out.
    """
    sent = {name.lower() for name in response.request.headers} | {"authorization"}
    return {
        name: value
        for name, value in response.headers.items()
        if name.startswith(RATE_LIMIT_PREFIXES) and name not in sent
    }


def backend_http_error(
    response: httpx.Response, message: str, snippet: str
) -> BackendHttpError:
    """Build a ``BackendHttpError`` with the retry hints of ``response``.

    Args:
        response: Error or redirect response whose ``request`` is set.
        message: Error message; it holds no header value.
        snippet: Body snippet for the error, empty when withheld.

    Returns:
        The error with ``retry_after_seconds``, ``rate_limit`` and
        ``request_id`` set.
    """
    return BackendHttpError(
        message,
        status_code=response.status_code,
        body_snippet=snippet,
        retry_after_seconds=retry_after_seconds(response.headers),
        rate_limit=rate_limit_headers(response),
        request_id=request_id_of(response.request),
    )


def map_http_status(response: httpx.Response) -> BackendHttpError:
    """Build a backend HTTP error from a non-success status.

    Args:
        response: vLLM response with status at or above ``HTTP_ERROR_STATUS``.

    Returns:
        A ``BackendHttpError`` carrying status, a bounded body snippet and
        the retry hints.
    """
    snippet = body_snippet(response.text)
    message = f"vLLM HTTP {response.status_code}: {snippet}"
    return backend_http_error(response, message, snippet)


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
    return post_json_traced(client, url, body)[0]


def post_json_traced(
    client: httpx.Client, url: str, body: dict[str, Any]
) -> tuple[Any, str | None]:
    """POST like ``post_json`` and also return the request id that was sent.

    Args:
        client: Open HTTP client.
        url: Absolute endpoint URL.
        body: JSON request body.

    Returns:
        The parsed JSON value and the request id, or ``None`` when no
        request-id header is configured.

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
    return parse_json_response(response), request_id_of(response.request)
