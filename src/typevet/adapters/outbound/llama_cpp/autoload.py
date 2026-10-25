"""Send a llama.cpp router request without evicting another model (#461).

The llama.cpp router (``tools/server/server.cpp``) loads the model that a
request names before it proxies the request. When ``--models-max`` slots are
full, it unloads another model first. A request with ``?autoload=false`` does
not load: the router answers HTTP 400 ``model is not loaded`` instead. The
router proxies ``/v1/chat/completions``, ``/completion``, ``/props?model=``,
``/apply-template`` and ``/tokenize``. It serves ``GET /models`` and the
router ``GET /props`` itself.

``send_with_autoload`` and ``send_with_autoload_async`` take a ``send``
callable that sends one request with extra query parameters. With
``autoload`` on, they send it once with no extra parameter. With ``autoload``
off, they send it with ``autoload=false``. After a 400 ``model is not
loaded`` they read ``GET /models`` once. When no other model has the status
``loaded``, ``loading`` or ``sleeping``, or the router ``/props`` reports
``max_instances`` 0 (no limit) or above that count, they send the request once
more with ``autoload=true``. Otherwise they raise ``BackendBusyError``. A read
that fails also raises ``BackendBusyError``, so the adapter never loads a
model it cannot prove has a free slot.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp.autoload import send_with_autoload

    response = send_with_autoload(
        lambda params: client.post(url, json=body, params=params),
        client=client,
        base_url="http://127.0.0.1:8090/",
        model="gemma-4-31b-kv9-text",
        autoload=False,
    )
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: Status and JSON
      mapping, and the early-close retry for the reads
    - [typevet.domain.errors][]: ``BackendBusyError``
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.http_errors import body_snippet
from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    map_transport_error,
    parse_json_response,
    send_idempotent,
)
from typevet.domain.errors import BackendBusyError, GenerationError

__all__ = [
    "RUNNING_STATUSES",
    "is_not_loaded",
    "post_with_autoload",
    "send_with_autoload",
    "send_with_autoload_async",
]

RUNNING_STATUSES = frozenset({"loaded", "loading", "sleeping"})
"""Router ``status.value`` entries that hold a model slot (``is_running``)."""

_BAD_REQUEST = 400
_NOT_LOADED = "model is not loaded"
_OFF: Mapping[str, str] = {"autoload": "false"}
_ON: Mapping[str, str] = {"autoload": "true"}
_NONE: Mapping[str, str] = {}

Send = Callable[[Mapping[str, str]], httpx.Response]
AsyncSend = Callable[[Mapping[str, str]], Awaitable[httpx.Response]]


def is_not_loaded(response: httpx.Response) -> bool:
    """Return whether ``response`` is the router 400 ``model is not loaded``.

    Args:
        response: Router response with any status.

    Returns:
        ``True`` for status 400 whose JSON ``error.message``, or whose body
        text when the body is not JSON, is ``model is not loaded`` in any case.
    """
    if response.status_code != _BAD_REQUEST:
        return False
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return _NOT_LOADED in response.text.casefold()
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    return isinstance(message, str) and message.strip().casefold() == _NOT_LOADED


def _busy(response: httpx.Response) -> BackendBusyError:
    """Build the busy error from the router 400 response.

    Args:
        response: The 400 ``model is not loaded`` response.

    Returns:
        A ``BackendBusyError`` with the status and a body snippet.
    """
    return BackendBusyError(
        f"llama.cpp HTTP {response.status_code}: {_NOT_LOADED}",
        status_code=response.status_code,
        body_snippet=body_snippet(response.text),
    )


def _running_others(payload: Any, model: str) -> int:
    """Count the models other than ``model`` that hold a router slot.

    Args:
        payload: Parsed ``GET /models`` body.
        model: The model id the request named.

    Returns:
        The count of other entries whose ``status.value`` is running.

    Raises:
        GenerationError: When the body has no ``data`` list of objects.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        msg = "llama.cpp /models response missing a data list of objects"
        raise GenerationError(msg)
    count = 0
    for entry in data:
        status = entry.get("status")
        value = status.get("value") if isinstance(status, dict) else None
        if entry.get("id") != model and value in RUNNING_STATUSES:
            count += 1
    return count


def _has_free_slot(payload: Any, running: int) -> bool:
    """Return whether the router ``/props`` reports a slot beyond ``running``.

    Args:
        payload: Parsed router ``GET /props`` body.
        running: Count of other running models.

    Returns:
        ``True`` when ``max_instances`` is 0 (no limit) or above ``running``.

    Raises:
        GenerationError: When ``max_instances`` is not a non-negative integer.
    """
    limit = payload.get("max_instances") if isinstance(payload, dict) else None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        msg = "llama.cpp router /props response missing max_instances"
        raise GenerationError(msg)
    return limit == 0 or limit > running


def _read(client: httpx.Client, base_url: str, path: str) -> Any:
    """Read one router-local JSON route, with the early-close retry.

    Returns:
        The parsed JSON body.
    """
    url = urljoin(base_url, path)
    response = send_idempotent(lambda: client.get(url))
    ensure_success_status(response)
    return parse_json_response(response)


async def _read_async(client: httpx.AsyncClient, base_url: str, path: str) -> Any:
    """Read one router-local JSON route with the async client.

    Returns:
        The parsed JSON body.

    Raises:
        TransportError: When the HTTP client fails before a response.
    """
    try:
        response = await client.get(urljoin(base_url, path))
    except httpx.HTTPError as exc:
        raise map_transport_error(exc) from exc
    ensure_success_status(response)
    return parse_json_response(response)


def send_with_autoload(
    send: Send,
    *,
    client: httpx.Client,
    base_url: str,
    model: str,
    autoload: bool,
) -> httpx.Response:
    """Send one model request and load the model only into a free slot.

    Args:
        send: Sends the request once with the given extra query parameters.
        client: Client for the ``/models`` and router ``/props`` reads.
        base_url: Router root with a trailing slash.
        model: The model id that the request names.
        autoload: When ``True``, send once with no extra parameter.

    Returns:
        The response of the first request, or of the one retry.

    Raises:
        BackendBusyError: When another model holds the last router slot, when
            a read fails, or when the retry also answers ``model is not
            loaded``.
    """
    if autoload:
        return send(_NONE)
    response = send(_OFF)
    if not is_not_loaded(response):
        return response
    try:
        running = _running_others(_read(client, base_url, "models"), model)
        free = running == 0 or _has_free_slot(_read(client, base_url, "props"), running)
    except GenerationError as exc:
        raise _busy(response) from exc
    if not free:
        raise _busy(response)
    retry = send(_ON)
    if is_not_loaded(retry):
        raise _busy(retry)
    return retry


def post_with_autoload(
    client: httpx.Client,
    path: str,
    body: Mapping[str, Any],
    *,
    autoload: bool,
) -> httpx.Response:
    """POST ``body`` to ``path`` on the client base URL with the autoload rules.

    The POST changes no server state, so an early close gets one retry (#305).
    ``body["model"]`` names the model.

    Args:
        client: Client with the router base URL.
        path: Router path, for example ``/apply-template`` or ``/tokenize``.
        body: JSON body that holds the ``model`` id.
        autoload: When ``True``, send once with no extra parameter.

    Returns:
        The response of the first request, or of the one retry.
    """
    return send_with_autoload(
        lambda params: send_idempotent(
            lambda: client.post(path, json=body, params=params)
        ),
        client=client,
        base_url=str(client.base_url),
        model=str(body["model"]),
        autoload=autoload,
    )


async def send_with_autoload_async(
    send: AsyncSend,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    autoload: bool,
) -> httpx.Response:
    """Async form of ``send_with_autoload`` with the same rules.

    Args:
        send: Sends the request once with the given extra query parameters.
        client: Async client for the ``/models`` and router ``/props`` reads.
        base_url: Router root with a trailing slash.
        model: The model id that the request names.
        autoload: When ``True``, send once with no extra parameter.

    Returns:
        The response of the first request, or of the one retry.

    Raises:
        BackendBusyError: When another model holds the last router slot, when
            a read fails, or when the retry also answers ``model is not
            loaded``.
    """
    if autoload:
        return await send(_NONE)
    response = await send(_OFF)
    if not is_not_loaded(response):
        return response
    try:
        running = _running_others(await _read_async(client, base_url, "models"), model)
        free = running == 0 or _has_free_slot(
            await _read_async(client, base_url, "props"), running
        )
    except GenerationError as exc:
        raise _busy(response) from exc
    if not free:
        raise _busy(response)
    retry = await send(_ON)
    if is_not_loaded(retry):
        raise _busy(retry)
    return retry
