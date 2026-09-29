"""Async vLLM ``/v1/chat/completions`` adapter with a per-adapter POST limit.

The request body comes from ``vllm_generation.generation_body``, the builder
that ``VllmGenerationAdapter`` also uses. The error mapping and the result
validation are the same as for the sync adapter. An ``asyncio.Semaphore`` limits the number of
POSTs that one adapter has in flight; the default of 1 sends requests one at a
time. The limit holds per event loop: when a call runs on a new loop, the
adapter makes a new semaphore for that loop, and it keeps a weak reference to
one loop only. The semaphore does not move the HTTP client. An httpx
``AsyncClient`` with a connection pool binds to the first loop that uses it,
and a call from a later loop can raise ``RuntimeError: Event loop is closed``.
Only an injected client whose transport works on any loop, such as
``httpx.MockTransport``, can serve a second ``asyncio.run``. Otherwise build
one adapter per event loop. A caller that cancels ``generate`` gets
``asyncio.CancelledError`` unchanged, and the slot is released for the next
queued call. An httpx timeout, like every other ``httpx.HTTPError``, becomes
``TransportError``. Other exceptions from the client, such as that
``RuntimeError``, propagate unchanged.
The adapter makes one POST per call. It does not retry and does not fall back
to unconstrained generation.

Examples:
    ```python
    from typevet.adapters.outbound.vllm_generation_async import (
        AsyncVllmGenerationAdapter,
    )

    async with AsyncVllmGenerationAdapter(max_concurrency=2) as port:
        pass  # await port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.vllm_generation][]: Sync adapter and body builder
    - [typevet.adapters.outbound.chat_completion][]: Content and schema checks
    - [typevet.adapters.outbound.async_llama_cpp][]: llama.cpp async counterpart
    - [typevet.adapters.outbound.vllm_http][]: Shared vLLM HTTP error mapping
    - [typevet.ports.async_generation][]: AsyncGenerationPort
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.chat_completion import extract_content, validated_value
from typevet.adapters.outbound.vllm_generation import generation_body
from typevet.adapters.outbound.vllm_http import (
    ensure_success_status,
    map_transport_error,
    parse_json_response,
)
from typevet.domain.models import GenerationRequest, GenerationResult


class AsyncVllmGenerationAdapter:
    """Call a vLLM server asynchronously with ``structured_outputs`` constraints.

    Attributes:
        _base_url (str): Server root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.AsyncClient | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.
        _max_concurrency (int): Maximum POSTs in flight for this adapter.
        _slots (asyncio.Semaphore): Limit on POSTs in flight on the current loop.
        _slots_loop (weakref.ref[asyncio.AbstractEventLoop] | None): Loop that
            ``_slots`` serves, or ``None`` before the first call.

    Examples:
        ```python
        from typevet.adapters.outbound.vllm_generation_async import (
            AsyncVllmGenerationAdapter,
        )

        AsyncVllmGenerationAdapter("http://127.0.0.1:8000", max_concurrency=2)
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
        max_concurrency: int = 1,
    ) -> None:
        """Create the adapter.

        Args:
            base_url: vLLM server root URL.
            timeout: Request timeout in seconds for an owned client.
            client: Optional caller-built httpx async client, for example one
                that carries authentication headers. The adapter does not
                close it.
            max_concurrency: Maximum POSTs in flight for this adapter on
                each event loop.

        Raises:
            ValueError: When ``max_concurrency`` is less than 1.
        """
        if max_concurrency < 1:
            msg = "max_concurrency must be at least 1"
            raise ValueError(msg)
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._max_concurrency = max_concurrency
        self._slots = asyncio.Semaphore(max_concurrency)
        self._slots_loop: weakref.ref[asyncio.AbstractEventLoop] | None = None

    async def close(self) -> None:
        """Close the owned HTTP client when the adapter created it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        """Enter a context that closes the owned client on exit."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close resources."""
        await self.close()

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """POST one constrained chat completion and validate the result.

        The call waits for a free slot of the running loop's semaphore before
        the POST and releases the slot when the POST ends, fails or is
        cancelled. ``generation_body`` builds the request, and the
        ``chat_completion`` helpers read the reply and check it against the
        schema.

        Args:
            request: Prompt, schema, served model name and optional images.

        Returns:
            Validated structured value.

        Raises:
            TransportError: When the HTTP client raises an
                ``httpx.HTTPError`` before a response, including a timeout.
            BackendHttpError: When vLLM returns HTTP status 400 or above.
            GenerationError: On a non-JSON body, bad shape or non-JSON content.
            SchemaValidationError: When the value is not an object, is
                non-finite or fails the schema.
        """
        schema_obj = dict(request.schema)
        body = generation_body(request, schema_obj)
        url = urljoin(self._base_url, "v1/chat/completions")
        client = self._ensure_client()
        async with self._loop_slots():
            payload = await _post_json(client, url, body)
        raw_text = extract_content(payload, "vLLM")
        value = validated_value(raw_text, schema_obj)
        return GenerationResult(value=value, model=request.model, raw_text=raw_text)

    def _loop_slots(self) -> asyncio.Semaphore:
        """Return the semaphore for the running event loop.

        An ``asyncio.Semaphore`` binds to one loop. When the running loop is
        not the loop that ``_slots`` serves, a new semaphore replaces it, so
        the adapter keeps one semaphore and a weak reference to one loop.

        Returns:
            The semaphore that limits POSTs on the running loop.
        """
        loop = asyncio.get_running_loop()
        owner = None if self._slots_loop is None else self._slots_loop()
        if owner is not loop:
            if self._slots_loop is not None:
                self._slots = asyncio.Semaphore(self._max_concurrency)
            self._slots_loop = weakref.ref(loop)
        return self._slots

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.AsyncClient``.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client


async def _post_json(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> Any:
    """POST a JSON body to vLLM once and return the parsed JSON reply.

    ``asyncio.CancelledError`` is not an httpx error, so it propagates
    unchanged.

    Args:
        client: Open async HTTP client.
        url: Absolute endpoint URL.
        body: JSON request body.

    Returns:
        Parsed JSON value from a successful response.

    Raises:
        TransportError: When the httpx client raises an ``httpx.HTTPError``,
            including a timeout. Other exceptions propagate unchanged.
    """
    try:
        response = await client.post(url, json=body)
    except httpx.HTTPError as exc:
        raise map_transport_error(exc) from exc
    ensure_success_status(response)
    return parse_json_response(response)
