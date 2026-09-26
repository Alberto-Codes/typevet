"""Async llama.cpp OpenAI-compat adapter for constrained JSON Schema generation.

Examples:
    ```python
    from typevet.adapters.outbound.async_llama_cpp import AsyncLlamaCppGenerationAdapter

    async with AsyncLlamaCppGenerationAdapter() as port:
        pass  # await port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: Sync adapter
    - [typevet.adapters.outbound.async_fake][]: Offline fake for tests
"""

from __future__ import annotations

import json
from typing import Any, Self
from urllib.parse import urljoin

import httpx
import jsonschema

from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest, GenerationResult

_HTTP_ERROR_STATUS = 400


class AsyncLlamaCppGenerationAdapter:
    """Call a local llama.cpp router with ``response_format`` json_schema.

    Attributes:
        _base_url (str): Router root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.AsyncClient | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.

    Examples:
        ```python
        from typevet.adapters.outbound.async_llama_cpp import (
            AsyncLlamaCppGenerationAdapter,
        )

        AsyncLlamaCppGenerationAdapter(base_url="http://127.0.0.1:8090")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        *,
        timeout: float = 300.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            base_url: llama.cpp OpenAI-compat base URL.
            timeout: Request timeout in seconds.
            client: Optional shared httpx async client (tests inject a fake).
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

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
        """POST chat completions with a JSON Schema response format.

        Args:
            request: Prompt, schema and model alias on the router.

        Returns:
            Validated structured value.

        Raises:
            GenerationError: On HTTP or JSON parse failure.
            SchemaValidationError: When the payload fails the schema (fail-fast).
        """
        schema_obj = dict(request.schema)
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "typevet_result",
                    "schema": schema_obj,
                },
            },
        }
        url = urljoin(self._base_url, "v1/chat/completions")
        client = await self._ensure_client()
        try:
            response = await client.post(url, json=body)
        except httpx.HTTPError as exc:
            msg = f"llama.cpp request failed: {exc}"
            raise GenerationError(msg) from exc

        if response.status_code >= _HTTP_ERROR_STATUS:
            msg = f"llama.cpp HTTP {response.status_code}: {response.text[:500]}"
            raise GenerationError(msg)

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            msg = "llama.cpp returned non-JSON HTTP body"
            raise GenerationError(msg) from exc

        raw_text = LlamaCppGenerationAdapter._extract_content(payload)
        try:
            value = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = "model content was not valid JSON"
            raise GenerationError(msg) from exc

        if not isinstance(value, dict):
            msg = "model JSON root must be an object"
            raise SchemaValidationError(msg, payload=value)

        try:
            jsonschema.validate(instance=value, schema=schema_obj)
        except jsonschema.ValidationError as exc:
            raise SchemaValidationError(
                f"output failed schema: {exc.message}",
                payload=value,
            ) from exc

        return GenerationResult(
            value=value,
            model=request.model,
            raw_text=raw_text,
        )

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.AsyncClient``.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client
