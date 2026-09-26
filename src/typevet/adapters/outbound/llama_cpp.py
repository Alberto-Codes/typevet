"""llama.cpp OpenAI-compat adapter for constrained JSON Schema generation.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter

    with LlamaCppGenerationAdapter() as port:
        pass  # call port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.fake][]: Offline fake for tests
    - [typevet.adapters.outbound.generation_finite][]: Non-finite float guard
    - [typevet.adapters.outbound.llama_cpp_http][]: Shared HTTP error mapping
    - [typevet.domain.errors][]: TransportError, BackendHttpError
    - [typevet.domain.models][]: GenerationRequest
"""

from __future__ import annotations

import json
from typing import Any, Self
from urllib.parse import urljoin

import httpx
import jsonschema

from typevet.adapters.outbound.generation_finite import reject_non_finite_numbers
from typevet.adapters.outbound.llama_cpp_http import (
    ensure_success_status,
    map_transport_error,
    parse_json_response,
)
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest, GenerationResult


class LlamaCppGenerationAdapter:
    """Call a local llama.cpp router with ``response_format`` json_schema.

    Attributes:
        _base_url (str): Router root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.

    Examples:
        ```python
        from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter

        LlamaCppGenerationAdapter(base_url="http://127.0.0.1:8090")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Create the adapter.

        Args:
            base_url: llama.cpp OpenAI-compat base URL.
            timeout: Request timeout in seconds.
            client: Optional shared httpx client (tests inject a fake).
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        """Close the owned HTTP client when the adapter created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        """Enter a context that closes the owned client on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close resources."""
        self.close()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """POST chat completions with a JSON Schema response format.

        Args:
            request: Prompt, schema and model alias on the router.

        Returns:
            Validated structured value.

        Raises:
            TransportError: When the HTTP client fails before a response.
            BackendHttpError: When llama.cpp returns HTTP status 400 or above.
            GenerationError: On other parse or response-shape failure.
            SchemaValidationError: When the payload is non-finite or fails schema.
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
        client = self._ensure_client()
        try:
            response = client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise map_transport_error(exc) from exc

        ensure_success_status(response)
        payload = parse_json_response(response)

        raw_text = self._extract_content(payload)
        try:
            value = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = "model content was not valid JSON"
            raise GenerationError(msg) from exc

        if not isinstance(value, dict):
            msg = "model JSON root must be an object"
            raise SchemaValidationError(msg, payload=value)

        reject_non_finite_numbers(value)

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

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    @staticmethod
    def _extract_content(payload: dict[str, Any]) -> str:
        """Pull the assistant message content from a chat completion body.

        Args:
            payload: Parsed chat completion JSON object.

        Returns:
            Non-empty assistant message content string.

        Raises:
            GenerationError: When the payload shape is wrong or content is empty.
        """
        try:
            choices = payload["choices"]
            message = choices[0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = "llama.cpp response missing choices[0].message.content"
            raise GenerationError(msg) from exc
        if not isinstance(content, str) or not content.strip():
            msg = "llama.cpp returned empty message content"
            raise GenerationError(msg)
        return content
