"""vLLM ``/v1/chat/completions`` adapter for constrained JSON Schema generation.

The request asks vLLM for structured output through
``structured_outputs: {"json": <schema>}``. The server applies the served chat
template to one user message and does not start a thinking turn. Before the
request, ``chat_completion.check_request_schema`` rejects a malformed schema
with ``ValueError``; no request is sent. The adapter makes one POST. It does
not retry and does not fall back to unconstrained generation. Status,
transport and non-JSON body failures go through ``vllm_http``. A request with
images sends the content as ``text`` and ``image_url`` blocks from
``vllm_content``; a text request sends a string.

Examples:
    ```python
    from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter

    with VllmGenerationAdapter("http://127.0.0.1:8000") as port:
        pass  # call port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp counterpart
    - [typevet.adapters.outbound.chat_completion][]: Content and schema checks
    - [typevet.adapters.outbound.vllm_content][]: Image content blocks
    - [typevet.adapters.outbound.vllm_http][]: Shared vLLM HTTP error mapping
    - [typevet.domain.errors][]: GenerationError, SchemaValidationError
    - [typevet.domain.models][]: GenerationRequest
"""

from __future__ import annotations

from typing import Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.chat_completion import (
    check_request_schema,
    extract_content,
    validated_value,
)
from typevet.adapters.outbound.vllm_content import content_blocks
from typevet.adapters.outbound.vllm_http import post_json
from typevet.domain.models import GenerationRequest, GenerationResult


class VllmGenerationAdapter:
    """Call a vLLM server with ``structured_outputs`` JSON Schema constraints.

    Attributes:
        _base_url (str): Server root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.

    Examples:
        ```python
        VllmGenerationAdapter(base_url="http://127.0.0.1:8000")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Create the generation adapter.

        Args:
            base_url: vLLM server root URL.
            timeout: Request timeout in seconds for an owned client.
            client: Optional caller-built httpx client, for example one that
                carries authentication headers. The adapter does not close it.
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
        """POST one constrained chat completion and validate the result.

        ``generation_body`` builds the request. ``chat_completion`` helpers
        read the reply and check it against the schema.

        Args:
            request: Prompt, schema, served model name and optional images.

        Returns:
            Validated structured value.

        Raises:
            TransportError: When the HTTP client fails before a response.
            BackendHttpError: When vLLM returns HTTP status 400 or above.
            GenerationError: On a non-JSON body, bad shape or non-JSON content.
            SchemaValidationError: When the value is not an object, is
                non-finite or fails the schema.
            ValueError: When the request schema is not a valid JSON Schema.
                No request is sent.
        """
        schema_obj = dict(request.schema)
        check_request_schema(schema_obj)
        body = generation_body(request, schema_obj)
        url = urljoin(self._base_url, "v1/chat/completions")
        payload = post_json(self._ensure_client(), url, body)
        raw_text = extract_content(payload, "vLLM")
        value = validated_value(raw_text, schema_obj)
        return GenerationResult(value=value, model=request.model, raw_text=raw_text)

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client


def generation_body(
    request: GenerationRequest, schema: dict[str, Any]
) -> dict[str, Any]:
    """Build the vLLM structured-output chat completion body.

    The sync and async adapters both send this body.

    Args:
        request: Prompt, served model name and optional images.
        schema: JSON Schema object from the request.

    Returns:
        JSON body with the model, one user message, ``temperature`` 0, the
        ``structured_outputs`` schema, ``add_generation_prompt`` and
        ``chat_template_kwargs`` that turn thinking off.
    """
    content: str | list[dict[str, Any]] = request.prompt
    if request.media:
        content = content_blocks(request.prompt, request.media)
    return {
        "model": request.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "structured_outputs": {"json": schema},
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
