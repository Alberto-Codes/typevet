"""vLLM ``/v1/chat/completions`` adapter for constrained JSON Schema generation.

The request asks vLLM for structured output through
``structured_outputs: {"json": <schema>}``. The server applies the served chat
template to one user message and does not start a thinking turn. The adapter
makes one POST. It does not retry and does not fall back to unconstrained
generation. Status, transport and non-JSON body failures go through
``vllm_http``. A request with images sends the content as ``text`` and
``image_url`` blocks from ``vllm_content``; a text request sends a string.

Examples:
    ```python
    from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter

    with VllmGenerationAdapter("http://127.0.0.1:8000") as port:
        pass  # call port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp counterpart
    - [typevet.adapters.outbound.generation_finite][]: Non-finite float guard
    - [typevet.adapters.outbound.vllm_content][]: Image content blocks
    - [typevet.adapters.outbound.vllm_http][]: Shared vLLM HTTP error mapping
    - [typevet.domain.errors][]: GenerationError, SchemaValidationError
    - [typevet.domain.models][]: GenerationRequest
"""

from __future__ import annotations

import json
from typing import Any, Self
from urllib.parse import urljoin

import httpx
import jsonschema

from typevet.adapters.outbound.generation_finite import reject_non_finite_numbers
from typevet.adapters.outbound.vllm_content import content_blocks
from typevet.adapters.outbound.vllm_http import post_json
from typevet.domain.errors import GenerationError, SchemaValidationError
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
        """
        schema_obj = dict(request.schema)
        content: str | list[dict[str, Any]] = request.prompt
        if request.media:
            content = content_blocks(request.prompt, request.media)
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "structured_outputs": {"json": schema_obj},
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = urljoin(self._base_url, "v1/chat/completions")
        payload = post_json(self._ensure_client(), url, body)
        raw_text = _extract_content(payload)
        value = _validated_value(raw_text, schema_obj)
        return GenerationResult(value=value, model=request.model, raw_text=raw_text)

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client


def _extract_content(payload: Any) -> str:
    """Pull the assistant message content from a chat completion body.

    Args:
        payload: Parsed chat completion JSON value.

    Returns:
        Non-empty assistant message content string.

    Raises:
        GenerationError: When the payload shape is wrong or content is empty.
    """
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "vLLM response missing choices[0].message.content"
        raise GenerationError(msg) from exc
    if not isinstance(content, str) or not content.strip():
        msg = "vLLM returned empty message content"
        raise GenerationError(msg)
    return content


def _validated_value(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Parse model content and validate it against the request schema.

    Args:
        raw_text: Assistant message content.
        schema: JSON Schema object from the request.

    Returns:
        Parsed JSON object that satisfies the schema.

    Raises:
        GenerationError: When the content is not valid JSON.
        SchemaValidationError: When the root is not an object, a number is
            non-finite or the value fails the schema.
    """
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
        jsonschema.validate(instance=value, schema=schema)
    except jsonschema.ValidationError as exc:
        raise SchemaValidationError(
            f"output failed schema: {exc.message}",
            payload=value,
        ) from exc
    return value
