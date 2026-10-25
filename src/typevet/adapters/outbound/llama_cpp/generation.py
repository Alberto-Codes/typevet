"""llama.cpp OpenAI-compat adapter for constrained JSON Schema generation.

Before the request, ``chat_completion.check_request_schema`` rejects a
malformed schema with ``ValueError``; no request is sent.

With ``autoload=False``, the request carries ``?autoload=false``, and a router
that another model holds raises ``BackendBusyError`` (#461).

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter

    with LlamaCppGenerationAdapter() as port:
        pass  # call port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.fake][]: Offline fake for tests
    - [typevet.adapters.outbound.chat_completion][]: Content and schema checks
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: Shared HTTP error mapping
    - [typevet.domain.errors][]: TransportError, BackendHttpError,
      GenerationUnsupportedCapabilityError
    - [typevet.domain.models][]: GenerationRequest
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.chat_completion import (
    check_request_schema,
    extract_content,
    validated_value,
)
from typevet.adapters.outbound.llama_cpp.autoload import send_with_autoload
from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    map_transport_error,
    parse_json_response,
)
from typevet.domain.errors import GenerationUnsupportedCapabilityError
from typevet.domain.models import GenerationRequest, GenerationResult


class LlamaCppGenerationAdapter:
    """Call a local llama.cpp router with ``response_format`` json_schema.

    The adapter sends text only. It refuses a request with images before any
    HTTP call and never drops an image silently.

    Attributes:
        _base_url (str): Router root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.
        _autoload (bool): Whether the router may load the model freely.

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
        autoload: bool = True,
    ) -> None:
        """Create the adapter.

        Args:
            base_url: llama.cpp OpenAI-compat base URL.
            timeout: Request timeout in seconds.
            client: Optional shared httpx client (tests inject a fake).
            autoload: When ``False``, send ``?autoload=false`` and raise
                ``BackendBusyError`` rather than evict another model (#461).
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._autoload = autoload

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

        The body sets ``chat_template_kwargs`` ``enable_thinking`` to the JSON
        boolean ``false``, as the vLLM adapters do.

        ``chat_completion.extract_content`` reads the reply and
        ``chat_completion.validated_value`` checks it against the schema.

        Args:
            request: Prompt, schema and model alias on the router.

        Returns:
            Validated structured value.

        Raises:
            TransportError: When the HTTP client fails before a response.
            BackendHttpError: When llama.cpp returns HTTP status 400 or above.
            BackendBusyError: With ``autoload`` off, when another model holds
                the last router slot.
            GenerationError: On other parse or response-shape failure.
            GenerationUnsupportedCapabilityError: When the request carries
                images; no HTTP call is made.
            SchemaValidationError: When the payload is non-finite or fails schema.
            ValueError: When the request schema is not a valid JSON Schema.
                No request is sent.
        """
        self._reject_media(request)
        schema_obj = dict(request.schema)
        check_request_schema(schema_obj)
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
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = urljoin(self._base_url, "v1/chat/completions")
        client = self._ensure_client()

        def send(params: Mapping[str, str]) -> httpx.Response:
            """POST the body once with ``params`` as extra query parameters.

            Returns:
                The router response, with any status.

            Raises:
                TransportError: When the HTTP client fails before a response.
            """
            try:
                return client.post(url, json=body, params=params)
            except httpx.HTTPError as exc:
                raise map_transport_error(exc) from exc

        response = send_with_autoload(
            send,
            client=client,
            base_url=self._base_url,
            model=request.model,
            autoload=self._autoload,
        )

        ensure_success_status(response)
        payload = parse_json_response(response)

        raw_text = extract_content(payload, "llama.cpp")
        value = validated_value(raw_text, schema_obj)
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
    def _reject_media(request: GenerationRequest) -> None:
        """Refuse a request with images, which this adapter does not send.

        Args:
            request: The generation ask to check before any HTTP call.

        Raises:
            GenerationUnsupportedCapabilityError: When ``request.media`` is
                not empty.
        """
        if request.media:
            msg = (
                "llama.cpp generation adapters do not send images; "
                f"request has {len(request.media)} image(s)"
            )
            raise GenerationUnsupportedCapabilityError(msg)
