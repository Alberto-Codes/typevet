"""Domain failures for typed generation.

Examples:
    ```python
    from typevet.domain.errors import SchemaValidationError

    err = SchemaValidationError("bad field", payload={"x": 1})
    assert err.payload == {"x": 1}
    ```

See Also:
    - [typevet.domain.models][]: Request and result types
    - [typevet.adapters.outbound.llama_cpp_http][]: httpx to domain error mapping

Attributes:
    BackendHttpError (type): llama.cpp HTTP status 400 or above.
    GenerationError (type): Base failure for a generation call.
    SchemaValidationError (type): Output failed the requested schema.
    TransportError (type): HTTP client failure before a response.
"""

from __future__ import annotations


class GenerationError(Exception):
    """A typed generation call failed before a valid result existed.

    Examples:
        ```python
        from typevet.domain.errors import GenerationError

        raise GenerationError("transport failed")
        ```
    """


class TransportError(GenerationError):
    """The HTTP client failed before a usable llama.cpp response arrived.

    Attributes:
        status_code (None): Always ``None`` for transport failures.
        body_snippet (None): Always ``None`` when no response body was read.

    Examples:
        ```python
        from typevet.domain.errors import TransportError

        raise TransportError("llama.cpp request failed: connection refused")
        ```
    """

    status_code: None
    body_snippet: None

    def __init__(self, message: str) -> None:
        """Record a transport failure message.

        Args:
            message: Human-readable summary of the client failure.
        """
        super().__init__(message)
        self.status_code = None
        self.body_snippet = None


class BackendHttpError(GenerationError):
    """llama.cpp returned an HTTP error status (400 or above).

    Attributes:
        status_code (int): HTTP status from the router.
        body_snippet (str): Truncated response body text for diagnostics.

    Examples:
        ```python
        from typevet.domain.errors import BackendHttpError

        raise BackendHttpError(
            "llama.cpp HTTP 500: internal",
            status_code=500,
            body_snippet="internal",
        )
        ```
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body_snippet: str,
    ) -> None:
        """Record an HTTP error status and response snippet.

        Args:
            message: Human-readable summary including status and snippet.
            status_code: HTTP status from llama.cpp.
            body_snippet: Truncated response body text.
        """
        super().__init__(message)
        self.status_code = status_code
        self.body_snippet = body_snippet


class SchemaValidationError(GenerationError):
    """The model output did not validate against the requested schema.

    Attributes:
        payload (object | None): Parsed value that failed validation, when set.
        args (tuple): Standard exception args (message first).

    Examples:
        ```python
        from typevet.domain.errors import SchemaValidationError

        raise SchemaValidationError("missing key", payload={})
        ```
    """

    def __init__(self, message: str, *, payload: object | None = None) -> None:
        """Record the failure and the optional rejected payload.

        Args:
            message: Human-readable validation summary.
            payload: Parsed value that failed validation, when available.
        """
        super().__init__(message)
        self.payload = payload
