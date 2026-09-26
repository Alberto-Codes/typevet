"""Domain failures for typed generation.

Examples:
    ```python
    from typevet.domain.errors import SchemaValidationError

    err = SchemaValidationError("bad field", payload={"x": 1})
    assert err.payload == {"x": 1}
    ```

See Also:
    - [typevet.domain.models][]: Request and result types
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
