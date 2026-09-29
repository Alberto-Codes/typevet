"""HTTP error constants that every backend's error mapping shares.

A status at or above ``HTTP_ERROR_STATUS`` is a backend failure. The error
keeps at most ``BODY_SNIPPET_MAX`` characters of the response body.

Attributes:
    HTTP_ERROR_STATUS (int): Lowest status that the adapters treat as failure.
    BODY_SNIPPET_MAX (int): Maximum characters of body text in an error.

Examples:
    ```python
    from typevet.adapters.outbound.http_errors import body_snippet

    body_snippet("x" * 600)  # first 500 characters
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: llama.cpp error mapping
    - [typevet.adapters.outbound.vllm_http][]: vLLM error mapping
    - [typevet.domain.errors][]: BackendHttpError
"""

from __future__ import annotations

HTTP_ERROR_STATUS = 400
BODY_SNIPPET_MAX = 500


def body_snippet(text: str) -> str:
    """Return a bounded snippet of response text for error metadata.

    Args:
        text: Full HTTP response body text.

    Returns:
        At most ``BODY_SNIPPET_MAX`` characters from ``text``.
    """
    return text[:BODY_SNIPPET_MAX]
