"""Read a chat completion reply and validate its content against a schema.

The llama.cpp and vLLM generation adapters, sync and async, use these two
steps after the HTTP call. ``extract_content`` names the backend in its error
messages. ``validated_value`` messages do not name a backend.

Examples:
    ```python
    from typevet.adapters.outbound.chat_completion import (
        extract_content,
        validated_value,
    )

    payload = {"choices": [{"message": {"content": '{"n": 1}'}}]}
    raw = extract_content(payload, "vLLM")
    validated_value(raw, {"type": "object"})
    ```

See Also:
    - [typevet.adapters.outbound.generation_finite][]: Non-finite float guard
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp consumer
    - [typevet.adapters.outbound.vllm_generation][]: vLLM consumer
    - [typevet.domain.errors][]: GenerationError, SchemaValidationError
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from typevet.adapters.outbound.generation_finite import reject_non_finite_numbers
from typevet.domain.errors import GenerationError, SchemaValidationError


def extract_content(payload: Any, backend: str) -> str:
    """Pull the assistant message content from a chat completion body.

    Args:
        payload: Parsed chat completion JSON value.
        backend: Backend name for error messages, such as ``"vLLM"``.

    Returns:
        Non-empty assistant message content string.

    Raises:
        GenerationError: When the payload shape is wrong or content is empty.
    """
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = f"{backend} response missing choices[0].message.content"
        raise GenerationError(msg) from exc
    if not isinstance(content, str) or not content.strip():
        msg = f"{backend} returned empty message content"
        raise GenerationError(msg)
    return content


def validated_value(raw_text: str, schema: dict[str, Any]) -> dict[str, Any]:
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
