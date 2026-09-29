"""Check a request schema, read a chat completion reply and validate it.

The llama.cpp and vLLM generation adapters, sync and async, call
``check_request_schema`` before the HTTP call and the other two steps after
it. ``extract_content`` names the backend in its error messages.
``validated_value`` and ``check_request_schema`` messages do not name a
backend.

Attributes:
    SCHEMA_CHECK_CACHE_SIZE (int): Most passing schemas that
        ``check_request_schema`` remembers.

Examples:
    ```python
    from typevet.adapters.outbound.chat_completion import (
        check_request_schema,
        extract_content,
        validated_value,
    )

    check_request_schema({"type": "object"})
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

import functools
import json
from collections.abc import Mapping
from typing import Any

import jsonschema

from typevet.adapters.outbound.generation_finite import reject_non_finite_numbers
from typevet.domain.errors import GenerationError, SchemaValidationError

SCHEMA_CHECK_CACHE_SIZE = 256
_SCHEMA_ERROR_PREFIX = "schema is not a valid JSON Schema: "


def check_request_schema(schema: Mapping[str, Any]) -> None:
    """Reject a request schema that is not a valid JSON Schema.

    The Draft 2020-12 meta-schema check runs once per distinct schema. The
    cache key is the schema as JSON with sorted keys, so key order does not
    matter. The cache holds at most ``SCHEMA_CHECK_CACHE_SIZE`` passing
    schemas and is safe to share between threads. A failing schema is not
    cached. A schema that ``json.dumps`` cannot encode (a set, keys of
    mixed types, ``NaN``) cannot be sent either, so it fails the check.

    Args:
        schema: JSON Schema object from the request.

    Raises:
        ValueError: When the schema fails the meta-schema or is not
            JSON-encodable. The message starts with
            ``schema is not a valid JSON Schema:``.
    """
    try:
        key = json.dumps(dict(schema), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        msg = f"{_SCHEMA_ERROR_PREFIX}not JSON-encodable ({exc})"
        raise ValueError(msg) from None
    _check_schema_json(key)


@functools.lru_cache(maxsize=SCHEMA_CHECK_CACHE_SIZE)
def _check_schema_json(key: str) -> None:
    """Run the meta-schema check on one encoded schema.

    Args:
        key: Schema encoded as JSON with sorted keys.

    Raises:
        ValueError: When the schema fails the Draft 2020-12 meta-schema.
    """
    try:
        jsonschema.Draft202012Validator.check_schema(json.loads(key))
    except jsonschema.SchemaError as exc:
        msg = f"{_SCHEMA_ERROR_PREFIX}{exc.message}"
        raise ValueError(msg) from None


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
